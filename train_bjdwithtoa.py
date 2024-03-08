import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import random
import numpy as np
import argparse
from tqdm import tqdm
from glob import glob

import torch
import torch.utils.checkpoint
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from monai.utils import set_determinism
from monai.data import DataLoader, DistributedSampler, decollate_batch, Dataset
from monai.transforms import (
    ScaleIntensity,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandAxisFlipd,
)

from diffusion_tool.schedulers import DDPMScheduler, DDIMScheduler
from diffusion_tool.nets import DiffusionModelUNet
from networks.TextAdapter import MultiTextAdapter

from data.diff_datamodule import CFPDataset

from transformers import CLIPTextModel, CLIPTokenizer
from diffusion_tool.inferers import DiffusionInferer

#for experiment reproducibility
seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_determinism(seed=12345)

@record
class Trainer():
    def __init__(self, args) -> None:
        #initialize ddp
        self.init_ddp()
        self.args = args
        self.tasks_name = ['mnms', 'Fundus', 'Prostate']
        self.val_prompt_list = ['A magnetic resonance of cardiac with a left ventricle, a myocardium and a right ventricle',
                        'A fundus photography with a optic cup and a optic disc',
                        'A magnetic resonance of a prostate']
        self.current_iteration = args.start_iter
        base_path = args.out_dir
        
        self.saved_model_path = os.path.join(base_path, 'saved_model')
        self.fake_dataset_path = os.path.join(base_path, 'fake_dataset')
        os.system('cp train_bjdwithtoa.py ' + base_path + '/' + 'train_bjdwithtoa.py.backup')
        os.system('cp datamodule.py ' + base_path + '/' + 'datamodule.py.backup')
        
    
    def init_ddp(self, ):
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank != 0:
            f = open(os.devnull, "w")
            sys.stdout = sys.stderr = f
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        self.device = device
        
    
    def save_img_np(self, img, path):
        np.savez_compressed(path, img.astype(np.uint8))
        
    def save_mask_np(self, mask, path):
        np.savez_compressed(path, mask)
    
    def get_replay_transform(self,):
        input_transforms = Compose(
                [
                    LoadImaged(keys=["image", "mask"], image_only=False),
                    EnsureChannelFirstd(keys=["mask"]),
                    ScaleIntensityd(keys=["image", "mask"], minv=-1, maxv=1),
                    RandAxisFlipd(keys=['image','mask'], prob=0.5, lazy=False),
                ]
            )
        return input_transforms
    
    def train(self, ):
        args = self.args
        device = self.device
        
        print("creating CLIP model...")
        tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_clip, subfolder="tokenizer", revision=None
            )
        text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_clip, subfolder="text_encoder", revision=None
            ).to(device)
        text_encoder.requires_grad_(False)
        text_adapter = MultiTextAdapter(num_classes=3).to(device)
        
        ddpm_scheduler = DDPMScheduler(schedule="scaled_linear_beta", num_train_timesteps=1000,
                                       beta_start=0.0015, beta_end=0.0205)
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000, schedule="scaled_linear_beta", 
            beta_start=0.0015, beta_end=0.0205, clip_sample=True
        )
        ddim_scheduler.set_timesteps(num_inference_steps=args.ddim_steps)
        inferer = DiffusionInferer(ddim_scheduler)
        
        l1loss = nn.L1Loss()
        
        post_transform = []
        for task_label, task_name in enumerate(self.tasks_name):
            if task_name == 'mnms':
                post_transform.append(ScaleIntensity(minv=0, maxv=3))
            elif task_name == 'Fundus':
                post_transform.append(ScaleIntensity(minv=0, maxv=2))
            else:
                post_transform.append(ScaleIntensity(minv=0, maxv=1))
        
        cond_list = [0, 1]
        datasets = CFPDataset(args.data_dir, task_name=self.tasks_name, mode='seperate').get_dataset()
        for task_label, task_name in enumerate(self.tasks_name):
            print('create diffusion model ... ')
            if task_label == 0:
                model = DiffusionModelUNet(spatial_dims=2,
                                        in_channels=4,
                                        out_channels=4,
                                        num_res_blocks=2,
                                        num_channels=[128,128,256,256,512,512],
                                        attention_levels=[False, False, False, True,True,True],
                                        with_conditioning=True,
                                        cross_attention_dim=768,
                                        num_head_channels=64)
                model.to(device)
            
                #initialize model
                model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
            else:
                model = DiffusionModelUNet(spatial_dims=2,
                                        in_channels=4,
                                        out_channels=4,
                                        num_res_blocks=2,
                                        num_channels=[128,128,256,256,512,512],
                                        attention_levels=[False, False, False, True,True,True],
                                        with_conditioning=True,
                                        cross_attention_dim=768,
                                        num_head_channels=64)
                model.to(device)
            
                #initialize model
                model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
                # load previous model
                model_ckpt_previous = os.path.join(self.saved_model_path, 'task--{:}'.format(task_label-1), '{:}.pth'.format(args.total_iterations))
                checkpoint = torch.load(model_ckpt_previous, map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint['model'])
                del checkpoint, model_ckpt_previous
                
            opt = optim.AdamW(model.parameters(), lr=args.lr)
            scaler = torch.cuda.amp.GradScaler()
            
            
            print("creating data loader...")
            if task_label == 0:
                train_sampler = DistributedSampler(dataset=datasets[task_label], even_divisible=True, shuffle=True)
                dataloader = DataLoader(datasets[task_label], 
                                    batch_size=args.batch_size, 
                                    num_workers=4,
                                    shuffle=False,
                                    drop_last=True,
                                    pin_memory=True,
                                    sampler=train_sampler)
                
                dataloader_iter = iter(dataloader)
            else:
                train_sampler = DistributedSampler(dataset=datasets[task_label], even_divisible=True, shuffle=True)
                dataloader = DataLoader(datasets[task_label], 
                                    batch_size=args.batch_size // 2, 
                                    num_workers=4,
                                    shuffle=False,
                                    drop_last=True,
                                    pin_memory=True,
                                    sampler=train_sampler)
                
                dataloader_iter = iter(dataloader)
                
                fake_loader_list = []
                fake_loader_iter_list = []
                for pre_task_index in range(task_label):
                    fake_imgs_fold = os.path.join(self.fake_dataset_path, self.tasks_name[pre_task_index], "imagesTr")
                    fake_masks_fold = os.path.join(self.fake_dataset_path, self.tasks_name[pre_task_index], "labelsTr")
                    
                    imgs_path = sorted(glob(os.path.join(fake_imgs_fold, "*.npz")))
                    masks_path = sorted(glob(os.path.join(fake_masks_fold, "*.npz")))
                    
                    fake_dicts = [{"image": img, "mask": mask, 'task_label': pre_task_index, 'prompt': self.val_prompt_list[pre_task_index]} for img, mask in zip(imgs_path, masks_path)]
                    fake_dataset = Dataset(data=fake_dicts, transform=self.get_replay_transform())
                    fake_train_sampler = DistributedSampler(dataset=fake_dataset, even_divisible=True, shuffle=True)
                    fake_loader = DataLoader(fake_dataset, 
                                        batch_size=args.batch_size // (2 * task_label),
                                        shuffle=False,
                                        drop_last=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        sampler=fake_train_sampler,)
                    
                    fake_loader_list.append(fake_loader)
                
                for fake_loader in fake_loader_list:
                    fake_loader_iter_list.append(iter(fake_loader))
            epochs = 0
            with tqdm(total=args.total_iterations) as pbar:
                for iteration in range(self.current_iteration, args.total_iterations+1):
                    if iteration % 100 == 0 and iteration > 0:
                        pbar.update(100)
                    if task_label == 0:
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            epochs += 1
                            train_sampler.set_epoch(epochs)
                            dataloader_iter = iter(dataloader)
                            batch = next(dataloader_iter)
                    else:
                        batch_previous_list = []
                        try:
                            batch_current = next(dataloader_iter)
                        except StopIteration:
                            epochs += 1
                            train_sampler.set_epoch(epochs)
                            dataloader_iter = iter(dataloader)
                            batch_current = next(dataloader_iter)
                        try:
                            for fake_loader_iter in fake_loader_iter_list:
                                batch_previous_list.append(next(fake_loader_iter))
                        except StopIteration:
                            fake_loader_iter_list = []
                            for fake_loader in fake_loader_list:
                                fake_loader_iter_list.append(iter(fake_loader))
                            for fake_loader_iter in fake_loader_iter_list:
                                batch_previous_list.append(next(fake_loader_iter))
                                
                                
                    with torch.cuda.amp.autocast(enabled=True):
                        if task_label == 0:
                            img = batch['image'].to(device)
                            mask = batch['mask'].to(device)
                            task_label_batch = int(batch['task_label'][0])
                            timesteps = torch.randint(0, ddpm_scheduler.num_train_timesteps, (img.shape[0],), device=device).long()
                            text_prompt = batch['prompt']
                            text_input = tokenizer(
                                    text_prompt,
                                    padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
                            encoder_hidden_states = text_encoder(text_input.input_ids.to(device))[0]
                            task_embedding = text_adapter(encoder_hidden_states, task_label_batch)
                        else:
                            img_current = batch_current['image'].to(device)
                            mask_current = batch_current['mask'].to(device)
                            task_label_batch = int(batch_current['task_label'][0])
                            text_prompt = batch_current['prompt']
                            text_input = tokenizer(
                                    text_prompt,
                                    padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
                            encoder_hidden_states = text_encoder(text_input.input_ids.to(device))[0]
                            task_embedding_current = text_adapter(encoder_hidden_states, task_label_batch)
                            img_previous_list = []
                            mask_previous_list = []
                            task_embedding_list = []
                            for batch_previous in batch_previous_list:
                                img_previous_list.append(batch_previous['image'].to(device))
                                mask_previous_list.append(batch_previous['mask'].to(device))
                                task_label_previous = int(batch_previous['task_label'][0])
                                text_prompt_previous = batch_previous['prompt']
                                text_input_previous = tokenizer(
                                        text_prompt_previous,
                                        padding="max_length",
                                        max_length=tokenizer.model_max_length,
                                        truncation=True,
                                        return_tensors="pt",
                                    )
                                encoder_hidden_states_previous = text_encoder(text_input_previous.input_ids.to(device))[0]
                                task_embedding_list.append(text_adapter(encoder_hidden_states_previous, task_label_previous))
                                
                            img_previous = torch.cat(img_previous_list, dim=0)
                            mask_previous = torch.cat(mask_previous_list, dim=0)
                            task_embedding_previous = torch.cat(task_embedding_list, dim=0)
                            del img_previous_list, mask_previous_list, task_embedding_list
                                
                            img = torch.cat([img_current, img_previous], dim=0)
                            mask = torch.cat([mask_current, mask_previous], dim=0)
                            task_embedding = torch.cat([task_embedding_current, task_embedding_previous], dim=0)
                                
                        #for joint denoising
                        noise_1 = torch.randn(args.batch_size, 4, 256, 256).to(device)
                        img_mask_uncond = torch.cat([img, mask], dim=1)
                        img_mask_uncond_noisy = ddpm_scheduler.add_noise(original_samples=img_mask_uncond, noise=noise_1, timesteps=timesteps)
                        target = noise_1
                        
                        #for conditional image-mask denoising
                        if cond_list[iteration % 2] == 0:
                            noise_2 = torch.randn(args.batch_size, 1, 256, 256).to(device)
                            mask_noisy = ddpm_scheduler.add_noise(original_samples=mask, noise=noise_2, timesteps=timesteps)
                            cond_img_mask_noisy = torch.cat([img, mask_noisy], dim=1)
                            cond_target = torch.cat((torch.zeros((args.batch_size, 3, 256, 256)).to(device), noise_2), dim=1)
                        
                        if cond_list[iteration % 2] == 1:
                            noise_3 = torch.randn(args.batch_size, 3, 256, 256).to(device)
                            img_noisy = ddpm_scheduler.add_noise(original_samples=img, noise=noise_3, timesteps=timesteps)
                            cond_img_mask_noisy = torch.cat([img_noisy, mask], dim=1)
                            cond_target = torch.cat((noise_3, torch.zeros((args.batch_size, 1 ,256, 256)).to(device)), dim=1)
                        
                        noise_pred = model(img_mask_uncond_noisy, timesteps, task_embedding)
                        cond_noise_pred = model(cond_img_mask_noisy, timesteps, task_embedding)
                    
                        loss = l1loss(noise_pred, target) + l1loss(cond_noise_pred, cond_target)
                    
                        #update model
                        opt.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        scaler.step(opt)
                        scaler.update()
                        
                    
                    if iteration % args.save_interval == 0:
                        lr = opt.param_groups[0]['lr']
                        print("train loss: %.5f lr: %.6f"%(loss.item(), lr))
                        
                        
                    if iteration % (args.save_interval*100) == 0 or iteration == args.total_iterations and dist.get_rank() == 0:
                        saved_path = os.path.join(self.saved_model_path, "task--%d"%task_label)
                        os.makedirs(saved_path, exist_ok=True)
                        torch.save({'model':model.state_dict(),
                                    'adapter':text_adapter.state_dict(),
                                    }, 
                                    os.path.join(saved_path, '%d.pth'%iteration)
                                    )
                        
            #ddim sampling for next learning round
            print("generating each task data for next learning round .... ")
            for generate_task_label in range(task_label+1):
                with torch.no_grad():
                    mask_save_path = os.path.join(self.fake_dataset_path, self.tasks_name[generate_task_label], 'labelsTr')
                    img_save_path = os.path.join(self.fake_dataset_path, self.tasks_name[generate_task_label], 'imagesTr')
                    os.makedirs(mask_save_path, exist_ok=True)
                    os.makedirs(img_save_path, exist_ok=True)
                    img_post_transform = ScaleIntensity(minv=0, maxv=255)
                    prompt = self.val_prompt_list[generate_task_label]
                    text_input = tokenizer(
                                prompt,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            )
                    encoder_hidden_states = text_encoder(text_input.input_ids.to(device))[0]
                    encoder_hidden_states_batch = encoder_hidden_states.repeat(args.test_batch_size, 1, 1)
                    task_embedding_test = text_adapter(encoder_hidden_states_batch, generate_task_label)
        
                    current_gen_num = 0
                    pbar = tqdm(total=args.n_sample)
                    while(current_gen_num < args.n_sample):
                        with torch.cuda.amp.autocast():
                            noise = torch.randn(args.test_batch_size, 4, 256, 256).to(device)
                            image_concate_mask = inferer.sample(input_noise=noise, 
                                                                diffusion_model=model, 
                                                                conditioning=task_embedding_test)
                            image = image_concate_mask[:, :3, :, :]
                            mask = image_concate_mask[:, 3:, :, :]
                            mask_post_transform = torch.stack([post_transform[task_label](i) for i in decollate_batch(mask)])
                            img_255 = torch.stack([img_post_transform(i) for i in decollate_batch(image)])
                            mask_discrete = torch.round(mask_post_transform).squeeze()
                        
                        
                        for batch_idx in range(args.test_batch_size):
                            mask_to_save= mask_discrete[batch_idx].squeeze().cpu().numpy().astype(np.uint8)
                            img_to_save = img_255[batch_idx].squeeze().cpu().numpy()
                            
                            fake_img_path = os.path.join(img_save_path, '{:}.npz'.format(current_gen_num + 1))
                            fake_mask_path = os.path.join(mask_save_path, '{:}.npz'.format(current_gen_num + 1))
                            self.save_img_np(img_to_save, fake_img_path)
                            self.save_mask_np(mask_to_save, fake_mask_path)
                                
                            current_gen_num += 1
                            pbar.update(1)
                            if current_gen_num >= args.n_sample:
                                break
                    
        dist.destroy_process_group()
                            
                   

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MultiTasksDataset', 
                        help='path of resource dataset')
    parser.add_argument('--out_dir', type=str, 
                        default='CGR', help='')
    parser.add_argument('--pretrained_clip', type=str, 
                        default='Pretrained_weights', help='')
    parser.add_argument('--total_iterations', type=int, default=400000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=4, help='mini batch number of images')
    parser.add_argument('--n_sample', type=int, default=3000, help='mini batch number of images')
    parser.add_argument('--test_batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--lr', type=float, default=1e-4, help='init learning rate')
    parser.add_argument('--save_interval', type=int, default=100, help='interval to save model')
    parser.add_argument('--ddim_steps', type=int, default=100, help='number of steps for ddim')
    parser.add_argument("--local_rank", type=int)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    Diffusion_trainer = Trainer(args)
    Diffusion_trainer.train()