import os
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from data.Datamodule import DataModule
from networks.IncrementModel import IncrementModel
from utils.utils import mkdir, tensor2im
import time
from monai.data import DataLoader, decollate_batch, Dataset
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from Trainer.BaseTrainer import BaseTrainer
from utils import tasks
from monai.networks.utils import one_hot
from monai.utils import set_determinism
from glob import glob
import copy
import numpy as np
from tqdm import tqdm
import random

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_determinism(seed=seed)

class ReplayTrainer(BaseTrainer):
    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        parser.add_argument('--fake_path', type=str, 
                            default="", 
                            help='fake dataset path')
        
        return parser

    def __init__(self, opt):
        BaseTrainer.__init__(self, opt)
        self.opt = opt
        os.system('cp Trainer/ReplayTrainer.py ' + self.base_checkpoint_path + '/' + 'ReplayTrainer.py.backup')
             
    def copy_params(self, model):
        flatten = copy.deepcopy(list(p.data for p in model.parameters()))
        return flatten
    
    def load_params(self, model, new_param):
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)
    
    
    def Train(self, ):
        self.device = self.set_device()
        dataset_name = self.opt.exp_name.split("_")[0]
        tasks_dict = tasks.get_task_labels(dataset=dataset_name, mode=self.opt.mode)

        # init tensorboard
        tensorboard_dir = os.path.join(self.base_checkpoint_path, self.opt.mode, "tensorboard")
        writer = SummaryWriter(tensorboard_dir)

        loss_fun = DiceCELoss(softmax=True)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        
        label_current = []
        label_old = []
        for step in range(len(tasks_dict)):
            # set dataset and dataloader
            dataset_path = os.path.join(self.opt.dataroot)
            data = DataModule(dataset_path=dataset_path,
                              current_sites_name=self.sites_name[step],
                              sites_name_all=self.sites_name,
                              dataset_json=self.dataset_json,
                              mode=self.opt.method)
            train_ds = Dataset(data=data.data_dicts, transform=data.train_transform)
            val_ds = Dataset(data=data.val_data_dicts, transform=data.val_transform)
            #set fake_dataset
            if step > 0:
                fake_loader_list = []
                fake_loader_iter_list = []
                for sites_index in range(step):
                    fake_imgs_fold = os.path.join(self.opt.fake_path, self.sites_name[sites_index], "imagesTr")
                    fake_masks_fold = os.path.join(self.opt.fake_path, self.sites_name[sites_index], "labelsTr")
                    
                    imgs_path = sorted(glob(os.path.join(fake_imgs_fold, "*.npz")))
                    masks_path = sorted(glob(os.path.join(fake_masks_fold, "*.npz")))

                    fake_dicts = [{"image": img, "mask": mask} for img, mask in zip(imgs_path, masks_path)]
                    fake_dataset = Dataset(data=fake_dicts, transform=data.replay_train_transform)
                    
                    fake_loader = DataLoader(fake_dataset, 
                                        batch_size=self.opt.batch_size // (2 * step),
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=4,
                                        pin_memory=torch.cuda.is_available(),)
                    fake_loader_list.append(fake_loader)
                
                for fake_loader in fake_loader_list:
                    fake_loader_iter_list.append(iter(fake_loader))
                

            # set label
            
            if self.opt.mode == "class_incremental":
                if step > 0:
                    label_old = label_current
                label_current = label_current + tasks_dict[step]

            # set val_post_process
            val_post_process = self.get_post_process(num_class=len(label_current))
            
            self.grad_score_list = {}
            self.class_socre_list = {}
            if step == 0:
                model = IncrementModel(in_ch=3, out_ch=len(label_current),
                                        backbone=self.opt.model).to(self.device)
            else:
                # load previous step model
                model_old = IncrementModel(in_ch=3, out_ch=len(label_old),
                                            backbone=self.opt.model).to(self.device)
                model_old_step_path = os.path.join(self.base_checkpoint_path, self.opt.mode, "step_{:}".format(step), 
                                                   "model", "model_{:}.pt".format(self.opt.total_iterations))
                model_old.load_state_dict(torch.load(model_old_step_path))

                # load current step model
                model = IncrementModel(in_ch=3, out_ch=len(label_current),
                                        backbone=self.opt.model).to(self.device)
                
                if self.opt.mode == "class_incremental":
                    model.backbone.load_state_dict(model_old.backbone.state_dict(), strict=False)
                    with torch.no_grad():
                        model.classifier.conv.weight[:len(label_old)] = model_old.classifier.conv.weight
                        model.classifier.conv.bias[:len(label_old)] = model_old.classifier.conv.bias
                    model_old.requires_grad_(False)
                    
        
            if step == 0:
                train_loader = DataLoader(train_ds,
                                            batch_size=self.opt.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=4,
                                            pin_memory=torch.cuda.is_available(),
                                            )
            else:
                train_loader = DataLoader(train_ds,
                                        batch_size=self.opt.batch_size // 2,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=4,
                                        pin_memory=torch.cuda.is_available(),
                                        )
            
            val_loader = DataLoader(val_ds,
                                    batch_size=self.opt.val_batch_size,
                                    num_workers=4,
                                    pin_memory=torch.cuda.is_available(),
                                    )
            train_loader_iter = iter(train_loader)
            step_fold_path = os.path.join(self.base_checkpoint_path, self.opt.mode, "step_{:}".format(step + 1))
            log_path = os.path.join(step_fold_path, "log")
            io = self.init_io(log_path)
            io.cprint("{:}/{:} step training".format(step + 1, len(tasks_dict)))
            model_fold_savepath = os.path.join(step_fold_path, 'model')
            mkdir(model_fold_savepath)
            # define optimizer
            optimizer = optim.Adam(model.parameters(), lr=self.opt.lr, weight_decay=3e-5)
            # define scheduler
            if step == 0:
                total_epochs = self.opt.total_iterations // (len(train_ds) // self.opt.batch_size)
            else:
                total_epochs = self.opt.total_iterations // (len(train_ds) // (self.opt.batch_size // 2))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs + 20, eta_min=0.1 * self.opt.lr)
            
            best_metric = 0
            best_metric_iter = 0
            if step == 0:
                with tqdm(total=self.opt.total_iterations) as pbar:
                    for iteration in range(self.opt.start_iteration, self.opt.total_iterations):
                        if iteration % 100 == 0 and iteration > 0:
                                pbar.update(100)
                        
                        model.train()
                        
                        try:
                            batch = next(train_loader_iter)
                        except StopIteration:
                            scheduler.step()
                            train_loader_iter = iter(train_loader)
                            batch = next(train_loader_iter)
                        
                        img = batch["image"].to(self.device)
                        gt = batch["mask"]
                        gt_one_hot = one_hot(gt, num_classes=len(label_current), dim=1).to(self.device)
                        
                        pred = model(img)
                        loss = loss_fun(pred, gt_one_hot)

                        # backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('step_{:}_train_loss'.format(step),
                                                        loss.item(),
                                                        iteration)
                        
                        # print information
                        if iteration % 100 == 0:
                            outstr = 'train-- iteration: %d/%d \n train_loss: %.6f, lr: %.6f \n' % (
                                iteration + 1,  self.opt.total_iterations,
                                loss.item(),
                                optimizer.param_groups[0]['lr'])
                            io.cprint(outstr)
                        
                        
                        check_interval = 3 * (len(train_ds) // self.opt.batch_size)  
                        if iteration % check_interval == 0 or iteration == self.opt.total_iterations - 1:
                            model.eval()
                            start_time = time.time()
                            with torch.no_grad():
                                for batch_idx, batch in enumerate(val_loader):
                                    img = batch["image"].to(self.device)
                                    gt = batch["mask"]
                                    gt_one_hot = one_hot(gt, num_classes=len(label_current), dim=1).to(self.device)

                                    pred = model(img)
                                    pred_current = [val_post_process(i) for i in decollate_batch(pred)]
                                    gt_convert = [i for i in decollate_batch(gt_one_hot)]

                                    dice_metric(y_pred=pred_current, y=gt_convert)
                                    
                                dice_metrics = dice_metric.aggregate().item()
                                dice_metric.reset()
                                
                                end_time = time.time()
                                val_time = end_time - start_time
                            metrics = dice_metrics
                            io.cprint("The inference time in this epoch: {:} s".format(val_time))
                            io.cprint("Current epoch average metric: {:.4f}".format(metrics))
                            writer.add_scalar('step_{:}_val_dice'.format(step), dice_metrics, iteration)
                            

                            if metrics >= best_metric:
                                best_metric = metrics
                                best_metric_iter = iteration
                        
                        if iteration % 5000 == 0 or iteration == self.opt.total_iterations - 1:
                            torch.save(model.state_dict(), model_fold_savepath + "/model_{:}.pt".format(iteration + 1))
                            io.cprint("saved new best metric model")
                            io.cprint(
                                f"current iteration: {iteration + 1} current mean dice: {metrics:.4f}"
                                f"\nbest mean metric: {best_metric:.4f} "
                                f"at epoch: {best_metric_iter}"
                            )

            else:
                with tqdm(total=self.opt.total_iterations) as pbar:
                    for iteration in range(self.opt.start_iteration, self.opt.total_iterations):
                        if iteration % 100 == 0 and iteration > 0:
                            pbar.update(100)
                        
                        model.train()
                        
                        batch_previous_list = []
                        try:
                            batch_current = next(train_loader_iter)
                        except StopIteration:
                            scheduler.step()
                            train_loader_iter = iter(train_loader)
                            batch_current = next(train_loader_iter)
                        try:
                            for fake_loader_iter in fake_loader_iter_list:
                                batch_previous_list.append(next(fake_loader_iter))
                        except StopIteration:
                            fake_loader_iter_list = []
                            for fake_loader in fake_loader_list:
                                fake_loader_iter_list.append(iter(fake_loader))
                            for fake_loader_iter in fake_loader_iter_list:
                                batch_previous_list.append(next(fake_loader_iter))
                                
                        
                        img_current = batch_current["image"].to(self.device)
                        gt_current = batch_current["mask"]
                        gt_onehot_current = one_hot(gt_current, num_classes=len(label_current), dim=1).to(self.device)
                       
                        
                        img_previous_list = []
                        gt_previous_list = []
                        for batch_previous in batch_previous_list: 
                            img_previous_list.append(batch_previous["image"].to(self.device))
                            gt_previous_list.append(one_hot(batch_previous["mask"], num_classes=len(label_current), dim=1).to(self.device))
                              
                        
                        img_previous = torch.cat(img_previous_list, dim=0)
                        gt_onehot_previous = torch.cat(gt_previous_list, dim=0)
                        del img_previous_list, gt_previous_list

                        
                        #rand_index = torch.randperm(self.opt.batch_size)[:self.opt.batch_size // 2].cuda()
                        img = torch.cat((img_current, img_previous), dim=0)
                        gt_onehot = torch.cat((gt_onehot_current, gt_onehot_previous), dim=0)

                        pred = model(img)
                        loss = loss_fun(pred, gt_onehot)

                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('step_{:}_train_loss'.format(step),
                                                        loss.item(),
                                                        iteration)
                        
                       
                        if iteration % 100 == 0:
                            outstr = 'train-- iteration: %d/%d \n train_loss: %.6f, lr: %.6f \n' % (
                                iteration + 1,  self.opt.total_iterations,
                                loss.item(),
                                optimizer.param_groups[0]['lr'])
                            io.cprint(outstr)
                        
                        
                        check_interval = 3 * (len(train_ds) // self.opt.batch_size)  
                        if iteration % check_interval == 0 or iteration == self.opt.total_iterations - 1:
                            model.eval()
                            start_time = time.time()
                            with torch.no_grad():
                                for batch_idx, batch in enumerate(val_loader):
                                    img = batch["image"].to(self.device)
                                    gt = batch["mask"]
                                    gt_one_hot = one_hot(gt, num_classes=len(label_current), dim=1).to(self.device)

                                    pred = model(img)
                                    pred_current = [val_post_process(i) for i in decollate_batch(pred)]
                                    gt_convert = [i for i in decollate_batch(gt_one_hot)]

                                    
                                    dice_metric(y_pred=pred_current, y=gt_convert)
                                    
                                dice_metrics = dice_metric.aggregate().item()
                                dice_metric.reset()
                                end_time = time.time()
                                val_time = end_time - start_time
                            metrics = dice_metrics

                            io.cprint("The inference time in this epoch: {:} s".format(val_time))
                            io.cprint("Current epoch average dice: {:.4f}".format(metrics))
                            writer.add_scalar('step_{:}_val_dice'.format(step), dice_metrics, iteration)
                            

                            if metrics >= best_metric:
                                best_metric = metrics
                                best_metric_iter = iteration
                        
                        if iteration % 5000 == 0 or iteration == self.opt.total_iterations - 1:
                            torch.save(model.state_dict(), model_fold_savepath + "/model_{:}.pt".format(iteration + 1))
                            io.cprint("saved new best metric model")
                            io.cprint(
                                f"current iteration: {iteration + 1} current mean dice: {metrics:.4f}"
                                f"\nbest mean dice: {best_metric:.4f} "
                                f"at epoch: {best_metric_iter}"
                            )
                
        
        

            
            
            
            



