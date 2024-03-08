from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    RandRotated,
    RandAxisFlipd,
    Lambdad,
)

import os
from glob import glob
import torch

class DataModule():
    def __init__(self,dataset_path, current_sites_name, sites_name_all, dataset_json, device="cpu", mode="cgr"):
       
        self.dataset_path = dataset_path
        self.sites_name_all = sites_name_all
        self.current_sites_name = current_sites_name
        self.dataset_json = dataset_json
        self.device = device
        self.mode = mode
        self.data_dicts = []
        self.val_data_dicts = []
        self.site_revised_num_list = self.get_site_revised_num_list()
        
        
        sites_fold = os.path.join(self.dataset_path, self.current_sites_name + "_dataset_npz")
        site_idx = [k for k, v in self.sites_name_all.items() if v == self.current_sites_name][0]
        
        site_imgs_paths = sorted(glob(os.path.join(sites_fold, "imagesTr", "*.npz")))
        site_masks_paths = sorted(glob(os.path.join(sites_fold, "labelsTr", "*.npz")))
        
        val_site_imgs_paths = sorted(glob(os.path.join(sites_fold, "imagesVal", "*.npz")))
        val_site_masks_paths = sorted(glob(os.path.join(sites_fold, "labelsVal", "*.npz")))
        
        self.data_dicts = [{"image": image_name, "mask": mask_name, "task_label":site_idx} for image_name, mask_name in
                    zip(site_imgs_paths, site_masks_paths)]
        self.val_data_dicts = [{"image": image_name, "mask": mask_name, "task_label":site_idx} for image_name, mask_name in
                    zip(val_site_imgs_paths, val_site_masks_paths)]
        self.get_transform()
        
    def get_transform(self):
        
        self.train_transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            Lambdad(keys=["mask"], func=lambda x: self.revise_label(x)),
            EnsureChannelFirstd(keys=["mask"]),
            NormalizeIntensityd(keys="image"),
            RandAxisFlipd(keys=['image','mask'], prob=0.5, lazy=False),
            RandRotated(keys=['image', "mask"], prob=0.2, range_x=30, range_y=30, lazy=False),
        ]
        )
        
        self.val_transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            Lambdad(keys=["mask"], func=lambda x: self.revise_label(x)),
            EnsureChannelFirstd(keys=["mask"]),
            NormalizeIntensityd(keys="image"),
        ]
        )
        self.replay_train_transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            Lambdad(keys=["mask"], func=lambda x: self.replay_revise_label(x)),
            EnsureChannelFirstd(keys=["mask"]),
            NormalizeIntensityd(keys="image"),
            RandAxisFlipd(keys=['image','mask'], prob=0.5, lazy=False),
            RandRotated(keys=['image', "mask"], prob=0.2, range_x=30, range_y=30, lazy=False),
        ]
        )
        
    
    def get_site_revised_num_list(self, ):
        
        site_num_classes_list = [0]
        site_num_classes = 0
        for site_idx in self.sites_name_all:
            site_name = self.sites_name_all[site_idx]
            if site_idx == 0:
                site_num_classes += self.dataset_json[site_name]["num_class"] - 1 
                
            else:
                site_num_classes_list.append(site_num_classes)
                site_num_classes += self.dataset_json[site_name]["num_class"] - 1
        
        return site_num_classes_list
            
    
    def revise_label(self, mask):
        task_name = mask.meta['filename_or_obj'].split('/')[-4].split('_')[0]
        assert task_name != 'Prostate' or torch.max(mask) <= 1, mask.meta['filename_or_obj']
        task_label = [k for k, v in self.sites_name_all.items() if v == task_name]
        
        if task_label[0] == 0:
            return mask
        else:
            mask = torch.where(mask == 0, mask, mask + self.site_revised_num_list[task_label[0]])
            return mask
        
    def replay_revise_label(self, mask):
        task_name = mask.meta['filename_or_obj'].split('/')[-3]
        assert task_name != 'Prostate' or torch.max(mask) <= 1, mask.meta['filename_or_obj']
        task_label = [k for k, v in self.sites_name_all.items() if v == task_name]
        
        if task_label[0] == 0:
            return mask
        else:
            mask = torch.where(mask == 0, mask, mask + self.site_revised_num_list[task_label[0]])
            return mask
        
    