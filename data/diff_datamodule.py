from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandAxisFlipd,
)
from monai.data import Dataset
import os
from glob import glob
import numpy as np

class CFPDataset():
    def __init__(self, dataroot, task_name, mode='seperate') -> None:
        
        if mode == 'seperate':
            datasets = []
            for task_label, task_name_current in enumerate(task_name):
                img_task_name_current = task_name_current + '_dataset_npz'
                mask_task_name_current = task_name_current + '_dataset_npz'
                imgs_path = sorted(glob(os.path.join(dataroot, img_task_name_current, "imagesTr", "*.npz")))
                masks_path = sorted(glob(os.path.join(dataroot, mask_task_name_current, "labelsTr", "*.npz")))
                data_dicts = [{"image": img, "mask": mask, 'task_label': task_label, 'prompt': self.get_text_prompt_from_image(mask, task_label)} 
                               for img, mask in zip(imgs_path, masks_path)]
                datasets.append(Dataset(data=data_dicts, transform=self.get_input_transform()))
            self.dataset = datasets
        

    def get_input_transform(self):
        input_transforms = Compose(
                [
                    LoadImaged(keys=["image", "mask"], image_only=False),
                    EnsureChannelFirstd(keys=["mask"]),
                    ScaleIntensityd(keys=["image", "mask"], minv=-1, maxv=1),
                    RandAxisFlipd(keys=['image','mask'], prob=0.5, lazy=False),
                ]
            )
        return input_transforms
    
    
    def get_test_input_transform(self):
        input_transforms = Compose(
                [
                    LoadImaged(keys=["image", "mask"], image_only=False),
                    EnsureChannelFirstd(keys=["mask"]),
                    ScaleIntensityd(keys=["image"], minv=-1, maxv=1),
                ]
            )
        return input_transforms

    def isgraymask(self, mask):
        if mask.shape[0] == 1:
            mask = mask.repeat(3,1,1)
        return mask

    def get_dataset(self):
        return self.dataset
    
    def get_text_prompt_from_image(self, mask_path, task_name):
        mask_array = np.load(mask_path)['arr_0'].astype(np.uint8)
        appear_label = np.unique(mask_array)
        del mask_array
        if task_name == 'mnms':
            descriptions = {
                1: "a left ventricle",
                2: "a myocardium",
                3: "a right ventricle"
            }
            text_parts = [descriptions[index] for index in appear_label if index in descriptions]
            text_prompt = "A magnetic resonance of cardiac with " + ", ".join(text_parts)
        
        if task_name == 'Fundus':
            descriptions = {
                1: "a optic cup",
                2: "a optic disc"
            }
            text_parts = [descriptions[index] for index in appear_label if index in descriptions]
            text_prompt = "A fundus photography with " + ", ".join(text_parts)
            
        if task_name == 'Prostate':
            descriptions = {
                1: "a prostate"
            }
            text_parts = [descriptions[index] for index in appear_label if index in descriptions]
            text_prompt = "A magnetic resonance of " + ", ".join(text_parts)
            
        return text_prompt
    
    

