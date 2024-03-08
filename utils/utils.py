import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import numpy as np

class myloss(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.bceloss = nn.BCELoss(reduction="none")
        

    def forward(self, pred, gt):
        N = gt.size(0)
        
        bce_loss = self.bceloss(pred, gt)
        dice_loss = 1 - diceCoeff(pred, gt)
        bce_loss = torch.mean(bce_loss.view(N,-1), dim=1)
        loss = dice_loss + self.alpha * bce_loss
        loss = torch.mean(loss)
        return loss

def diceCoeff(pred, gt, smooth=1e-5):
    """ computational formula
        dice = (2 * (pred ∩ gt)) / (pred U  gt)
    """
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)

    return dice

class IOStream():
    def __init__(self, path):
        f_init = open(path, 'w')
        f_init.close()
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_json(path):
    f = open(path, 'r')
    content = f.read()
    dict = json.loads(content)
    return dict

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
