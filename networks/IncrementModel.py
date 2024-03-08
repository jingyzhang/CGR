import torch.nn as nn
from monai.networks.blocks import Convolution

from networks.res_unet import res_unet_backbone


class IncrementModel(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, backbone="Res_UNet", spatial_dims=2):
        super().__init__()
        if backbone == "res_unet":
            self.backbone = res_unet_backbone(in_ch=in_ch, spatial_dims=spatial_dims)
        
        base_channel = self.backbone.base_channel

        self.classifier = Convolution(l
            spatial_dims=spatial_dims,
            in_channels=base_channel,
            out_channels=out_ch,
            kernel_size=1,
            conv_only=True
        )
    
    def forward(self, x):
        x = self.backbone(x)
        out = self.classifier(x)
        return out
    
