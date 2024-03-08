import torch.nn as nn
import numpy as np
from monai.networks.blocks import ResidualUnit, Convolution
import torch

class ResidualBlock(nn.Module):
    """
    Basic Convolution Block with residual connection
    """
    def __init__(self, in_ch, out_ch, spatial_dims=2, strides=1):
        super().__init__()
        self.convs = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=out_ch,
            strides=strides,
        )

    def forward(self, x):
        out = self.convs(x)
        return out

class down_conv(nn.Module):
    """
    Downsampling based on a conv layer with stride 2
    """
    def __init__(self, in_ch, spatial_dims=2):
        super().__init__()
        self.down_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=in_ch,
            strides=2,
            kernel_size=1,
            conv_only=True
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_dims=2):
        super().__init__()
        self.upsample = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=in_ch,
            strides=2,
            kernel_size=3,
            conv_only=True,
            is_transposed=True,
            output_padding=1
        )
        self.upconv = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=in_ch * 2,
            out_channels=out_ch,
            subunits=1,
        )
    
    def forward(self, f_de, f_en):
        x = self.upsample(f_de)  
        f_concat = torch.concat((x, f_en), dim=1)  
        x = self.upconv(f_concat)
        return x


class res_unet_backbone(nn.Module):
    def __init__(self, in_ch=1, spatial_dims=2):
        super().__init__()
        self.base_channel = 32
        layer_channel = self.base_channel * np.array([1, 2, 4, 8, 16])
        

        self.en_conv1 = ResidualBlock(in_ch, layer_channel[0])
        self.en_conv2 = ResidualBlock(layer_channel[0], layer_channel[1])
        self.en_conv3 = ResidualBlock(layer_channel[1], layer_channel[2])
        self.en_conv4 = ResidualBlock(layer_channel[2], layer_channel[3])
        self.en_conv5 = ResidualBlock(layer_channel[3], layer_channel[4])

        self.center = ResidualBlock(layer_channel[4], layer_channel[3])
        
        # self.de_conv1 = ResidualBlock(layer_channel[0], layer_channel[0], strides=1)

        self.pool1 = down_conv(layer_channel[0])
        self.pool2 = down_conv(layer_channel[1])
        self.pool3 = down_conv(layer_channel[2])
        self.pool4 = down_conv(layer_channel[3])

        self.upsample4 = up_conv(layer_channel[3], layer_channel[2])
        self.upsample3 = up_conv(layer_channel[2], layer_channel[1])
        self.upsample2 = up_conv(layer_channel[1], layer_channel[0])
        self.upsample1 = up_conv(layer_channel[0], layer_channel[0])

    def forward(self, x):
        e1 = self.en_conv1(x)
        e1_pool = self.pool1(e1)

        e2 = self.en_conv2(e1_pool)
        e2_pool = self.pool2(e2)

        e3 = self.en_conv3(e2_pool)
        e3_pool = self.pool3(e3)

        e4 = self.en_conv4(e3_pool)
        e4_pool = self.pool4(e4)

        e5 = self.en_conv5(e4_pool)
        
        d5 = self.center(e5)
        
        d4 = self.upsample4(d5, e4)
        d3 = self.upsample3(d4, e3)
        d2 = self.upsample2(d3, e2)
        d1 = self.upsample1(d2, e1)
        
        return d1