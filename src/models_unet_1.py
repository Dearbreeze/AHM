import cv2
import torch
import imageio
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from re import L
from PIL import Image
from tkinter import X
from operator import xor
import matplotlib.pyplot as plt
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_channels']
        self.out_channels = self.params['out_channels']
        self.bilinear = self.params['bilinear']
        self.factor = self.params['factor']
        self.dropout = self.params['dropout']
        feature_num = self.params['init_feature_num']

        self.inc = DoubleConv(self.in_channels, feature_num, dropout_p=self.dropout[0])
        self.down1 = DownBlock(feature_num, feature_num*2, self.dropout[1])
        self.down2 = DownBlock(feature_num*2, feature_num*4, self.dropout[2])
        self.down3 = DownBlock(feature_num*4, feature_num*8, self.dropout[3])
        self.down4 = DownBlock(feature_num*8, feature_num*16 // self.factor, self.dropout[4])


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  
        return [x1, x2, x3, x4, x5]

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_channels = self.params['in_channels']
        self.out_channels = self.params['out_channels']
        self.bilinear = self.params['bilinear']
        self.factor = self.params['factor']
        feature_num = self.params['init_feature_num']

        self.up1 = UpBlock(feature_num*16, feature_num*8, self.bilinear, dropout_p=0.0)
        self.up2 = UpBlock(feature_num*8, feature_num*4, self.bilinear, dropout_p=0.0)
        self.up3 = UpBlock(feature_num*4, feature_num*2, self.bilinear, dropout_p=0.0)
        self.up4 = UpBlock(feature_num*2, feature_num * self.factor, self.bilinear, dropout_p=0.0)
        self.outc = OutConv(feature_num, self.out_channels)

    def forward(self, feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]
        x5 = feature[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)  # torch.Size([1, 32, 256, 256])
        out = self.outc(x)   # torch.Size([1, 1, 256, 256])

        return out

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels,init_feature_num=32, bilinear=False, **kwargs):
        super(UNet, self).__init__()
        factor = 2 if bilinear else 1
        params = {'in_channels': in_channels,
                  'init_feature_num': init_feature_num,
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                #   'dropout': [0.1, 0.0, 0.0, 0.0, 0.0],
                  'out_channels': out_channels,
                  'bilinear': bilinear,
                  'factor': factor}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        
    def forward(self, x):
        feature = self.encoder(x)
        output= self.decoder(feature)
        return output


def mae_cnn_img_unet(**kwargs):
    model = UNet(in_channels=1, out_channels=1, init_feature_num=32,bilinear=False, **kwargs)  # **kwargs to input $mask_ratios$
    return model

