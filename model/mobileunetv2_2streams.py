"""
Modified from https://github.com/d-li14/mobilenetv2.pytorch 

Reference from a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import math
from .utils import *

__all__ = ['mobileunetv2']

class MobileUNetV2_2Streams(nn.Module):
    def __init__(self, args):
        super(MobileUNetV2_2Streams, self).__init__()

        self.args = args

        # building first layer
        self.conv0 = conv_bn_relu(4, 32, 3, 2)

        # building encoder layers with inverted residual blocks
        self.enc1 = buildblock_InvertedResidual(32, 16, 1, 1, 1)
        self.enc2 = buildblock_InvertedResidual(16, 24, 2, 6, 2)
        self.enc3 = buildblock_InvertedResidual(24, 32, 3, 6, 2)
        self.enc4 = buildblock_InvertedResidual(32, 64, 4, 6, 2)
        self.enc5 = buildblock_InvertedResidual(64, 96, 3, 6, 1)
        self.enc6 = buildblock_InvertedResidual(96, 160, 3, 6, 2)
        self.enc7 = buildblock_InvertedResidual(160, 320, 1, 6, 1)

        # building decoder layers
        self.dec5 = conv_bn_relu_up(320, 160, 3)
        self.dec4 = conv_bn_relu_up(160+64, 96, 3)
        self.dec3 = conv_bn_relu_up(96+32, 64, 3)
        self.dec2 = conv_bn_relu_up(64+24, 32, 3)
        self.dec1 = conv_bn_relu_up(32+16, 24, 3)
        self.dep_conv = conv_bn_relu(24, 1, 3, 1, bn=False, relu=False)

        # building the low-level stream
        self.ls_conv0 = conv_bn_relu(4, 32, 3, 1)
        self.ls_conv1 = conv_bn_relu(32, 32, 3, 1)
        self.ls_conv2 = conv_bn_relu(32, 1, 3, 1, bn=False, relu=False)

        # initialize weigths
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.5, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        if self.args.depth_norm:
            bz = dep.shape[0]
            dep_max = torch.max(dep.view(bz,-1),1, keepdim=False)[0].view(bz,1,1,1)
            dep = dep/(dep_max+1e-4)

        x = torch.cat((rgb,dep),dim=1)
        
        # semantic stream
        fe0 = self.conv0(x)
        fe1 = self.enc1(fe0)
        fe2 = self.enc2(fe1)
        fe3 = self.enc3(fe2)
        fe4 = self.enc4(fe3)
        fe5 = self.enc5(fe4)
        fe6 = self.enc6(fe5)
        fe7 = self.enc7(fe6)

        fd5 = self.dec5(fe7)
        fd4 = self.dec4(torch.cat((fd5, fe4), dim=1))
        fd3 = self.dec3(torch.cat((fd4, fe3), dim=1))
        fd2 = self.dec2(torch.cat((fd3, fe2), dim=1))
        fd1 = self.dec1(torch.cat((fd2, fe1), dim=1))
        y = self.dep_conv(fd1)

        # low-level stream
        f_ls0 = self.ls_conv0(torch.cat((rgb,y),dim=1))
        f_ls1 = self.ls_conv1(f_ls0)
        f_ls2 = self.ls_conv2(f_ls1)

        pred_dep = f_ls2 + y

        if self.args.depth_norm:
            pred_dep = pred_dep * dep_max

        return pred_dep


def mobileunetv2(args):
    """
    Constructs a MobileNet V2 model
    """
    return MobileUNetV2_2Streams(args)

