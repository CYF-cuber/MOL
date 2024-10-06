from torch import nn, Tensor
import math
import torch
import numpy as np
from torch.nn import functional as F
from typing import Optional, Dict, Tuple

class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def forward(self, x: Tensor or Tuple[Tensor]) -> Tensor or Tuple[Tensor]:
        raise NotImplementedError

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
        
class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(Dropout, self).__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

class Dropout2d(nn.Dropout2d):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(Dropout2d, self).__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

class FCC(BaseModule):
    def __init__(self,
                 dim = 64,
                 meta_kernel_size = 32,
                 instance_kernel_method='crop',
                 use_pe:Optional[bool]=True,
                 mid_mix: Optional[bool]=False,
                 bias: Optional[bool]=True,
                 ffn_dim: Optional[int]=2,
                 ffn_dropout=0.0,
                 dropout=0.1):

        super(FCC, self).__init__()

        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.meta_kernel_1_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_1_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight
        self.meta_kernel_2_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_2_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight

        if bias:
            self.meta_1_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_1_W_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_W_bias = nn.Parameter(torch.randn(dim))
        else:
            self.meta_1_H_bias = None
            self.meta_1_W_bias = None
            self.meta_2_H_bias = None
            self.meta_2_W_bias = None

        self.instance_kernel_method = instance_kernel_method

        if use_pe:
            self.meta_pe_1_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_1_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.meta_pe_2_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_2_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))

        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix
        self.use_pe = use_pe
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            Dropout(p=dropout)
        )

    def get_instance_kernel(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_kernel_1_H[:, :, : instance_kernel_size,:], \
                   self.meta_kernel_1_W[:, :, :, :instance_kernel_size], \
                   self.meta_kernel_2_H[:, :, :instance_kernel_size, :], \
                   self.meta_kernel_2_W[:, :, :, :instance_kernel_size]

        elif self.instance_kernel_method == 'interpolation_bilinear':
            H_shape = [instance_kernel_size, 1]
            W_shape = [1, instance_kernel_size]
            return F.interpolate(self.meta_kernel_1_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_1_W, W_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_W, W_shape, mode='bilinear', align_corners=True),

        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def get_instance_pe(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_pe_1_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_1_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)

        elif self.instance_kernel_method == 'interpolation_bilinear':
            return F.interpolate(self.meta_pe_1_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_1_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)
        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        # print('chunked shape:')
        # print(x_1.shape) # [8, 2, 32, 32]
        x_1_res, x_2_res = x_1, x_2
        _, _, f_s, _ = x_1.shape

        K_1_H, K_1_W, K_2_H, K_2_W = self.get_instance_kernel(f_s)

        if self.use_pe:
            pe_1_H, pe_1_W, pe_2_H, pe_2_W = self.get_instance_pe(f_s)
            #print(type(pe_1_H),pe_1_H.shape)
            #print(type(x_1),x_1.shape)


        # print('pe_1_H shape:')
        # print(pe_1_H.shape)
        # **************************************************************************************************sptial part
        # pre norm
        if self.use_pe:
            x_1, x_2 = x_1 + pe_1_H, x_2 + pe_1_W

        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1
        x_1_1 = F.conv2d(torch.cat((x_1, x_1[:, :, :-1, :]), dim=2), weight=K_1_H, bias=self.meta_1_H_bias, padding=0,
                         groups=self.dim)
        x_2_1 = F.conv2d(torch.cat((x_2, x_2[:, :, :, :-1]), dim=3), weight=K_1_W, bias=self.meta_1_W_bias, padding=0,
                         groups=self.dim)
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        if self.use_pe:
            x_1_1, x_2_1 = x_1_1 + pe_2_W, x_2_1 + pe_2_H

        # stage 2
        x_1_2 = F.conv2d(torch.cat((x_1_1, x_1_1[:, :, :, :-1]), dim=3), weight=K_2_W, bias=self.meta_2_W_bias,
                         padding=0, groups=self.dim)
        x_2_2 = F.conv2d(torch.cat((x_2_1, x_2_1[:, :, :-1, :]), dim=2), weight=K_2_H, bias=self.meta_2_H_bias,
                         padding=0, groups=self.dim)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        # *************************************************************************************************channel part
        x_out = torch.cat((x_1, x_2), dim=1)

        return x_out

if __name__ == "__main__":

    model = FCC().cuda()
    #total_num = sum(p.numel() for p in model.parameters())
    #print(total_num)    