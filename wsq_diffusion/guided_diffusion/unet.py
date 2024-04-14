from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

# from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
# from fightingcv_attention.conv.CondConv import *
# import torch
# class Simam_module(torch.nn.Module):
#     def __init__(self,e_lambda=1e-4):
#         super(Simam_module,self).__init__()
#         self.act = nn.Sigmoid()
#         self.e_lambda = e_lambda
#     def forward(self,x):
#         b,c, h, w = x.size()
#         n=w*h-1
#         x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#         return x * self.act(y)
# class myblock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(myblock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.attn_model1 = Simam_module()
#         self.attn_model2 = Simam_module()
#         self.model1 = CondConv(in_planes=in_channels, out_planes=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.model2 = DepthwiseSeparableConvolution(in_channels, out_channels)
#         self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv1x1_3 = nn.Conv2d(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3x3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.sig1 = nn.Sigmoid()
#         self.sig2 = nn.Sigmoid()

#     def forward(self, x):
#         h1 = self.attn_model1(self.model1(x))
#         o1 = self.conv3x3_1(h1)
#         y1 = h1 * self.sig1(o1) + self.conv1x1_1(x)

#         h2 = self.attn_model2(self.model2(x))
#         o2 = self.conv3x3_2(h2)
#         y2 = h2 * self.sig2(o2) + self.conv1x1_2(x)

#         # Concatenate y1 and y2 along the channel dimension
#         y = torch.cat([y1, y2], dim=1)

#         # # Apply 1x1 convolution
#         y = self.conv1x1_3(y)
#         return y
    
from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from fightingcv_attention.conv.CondConv import *
import torch
class Simam_module(torch.nn.Module):
    def __init__(self,e_lambda=1e-4):
        super(Simam_module,self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
    def forward(self,x):
        b,c, h, w = x.size()
        n=w*h-1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)
class myblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(myblock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attn_model1 = Simam_module()
        self.attn_model2 = Simam_module()
        self.model1 = CondConv(in_planes=in_channels, out_planes=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.model2 = DepthwiseSeparableConvolution(in_channels, out_channels)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=1, padding=1,)
        self.conv1x1_3 = nn.Conv2d(2*out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3x3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,)
        self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

    def forward(self, x):
        h1 = self.attn_model1(self.model1(x))
        o1 = self.conv3x3_1(h1)
        y1 = h1 * self.sig1(o1) + self.conv1x1_1(x)

        h2 = self.attn_model2(self.model2(x))
        o2 = self.conv3x3_2(h2)
        y2 = h2 * self.sig2(o2) + self.conv1x1_2(x)

        # Concatenate y1 and y2 along the channel dimension
        y = torch.cat([y1, y2], dim=1)

        # # Apply 1x1 convolution
        y = self.conv1x1_3(y)
        return y
# class myblock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(myblock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.attn_model1 = Simam_module()
#         self.attn_model2 = Simam_module()
#         self.model1 = CondConv(in_planes=in_channels, out_planes=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.model2 = DepthwiseSeparableConvolution(in_channels, out_channels)
#         self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,)
#         self.conv1x1_2 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=1, padding=1,)
#         self.conv1x1_3 = nn.Conv2d(2*out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.conv3x3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,)
#         self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,)
#         self.sig1 = nn.Sigmoid()
#         self.sig2 = nn.Sigmoid()

#     def forward(self, x):
#         h1 = self.model1(x)
#         o1 = self.conv3x3_1(h1)
#         y1 = h1 * self.sig1(o1) + self.conv1x1_1(x)

#         h2 = self.model2(x)
#         o2 = self.conv3x3_2(h2)
#         y2 = h2 * self.sig2(o2) + self.conv1x1_2(x)

#         # Concatenate y1 and y2 along the channel dimension
#         y = torch.cat([y1, y2], dim=1)

#         # # Apply 1x1 convolution
#         y = self.conv1x1_3(y)
#         return y

import torch
import torch.nn as nn
import torch.nn.functional as F
# tensor=torch.ones(size=(2,1280,32,32))
# print(tensor)
class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out
class SE_ASPP(nn.Module):                       ##加入通道注意力机制
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(SE_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
        self.senet=SE_Block(in_planes=dim_out*5)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        seaspp1=self.senet(feature_cat)             #加入通道注意力机制
        # print('seaspp1:',seaspp1.shape)
        se_feature_cat=seaspp1*feature_cat
        result = self.conv_cat(se_feature_cat)
        result = self.downsample(result)
        # print('result:',result.shape)
        return result

# from torch import nn
# import torch
# import torch.nn.functional as F
 
# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)
 
# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())
 
#     def forward(self, x):
#         size = x.shape[-2:]
#         x = super(ASPPPooling, self).forward(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
 
# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels,atrous_rates):
#         super(ASPP, self).__init__()
#         self.out_channels = out_channels
#         modules = []
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))
 
#         rate1, rate2, rate3 = tuple(atrous_rates)
#         modules.append(ASPPConv(in_channels, out_channels, rate1))
#         modules.append(ASPPConv(in_channels, out_channels, rate2))
#         modules.append(ASPPConv(in_channels, out_channels, rate3))
#         modules.append(ASPPPooling(in_channels, out_channels))
 
#         self.convs = nn.ModuleList(modules)
 
#         self.project = nn.Sequential(
#             nn.Conv2d(5 * out_channels, out_channels, 3,2,1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5))
 
#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
# from torch import nn
# import torch
# import torch.nn.functional as F
 
# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)
 
# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())
 
#     def forward(self, x):
#         size = x.shape[-2:]
#         x = super(ASPPPooling, self).forward(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
 
# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels,atrous_rates):
#         super(ASPP, self).__init__()
#         self.out_channels = out_channels
#         modules = []
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))
 
#         rate1, rate2, rate3 = tuple(atrous_rates)
#         modules.append(ASPPConv(in_channels, out_channels, rate1))
#         modules.append(ASPPConv(in_channels, out_channels, rate2))
#         modules.append(ASPPConv(in_channels, out_channels, rate3))
#         modules.append(ASPPPooling(in_channels, out_channels))
 
#         self.convs = nn.ModuleList(modules)
 
#         self.project = nn.Sequential(
#             nn.Conv2d(5 * out_channels, out_channels, 3,1,1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5))
 
#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)

# class GLFM(nn.Module):
#     '''
#     单特征 进行通道加权,作用类似SE模块
#     '''

#     def __init__(self, channels=64, r=4):
#         super(GLFM, self).__init__()
#         inter_channels = int(channels // r)

#         self.local_att = nn.Sequential(
#             DepthwiseSeparableConvolution(channels,inter_channels),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             DepthwiseSeparableConvolution(inter_channels,channels),
#             nn.BatchNorm2d(channels),
#         )

#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         xl = self.local_att(x)
#         xg = self.global_att(x)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         return x * wei

# class Model(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(Model, self).__init__()
#         self.conv1x1_input = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn_0 = nn.BatchNorm2d(out_channels)
#         inter_channels = in_channels 
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv1x1_middle = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
#         self.bn_middle = nn.BatchNorm2d(inter_channels)


#         self.aspp = ASPP(inter_channels,inter_channels,[6,12,18])
#         self.gl_fm = GLFM(inter_channels,4)
#         self.bn_2 = nn.BatchNorm2d(inter_channels)
#         self.conv1x1_output = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
#         self.bn_output = nn.BatchNorm2d(out_channels)
#         self.convout = nn.Conv2d(out_channels, out_channels, 3,2,1)
        
#         # You will need to define the in_channels and out_channels as per your model's requirements.
        
#     def forward(self, x):
#         # Initial 1x1 conv and BN
#         h = self.conv1x1_input(x)
#         h = self.bn_0(h)

#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv1x1_middle(x)
#         x = self.bn_middle(x)
#         x = self.relu(x)
#         x = self.aspp(x)
#         x = self.gl_fm(x)
#         x = self.bn_2(x)
#         x = self.relu(x)
#         x = self.conv1x1_output(x)
#         x = self.bn_output(x)
#         out = self.convout(x + h)
#         return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
import torch
from torch import nn
from torch.nn import functional as F
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, stride=1):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
 
    def forward(self, x):
        size = x.shape[-2:]
        a = (int(size[0] / 2),int(size[0] / 2))
        torch_size = torch.Size(a)
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=torch_size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        
        for rate in atrous_rates:
            self.convs.append(ASPPConv(in_channels, out_channels, rate, stride=2))
        
        self.pooling = ASPPPooling(in_channels, out_channels)
        
        self.project = nn.Sequential(
            nn.Conv2d((len(atrous_rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        res = [conv(x) for conv in self.convs]
        pooling = self.pooling(x)
        #pooling = F.interpolate(pooling, size=size, mode='bilinear', align_corners=False)
        res.append(pooling)
        res = torch.cat(res, dim=1)
        return self.project(res)


class GLFM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(GLFM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            DepthwiseSeparableConvolution(channels,inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConvolution(inter_channels,channels),
            nn.BatchNorm2d(channels),
            nn.AdaptiveAvgPool2d(1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei
# Model类以及其他部分保持不变，仅需将ASPP模块的修改应用到Model中

# 注意：下面这部分Model的定义没有改变，仅为参考
class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
        self.conv1x1_input = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=2,padding=1)
        self.bn_0 = nn.BatchNorm2d(out_channels)
        inter_channels = in_channels 
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1_middle = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.bn_middle = nn.BatchNorm2d(inter_channels)

        # 注意这里的aspp模块已经根据之前的讨论进行了修改，以在内部实现下采样
        self.aspp = ASPP(inter_channels, inter_channels, [6,12,18])
        self.gl_fm = GLFM(inter_channels, 4)
        self.bn_2 = nn.BatchNorm2d(inter_channels)
        self.conv1x1_output = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        self.bn_output = nn.BatchNorm2d(out_channels)
        # 注意：去掉了之前用于下采样的卷积层
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # Initial 1x1 conv and BN
        h = self.conv1x1_input(x)
        h = self.bn_0(h)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1_middle(x)
        x = self.bn_middle(x)
        x = self.relu(x)

        # 通过ASPP模块处理x并实现下采样
        x = self.aspp(x)

        x = self.gl_fm(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv1x1_output(x)
        x = self.bn_output(x)

        # 最后的卷积层现在只是用于调整特征图，而不是下采样
        out = self.final_conv(x + h)
        return out

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)  #改98和131行
            #self.conv = myblock(self.channels, self.out_channels, 3, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        # if use_conv:
        #     self.op = myblock(
        #         self.channels, self.out_channels,3,2,1
        #     )
        # if use_conv:
        #     self.op = SE_ASPP(
        #         self.channels, self.out_channels,
        #     )
        if use_conv:
            self.op = Model(
                self.channels,self.out_channels,
            )
        # if use_conv:
        #     self.op = ASPP(
        #         self.channels,self.out_channels,[6,12,18]
        #     )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

# class Upsample(nn.Module):
#     """
#     An upsampling layer with an optional convolution.

#     :param channels: channels in the inputs and outputs.
#     :param use_conv: a bool determining if a convolution is applied.
#     :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
#                  upsampling occurs in the inner-two dimensions.
#     """

#     def __init__(self, channels, use_conv, dims=2, out_channels=None):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims
#         if use_conv:
#             self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)  #改98和131行

#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         if self.dims == 3:
#             x = F.interpolate(
#                 x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
#             )
#         else:
#             x = F.interpolate(x, scale_factor=2, mode="nearest")
#         if self.use_conv:
#             x = self.conv(x)
#         return x


# class Downsample(nn.Module):
#     """
#     A downsampling layer with an optional convolution.

#     :param channels: channels in the inputs and outputs.
#     :param use_conv: a bool determining if a convolution is applied.
#     :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
#                  downsampling occurs in the inner-two dimensions.
#     """

#     def __init__(self, channels, use_conv, dims=2, out_channels=None):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims
#         stride = 2 if dims != 3 else (1, 2, 2)
#         if use_conv:
#             self.op = conv_nd(
#                 dims, self.channels, self.out_channels, 3, stride=stride, padding=1
#             )
#         else:
#             assert self.channels == self.out_channels
#             self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        # self.attention = QKVAttention(1)
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    """ From this repository: https://github.com/yandex-research/ddpm-segmentation """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'qkv')
    return out

# def attention_from_qkv(qkv, num_heads):
#     bs, width, length = qkv.shape
#     assert width % (3 * num_heads) == 0
#     ch = width // (3 * num_heads)
#     q, k, v = qkv.reshape(bs * num_heads, ch * 3, length).split(ch, dim=1)
#     scale = 1 / math.sqrt(math.sqrt(ch))
#     weight = th.einsum(
#         "bct,bcs->bts", q * scale, k * scale
#     )  # More stable with f16 than dividing afterwards
#     weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
#     return weight

# import torch as th
# import math

# def calculate_noise_level(k, seq_length):
#     # Calculate the variance (as an estimate of noise) across the channels dimension
#     noise_level = th.var(k, dim=1, keepdim=True)  # Calculate variance across the 'ch' dimension
#     # Normalize the noise level to [0, 1] range
#     noise_level = (noise_level - noise_level.min()) / (noise_level.max() - noise_level.min())
#     # Reshape noise_level to have shape [batch_size * num_heads, 1, seq_length]
#     noise_level = noise_level.view(-1, 1, seq_length)
#     return noise_level

# def attention_from_qkv(qkv, num_heads):
#     bs, width, seq_length = qkv.shape
#     assert width % (3 * num_heads) == 0
#     ch = width // (3 * num_heads)

#     q, k, v = qkv.reshape(bs * num_heads, ch * 3, seq_length).split(ch, dim=1)
#     scale = 1 / math.sqrt(math.sqrt(ch))

#     # Calculate weights
#     weight = th.einsum("bct,bcs->bts", q * scale, k * scale)

#     # Calculate noise level and adjust
#     noise_level = calculate_noise_level(v, seq_length)
#     # Ensure the noise_level tensor is correctly shaped for broadcasting
#     adaptive_weight = weight * (1 - noise_level)

#     # Normalize weights
#     adaptive_weight = th.softmax(adaptive_weight.float(), dim=-1).type(weight.dtype)

#     return adaptive_weight

import torch as th
import math

def calculate_adaptive_adjustment(weights, seq_length):
    # 以权重的方差作为自适应调整的依据
    adjustment = th.var(weights, dim=2, keepdim=True)  # 计算沿序列长度维度的方差
    # 归一化调整到 [0, 1] 范围
    adjustment = (adjustment - adjustment.min()) / (adjustment.max() - adjustment.min())
    # 调整形状以匹配权重张量
    adjustment = adjustment.view(-1, 1, seq_length)
    return adjustment

def attention_from_qkv(qkv, num_heads):
    bs, width, seq_length = qkv.shape
    assert width % (3 * num_heads) == 0
    ch = width // (3 * num_heads)

    q, k, v = qkv.reshape(bs * num_heads, ch * 3, seq_length).split(ch, dim=1)
    scale = 1 / math.sqrt(ch)

    # 计算标准的注意力权重
    weight = th.einsum("bct,bcs->bts", q * scale, k * scale)

    # 计算自适应调整
    adaptive_adjustment = calculate_adaptive_adjustment(weight, seq_length)
    # 使用自适应调整来修改权重
    adaptive_weight = weight * (1 - adaptive_adjustment)

    # 应用 softmax 获取最终注意力概率
    adaptive_weight = th.softmax(adaptive_weight.float(), dim=-1).type(weight.dtype)

    return adaptive_weight

# import torch as th
# import math

# def calculate_adaptive_adjustment(q, k, v, seq_length):
#     # 计算 q, k, v 的统计特性
#     q_stat = th.var(q, dim=2, keepdim=True)
#     k_stat = th.var(k, dim=2, keepdim=True)
#     v_stat = th.var(v, dim=2, keepdim=True)

#     # 将统计特性组合起来生成调整因子
#     combined_stat = q_stat + k_stat + v_stat
#     adjustment = (combined_stat - combined_stat.min()) / (combined_stat.max() - combined_stat.min())
#     adjustment = adjustment.view(-1, 1, seq_length)
#     return adjustment

# def attention_from_qkv(qkv, num_heads):
#     bs, width, seq_length = qkv.shape
#     assert width % (3 * num_heads) == 0
#     ch = width // (3 * num_heads)

#     q, k, v = qkv.reshape(bs * num_heads, ch * 3, seq_length).split(ch, dim=1)
#     scale = 1 / math.sqrt(ch)

#     # 计算标准的注意力权重
#     weight = th.einsum("bct,bcs->bts", q * scale, k * scale)

#     # 计算自适应调整
#     adaptive_adjustment = calculate_adaptive_adjustment(q, k, v, seq_length)
#     # 使用自适应调整来修改权重
#     adaptive_weight = weight * (1 - adaptive_adjustment)

#     # 应用 softmax 获取最终注意力概率
#     adaptive_weight = th.softmax(adaptive_weight.float(), dim=-1).type(weight.dtype)

#     return adaptive_weight


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        sel_attn_depth=2,
        sel_attn_block="output"
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        # Register forward hook to the attention map
        if sel_attn_block == "middle":
            self.extract_attention = self.middle_block[1].attention
        elif sel_attn_block == "output":
            assert sel_attn_depth <= 8 and sel_attn_depth >= 0, "sel_attn_depth must be between 0 and 8"
            self.extract_attention = self.output_blocks[sel_attn_depth][1].attention
        else:
            raise ValueError("sel_attn_block must be 'middle' or 'output'")
        
        self.extract_attention.register_forward_hook(save_input_hook)
        
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, ref_img=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        
        # Return the attention map with the output
        qkv = self.extract_attention.qkv
        attention = attention_from_qkv(qkv, self.num_heads)
        return self.out(h),attention


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
