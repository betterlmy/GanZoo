import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

ACTIVATION_FUNCTION = nn.ReLU


class DoubleConv(nn.Module):
    """双重卷积使用两个卷积层可以在某种程度上减缓这种信息损失，因为第二层有机会重新强调第一层中可能被弱化的重要特征。"""
    """ (Convolution => [BatchNorm] => ReLU) * 2 """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            ACTIVATION_FUNCTION(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            ACTIVATION_FUNCTION(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样是减少图像分辨率的过程，它有助于网络在更深的层次上捕获更广泛和更抽象的特征。这对于理解图像的整体结构和上下文非常重要。"""
    """ Downscaling with maxpool then double conv """
    """下采样可以选择池化操作,也可以使用卷积操作"""
    """
    池化特点:
    1. 在某种程度上保留了最显著的特征，如边缘。
    2. 减少过拟合与模型复杂度：由于最大池化不包含可学习的参数，它有助于减少模型的复杂度，从而降低过拟合的风险。
    
    卷积特点:
    1.可学习的下采样：与最大池化不同，步长为2的卷积是一种可学习的下采样方法。这意味着网络可以学习在下采样过程中保留哪些信息。
    2.特征提取与下采样的结合：步长为2的卷积同时进行特征提取和下采样，这可以在降低空间维度的同时提取有用的特征。
    3.参数数量增加：由于卷积层包含可学习的参数，使用步长为2的卷积进行下采样会增加模型的参数数量。
    """

    def __init__(self, in_channels, out_channels, down_type='maxpool'):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        if down_type != 'maxpool':
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                ACTIVATION_FUNCTION(inplace=True),
                DoubleConv(out_channels, out_channels)
            )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """ Upscaling then double conv """
    """
    在Unet网络中，上采样模块是解码器部分的核心组件。
    它的主要作用是将经过编码器（Encoder）部分的下采样（Downsampling）和特征提取后的压缩特征图（Feature Map）逐步恢复到原始图像的尺寸。
    这一过程是通过逐层增加特征图的空间维度（即高度和宽度）来实现的，同时保留和恢复图像的重要特征。
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        :param x1: 上层传递下来的特征图
        :param x2: 跳跃连接传递的特征图
        :return:
        """
        x1 = self.up(x1)  # 对x1进行上采样
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/milesial/Pytorch-UNet/issues/18
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DualChannelUnet(nn.Module):
    def __init__(self, in_channels, out_channels, device, bilinear=True, maxpool=True):
        super(DualChannelUnet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        self.maxpool = maxpool
        self.device = device
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        self.to(self.device)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # 用于测试网络结构是否正确
    model = DualChannelUnet(2, 1)
    torchsummary.summary(model, (2, 224, 224))
