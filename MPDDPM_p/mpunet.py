import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Dict, List, Any
from segformer import Block as AttentionBlock
from segformer import PatchEmbed
from thop import profile


class DoubleConv(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2  不减少分辨率"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    """下采样  减少分辨率"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Decoder(nn.Module):
    def __init__(self, num_classes=1, base_c=32, bilinear=True):
        super(Decoder, self).__init__()
        factor = 2 if bilinear else 1

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x1, x2, x3, x4, x5 = x
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_c=32):
        super(Encoder, self).__init__()

        self.in_conv = DoubleConv(in_channels, base_c)

        self.down1 = Down(base_c, base_c * 2)  # /2
        self.down2 = Down(base_c * 2, base_c * 4)  # /4
        self.down3 = Down(base_c * 4, base_c * 8)  # /8

        factor = 2
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        dpr = [x.item() for x in torch.linspace(0, 0.1, 8)]

        cur = 0
        self.transformer_down1 = nn.ModuleList([AttentionBlock(32, 1, 8, dpr[cur + i]) for i in range(2)])
        self.norm1 = nn.LayerNorm(32)

        cur += 2
        self.transformer_down2 = nn.ModuleList([AttentionBlock(64, 2, 4, dpr[cur + i]) for i in range(2)])
        self.norm2 = nn.LayerNorm(64)
        cur += 2

        self.transformer_down3 = nn.ModuleList([AttentionBlock(128, 4, 2, dpr[cur + i]) for i in range(2)])
        self.norm3 = nn.LayerNorm(128)

        cur += 2
        self.transformer_down4 = nn.ModuleList([AttentionBlock(256, 8, 1, dpr[cur + i]) for i in range(2)])
        self.norm4 = nn.LayerNorm(256)

        self.patch_embed2 = PatchEmbed(64, 128, 3, 2)
        self.patch_embed3 = PatchEmbed(128, 256, 3, 2)

    def forward(self, x) -> list[torch.Tensor]:
        x1 = self.in_conv(x)  # x1.shape torch.Size([1, 32, 256, 256])

        x2 = self.down1(x1)  # x2.shape torch.Size([1, 64, 128, 128])

        B, C, H, W = x2.shape
        x2_transformer = x2.flatten(2).transpose(1, 2)

        for block in self.transformer_down2:
            x2_transformer = block(x2_transformer, H, W)
        x2_transformer = self.norm2(x2_transformer).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # ([1, 64, 128, 128])

        x3 = self.down2(x2)  # x3.shape torch.Size([1, 128, 64, 64])
        B, C, H, W = x3.shape

        x3_transformer = x3.flatten(2).transpose(1, 2) + self.patch_embed2(x2_transformer)[0]
        for block in self.transformer_down3:
            x3_transformer = block(x3_transformer, H, W)
        x3_transformer = self.norm3(x3_transformer).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # ([1, 128, 64, 64])

        x4 = self.down3(x3)  # x4.shape torch.Size([1, 256, 32, 32])
        B, C, H, W = x4.shape
        x4_transformer = x4.flatten(2).transpose(1, 2) + self.patch_embed3(x3_transformer)[0]
        for block in self.transformer_down4:
            x4_transformer = block(x4_transformer, H, W)
        x4_transformer = self.norm4(x4_transformer).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # ([1, 256, 32, 32])

        x4 = x4 + x4_transformer
        x5 = self.down4(x4)  # x5.shape torch.Size([1, 256, 16, 16])
        return [x1, x2, x3, x4, x5]


class MPUnet(nn.Module):
    def __init__(self, in_channels, out_channels, device=None):
        super(MPUnet, self).__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(num_classes=out_channels)
        self.device = "cpu"
        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encoder(x)
        return self.decoder(x)


if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 256)
    model = MPUnet(in_channels=1, out_channels=1)

    input = torch.randn(1, 1, 256, 256)

    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
