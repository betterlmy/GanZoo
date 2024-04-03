import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F
import warnings
import math


class CrossAttentionP(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        # kv will take a different input than q, hence separate layers
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            # Spatial reduction for key and value
            self.sr_k = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.sr_v = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm_k = nn.LayerNorm(dim)
            self.norm_v = nn.LayerNorm(dim)

    def forward(self, x: Tensor, context: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        # Generate query from x
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        # Spatial reduction for context if sr_ratio is set
        if self.sr_ratio > 1:
            context = context.permute(0, 2, 1).reshape(B, C, H, W)
            context_k = self.sr_k(context).reshape(B, C, -1).permute(0, 2, 1)
            context_k = self.norm_k(context_k)
            context_v = self.sr_v(context).reshape(B, C, -1).permute(0, 2, 1)
            context_v = self.norm_v(context_v)
        else:
            context_k = context
            context_v = context

        # Generate key and value from context
        k = self.k(context_k).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)
        v = self.v(context_v).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Combine heads and project
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelAttentionP(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionP, self).__init__()
        # Typical channel attention uses global average pooling as a start
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Then a shared MLP (multi-layer perceptron) to produce channel-wise weights
        self.shared_mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        # Apply global average pooling and max pooling
        avg_values = self.avg_pool(x).view(B, C)
        max_values = self.max_pool(x).view(B, C)
        # Pass through the shared MLP
        avg_attention = self.shared_mlp(avg_values)
        max_attention = self.shared_mlp(max_values)
        # Combine the attentions and apply sigmoid to get weights between 0 and 1
        attention = self.sigmoid(avg_attention + max_attention)
        # Reshape and apply the channel attention weights to the input
        attention = attention.view(B, C, 1, 1)
        return x * attention.expand_as(x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


############################################
# backbone 部分
class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        """
        注意力头
        :param dim: 输入维度
        :param head: 注意力头数目
        :param sr_ratio: 缩放倍数
        """
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    """
    深度可分离卷积。

    """

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4):
        """
        下采样模块
        :param c1: 输入通道数
        :param c2: 输出通道数
        :param patch_size: patch 大小
        :param stride: 下采样倍数
        """
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size // 2)  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        """
        这是一个标准的transformer block。

        :param dim: 输入维度
        :param head: 注意力头的维度
        :param sr_ratio:
        :param dpr:
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],  # [embed_dims, depths]
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class MiT(nn.Module):
    def __init__(self, model_name: str = 'B0', in_channels: int = 3):
        super().__init__()
        assert model_name in mit_settings.keys(), f"MiT model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        drop_path_rate = 0.1
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(in_channels, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        # torch.Size([1, 3136, 64])
        for blk in self.block1:
            x = blk(x, H, W)
        # x= torch.Size([1, 3136, 64])
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)  # ([1, 64, 56, 56])

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x1, x2, x3, x4


class FFN(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims: list, image_size=[224, 224], embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        self.image_size = image_size
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i + 1}", FFN(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i + 2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        seg = F.interpolate(seg, size=self.image_size, mode='bilinear', align_corners=False)  # to original image shape
        return seg


# class projection_conv(nn.Module):
#     """
#     A non-linear neck in DenseCL
#     The non-linear neck, fc-relu-fc, conv-relu-conv
#     """

#     def __init__(self, in_dim, hid_dim=2048, out_dim=128, s=4):
#         super(projection_conv, self).__init__()
#         self.is_s = s
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.mlp = nn.Sequential(nn.Linear(in_dim, hid_dim),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(hid_dim, out_dim))
#         self.mlp_conv = nn.Sequential(nn.Conv2d(in_dim, hid_dim, 1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(hid_dim, out_dim, 1))
#         if self.is_s:
#             self.pool = nn.AdaptiveAvgPool2d((s, s))
#         else:
#             self.pool = None

#     def forward(self, x):
#         # Global feature vector
#         x1 = self.avgpool(x)
#         x1 = x1.reshape(x1.size(0), -1)
#         x1 = self.mlp(x1)

#         # dense feature map
#         if self.is_s:
#             x = self.pool(x)                        # [N, C, S, S]
#         x2 = self.mlp_conv(x)
#         x2 = x2.view(x2.size(0), x2.size(1), -1)    # [N, C, SxS]

#         return x1, x2

class projection_conv(nn.Module):
    """
    A non-linear neck in DenseCL
    The non-linear neck, fc-relu-fc, conv-relu-conv
    """

    def __init__(self, in_dim, hid_dim=2048, out_dim=128, s=4):
        super(projection_conv, self).__init__()
        self.is_s = s
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim)
        )
        self.mlp_conv = nn.Sequential(nn.Conv2d(in_dim, hid_dim, 1),
                                      #   nn.BatchNorm2d(hid_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hid_dim, out_dim, 1))
        if self.is_s:
            self.pool = nn.AdaptiveAvgPool2d((s, s))
        else:
            self.pool = None

    def forward(self, x):
        # Global feature vector
        x1 = self.avgpool(x)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.mlp(x1)

        # dense feature map
        if self.is_s:
            x = self.pool(x)  # [N, C, S, S]
        x2 = self.mlp_conv(x)
        x2 = x2.view(x2.size(0), x2.size(1), -1)  # [N, C, SxS]

        return x1, x2


class SegFormer(nn.Module):
    def __init__(self, image_size=[224, 224], in_channels=3, num_classes=4, model_name: str = 'B1'):
        super().__init__()
        self.encoder = MiT(model_name=model_name, in_channels=in_channels)
        self.decoder = SegFormerHead(self.encoder.embed_dims, image_size=image_size, embed_dim=256,
                                     num_classes=num_classes)

    def val(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class SegFormer_Plus(nn.Module):
    def __init__(self, image_size=[224, 224], in_channels=3, num_classes=4, model_name: str = 'B1'):
        super().__init__()
        self.encoder = MiT(model_name=model_name, in_channels=in_channels)
        self.decoder = SegFormerHead(self.encoder.embed_dims, image_size=image_size, embed_dim=256,
                                     num_classes=num_classes)
        self.dense_projection_high = projection_conv(self.encoder.embed_dims[-1])
        self.dense_projection_head = projection_conv(num_classes, hid_dim=1024)

    def val(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        high_feature = self.dense_projection_high(feature[-1])
        head_feature = self.dense_projection_head(output)
        return output, high_feature, head_feature


class SegFormer_UniMatch(nn.Module):
    def __init__(self, image_size=[224, 224], in_channels=3, num_classes=4, model_name: str = 'B1'):
        super().__init__()
        self.encoder = MiT(model_name=model_name, in_channels=in_channels)
        self.decoder = SegFormerHead(self.encoder.embed_dims, image_size=image_size, embed_dim=256,
                                     num_classes=num_classes)

    def val(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

    def forward(self, x, need_fp=False):
        feature = self.encoder(x)

        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)

        output = self.decoder(feature)
        return output


if __name__ == "__main__":
    # x = torch.rand(2, 1, 224, 224)
    model = SegFormer(in_channels=3)
    # output, high_feature, head_feature = model(x)
    # print(output.shape)
    # print(high_feature.shape)
    # print(head_feature.shape)

    from thop import profile

    # model = convnext_tiny(num_classes=5)
    input = torch.randn(2, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
