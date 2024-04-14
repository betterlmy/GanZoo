import torch as th
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import math
import matplotlib.pyplot as plt

# 设置设备
device = th.device('cuda:5' if th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 定义自适应调整的计算
def calculate_adaptive_adjustment(weights, seq_length):
    adjustment = th.var(weights, dim=2, keepdim=True)
    adjustment = (adjustment - adjustment.min()) / (adjustment.max() - adjustment.min())
    adjustment = adjustment.view(-1, 1, seq_length)
    print(f"Adaptive adjustment shape: {adjustment.shape}")
    return adjustment


# 定义注意力计算函数
def attention_from_qkv(qkv, num_heads):
    q, k, v = qkv[0], qkv[1], qkv[2]
    bs_times_num_heads, ch, seq_length = q.shape
    scale = 1 / math.sqrt(ch)
    weight = th.einsum('bct,bcs->bts', q * scale, k * scale)
    print(f"Attention weight shape before adjustment: {weight.shape}")
    adaptive_adjustment = calculate_adaptive_adjustment(weight, seq_length)
    adaptive_weight = weight * (1 - adaptive_adjustment)
    adaptive_weight = th.softmax(adaptive_weight.float(), dim=-1).type(weight.dtype)
    print(f"Attention weight shape after adjustment: {adaptive_weight.shape}")
    return adaptive_weight


# 定义一个简化的QKV生成器
class QKVGenerator(nn.Module):
    def __init__(self, in_channels, out_channels_per_head, num_heads):
        super(QKVGenerator, self).__init__()
        total_out_channels = out_channels_per_head * num_heads * 3
        self.conv = nn.Conv2d(in_channels, total_out_channels, kernel_size=1)
        self.num_heads = num_heads
        self.out_channels_per_head = out_channels_per_head
        print(
            f"QKVGenerator initialized with total_out_channels: {total_out_channels}, num_heads: {num_heads}, out_channels_per_head: {out_channels_per_head}")

    def forward(self, x):
        bs, c, h, w = x.shape
        qkv = self.conv(x)
        qkv = qkv.permute(0, 2, 3, 1).view(bs, -1, 3, self.num_heads, self.out_channels_per_head).permute(2, 0, 3, 1, 4)
        qkv = qkv.reshape(3, bs * self.num_heads, self.out_channels_per_head, -1)
        print(f"QKV shape after reshape: {qkv.shape}")
        return qkv


# 加载和预处理图像
image_path = 'result/newmodel670000/00000.png'  # 更改为您的图像路径
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
x = transform(image).unsqueeze(0).to(device)
print(f"Image tensor shape after transform: {x.shape}")

# 初始化QKV生成器并应用
qkv_generator = QKVGenerator(in_channels=3, out_channels_per_head=32, num_heads=4).to(device)
qkv = qkv_generator(x)

# 计算注意力图
attention_map = attention_from_qkv(qkv, 4)

attention_map_cpu = attention_map[0].cpu().detach().numpy()

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, to_tensor

# 假设attention_map已经计算好并且已经转移到了CPU
attention_map_cpu = attention_map[0].cpu().detach().numpy()

# 将注意力图正规化到[0, 1]范围内
attention_map_normalized = (attention_map_cpu - np.min(attention_map_cpu)) / (
            np.max(attention_map_cpu) - np.min(attention_map_cpu))

# 将原始图像转换为Tensor，并确保它在同一设备上
original_image_tensor = to_tensor(image).to(attention_map.device)

# 调整注意力图大小以匹配原始图像的尺寸
# 假设original_image_tensor已经在某个特定的CUDA设备上
# 首先，确保attention_map_resized也在同一个设备上
attention_map_resized = torch.tensor(
    np.resize(attention_map_normalized, (original_image_tensor.shape[1], original_image_tensor.shape[2])),
    device=original_image_tensor.device)

# 现在，使用注意力图加权原始图像时，两个张量都在同一设备上
weighted_image = original_image_tensor * attention_map_resized.unsqueeze(0)  # 可能需要添加一个维度，以匹配原始图像的通道数

# 将加权后的图像转换回PIL图像以便可视化
weighted_image_pil = to_pil_image(weighted_image.cpu())  # 转回CPU以便使用PIL和matplotlib进行可视化

# 显示加权后的图像
plt.imshow(weighted_image_pil)
plt.title('Weighted Original Image by Attention Map')
plt.axis('off')  # 关闭坐标轴
plt.savefig('attention_overlay_visualization1.png')
