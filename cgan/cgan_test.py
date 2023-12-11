import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os

# GPU
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dimension = 100
# 图形处理过程
img_transform = transforms.Compose([
    transforms.ToTensor(),
])


####### 定义生成器 Generator #####
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dimension + 10, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )

    def forward(self, x):
        x = self.gen(x)
        return x


# 创建对象
G = generator()
G = G.to(device)
G.load_state_dict(torch.load('./generator_CGAN_z100.pth'))


def gen_one_num(num):
    gen_number = 9  # 生成9张图片
    label = torch.Tensor([num]).repeat(gen_number).long()
    label_onehot = torch.zeros((gen_number, 10))
    label_onehot[torch.arange(gen_number), label] = 1
    z = torch.randn((gen_number, z_dimension))
    z = torch.cat([z, label_onehot], 1).to(device)
    gens = G(z).view(z.shape[0], 1, 28, 28)
    return gens


for num in range(10):
    gens = gen_one_num(num).cpu()
    save_image(
        gens.data,
        "output/%d.png" % num,
        nrow=3,
        normalize=True,  # 将图片的像素值映射到【-1,1】之间
    )
