import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=64):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(output_channels, output_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels * 2)

    def forward(self, x):
        x = self.conv1(x)
        print(f"Shape after conv1: {x.shape}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        print(f"Shape after pool1: {x.shape}")

        x = self.conv2(x)
        print(f"Shape after conv2: {x.shape}")
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        print(f"Shape after pool2: {x.shape}")

        return x


# 假设我们的输入是单通道的256x256的CT图像
input_image = torch.rand(1, 1, 256, 256)  # Batch size 1

# 创建模型实例
cnn_model = SimpleCNN(input_channels=1, output_channels=64)

# 前向传播，获取CNN的输出特征图
cnn_features = cnn_model(input_image)
