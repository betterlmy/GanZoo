# 实验目的

使用模型进行医学CT影像重建，达到对带有噪声的低剂量CT影像进行修复获得相对清晰的影像

https://github.com/eriklindernoren/PyTorch-GAN这个仓库使用torch实现了许多GAN的模型
1. **GAN**: 基本的生成对抗网络。
2. **Conditional GAN (CGAN)**: 生成特定条件或类别的图像。
3. **Deep Convolutional GAN (DCGAN)**: 使用深度卷积网络的GAN，用于生成高质量图像。
4. **Enhanced Super-Resolution GAN (ESRGAN)**: 用于图像超分辨率。
5. **Pix2Pix**: 用于成对的图像到图像转换。
6. **Wasserstein GAN DIV (WGAN-DIV)**: 一种WGAN的变体，使用散度项来改善训练。

# 实验步骤

## 数据处理

### 生成成对的模糊与清晰图像（celebA）

适用于图像超分辨率、去噪和图像修复等深度学习任务。这个过程包括以下步骤：

1. **降低分辨率**：首先，将原始的256x256分辨率的图像压缩到64x64分辨率。这一步会导致图像丢失细节和清晰度，从而产生低质量的图像。
2. **上采样**：然后，使用插值方法（如双线性插值、双三次插值等）将这些低分辨率的图像重新放大到256x256分辨率。这一步会产生模糊的图像，因为插值算法会尝试填补缩小过程中丢失的信息，但无法完全恢复原始图像的细节和清晰度。
3. **生成成对数据**：最后，将每个原始的高分辨率（清晰）图像与其对应的通过上述过程生成的低分辨率（模糊）版本配对。这样，就得到了成对的数据集，其中包含了清晰和模糊的图像。

## 评价指标

### psnr

### ssim

### fid

低分数：理想情况下，较低的 FID 分数（接近 0）意味着生成图像的质量高，与真实图像集很相似。在实践中，一个较低的 FID
分数通常被视为生成模型表现良好的指标。

高分数：较高的 FID 分数表明生成图像与真实图像集之间存在较大差异，可能意味着生成图像的质量较低。

Inception v3 模型通常要求输入图像的尺寸为 3x299x299 像素。
需要：

1. 调整图像尺寸：使用图像处理方法（如双线性插值）将图像调整为 299x299 像素。

2. 调整通道数量：如果图像是单通道的（如灰度图），可以将其转换为三通道图像。这可以通过复制单通道三次来实现。
   转换代码

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# 假设 img 是一个 1x256x256 的灰度图像张量
img = torch.randn(1, 256, 256)  # 示例图像

# 将灰度图像转换为 RGB 图像
img_rgb = img.repeat(3, 1, 1)  # 复制通道

# 或者 image = Image.open(img_path).convert('RGB')

# 创建一个转换管道，包括调整尺寸和归一化
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 应用转换
img_transformed = transform(img_rgb)
# 现在 img_transformed 是一个 3x299x299 的张量，可以被 Inception v3 模型处理

```

计算步骤：

1. 计算均值和协方差：
   mu1, sigma1：计算第一组激活（act1，通常来自真实图像）的均值和协方差。
   mu2, sigma2：计算第二组激活（act2，通常来自生成图像）的均值和协方差。
   这里的均值和协方差是对特征空间中的特征分布进行的统计描述。

2. 计算 Fréchet 距离：
   ssdiff = np.sum((mu1 - mu2) ** 2.0)：计算两组均值之间的平方差。
   covmean = sqrtm(sigma1.dot(sigma2))：计算两组协方差矩阵乘积的平方根，这是 Fréchet 距离的一部分。
   fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)：计算最终的 FID 分数。这个公式是基于高斯分布的 Fréchet
   距离的计算方法，涉及均值差异和协方差矩阵。

3. 处理复数值：
   在计算 sqrtm 时，有时可能会产生复数结果，即使理论上结果应该是实数。因此，代码中检查 covmean 是否为复数对象，如果是，只取其实数部分。

### 具体实现代码

[fid.py](evalution/fid.py)

## GAN模型


## 基础GAN 2014年

为了测试 先使用celebA数据集，部分训练效果如下
![14800.png](gan%2Foutput1205celeba%2F15600.png)

CT数据集训练效果如下
![9600.png](gan%2Foutput1207ct%2F9600.png)

## CGAN条件模型
通过添加onehot（也就是条件）给gan训练，实现控制gan的输出
训练效果如下

![0.png](cgan%2Foutput%2F0.png) ![1.png](cgan%2Foutput%2F1.png)![2.png](cgan%2Foutput%2F2.png) ![3.png](cgan%2Foutput%2F3.png)![4.png](cgan%2Foutput%2F4.png) ![5.png](cgan%2Foutput%2F5.png) ![6.png](cgan%2Foutput%2F6.png)![7.png](cgan%2Foutput%2F7.png) ![8.png](cgan%2Foutput%2F8.png)![9.png](cgan%2Foutput%2F9.png)