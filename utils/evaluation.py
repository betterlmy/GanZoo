import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import matplotlib.pyplot as plt


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 10).unsqueeze(1)
    # plt.figure(figsize=(10, 4))
    # plt.plot(_1D_window, label="Gaussian Window")
    # plt.title("1D Gaussian Window Plot")
    # plt.xlabel("Position")
    # plt.ylabel("Weight")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    with torch.no_grad():
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    """SSIM 的值在 0 到 1 之间，1 表示两个比较的图像是完全相同的。"""
    if len(img1.shape) == 3:
        channel = img1.shape[0]
    else:
        channel = img1.shape[1]
    window = create_window(window_size, channel)
    window = window.to(img1.device)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(source, target, max_pixel=1.0):
    """
    高 PSNR 值 (>40 dB) 通常表示图像质量很高，误差很小。
    中等 PSNR 值 (30-40 dB) 可能表示图像质量良好，但存在一些可见的误差。
    低 PSNR 值 (<30 dB) 通常表示图像质量较差，误差较大。
    """
    with torch.no_grad():
        mse = F.mse_loss(source, target)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def heatmap_window(window):
    # 绘制热力图
    plt.imshow(window, cmap='hot')
    plt.colorbar()
    plt.title("Window Heatmap")
    plt.show()


if __name__ == '__main__':
    window = create_window(256, 1)
    heatmap_window(window.squeeze(0).squeeze(0))
