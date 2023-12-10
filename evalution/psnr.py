import torch
import torch.nn.functional as F


def psnr(source, target, max_pixel=1.0):
    """
    高 PSNR 值 (>40 dB) 通常表示图像质量很高，误差很小。
    中等 PSNR 值 (30-40 dB) 可能表示图像质量良好，但存在一些可见的误差。
    低 PSNR 值 (<30 dB) 通常表示图像质量较差，误差较大。
    """
    with torch.no_grad():
        mse = F.mse_loss(source, target)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))
