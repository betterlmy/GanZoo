import torch
import torch.nn.functional as F
import pytorch_ssim


def psnr(source, target, max_pixel=1.0):
    with torch.no_grad:
        mse = F.mse_loss(source, target)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def ssim(source, target):
    with torch.no_grad:
        return pytorch_ssim.ssim(source, target)
