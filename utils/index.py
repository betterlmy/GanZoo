import torch
import torch.nn.functional as F


def psnr(source, target=None, max_pixel=1.0):
    with torch.no_grad:
        mse = F.mse_loss(source, target)
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr
