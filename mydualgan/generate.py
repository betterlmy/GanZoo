from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from mydualgan.models import GeneratorUNet
from mydualgan.datasets import ImageDatasetGPU1
from utils import config
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from evalution import ssim, psnr


device = torch.device("cuda:6")
input_shape = (1, 256, 256)

generator = GeneratorUNet(1, 1).to(device)


generator.load_state_dict(
    torch.load("mydualgan/saved_models/aapm/db_generator_1900.pth")
)

transforms_ = [
    # 数据增强
    transforms.Resize((), Image.Resampling.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]


aapm_data = ImageDatasetGPU1("/root/lmy/aapm512", device=device, max_nums=0)


val_dataloader = DataLoader(
    aapm_data,
    batch_size=5,
    shuffle=False,
)


def sample_images():
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    generator.eval()

    real_A = imgs["A"].to(device)
    real_B = imgs["B"].to(device)
    fake_A = generator(real_B)
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    image_grid = torch.cat((real_B, fake_A, real_A), 1)
    ssim_score = ssim.ssim(fake_A, real_A)
    psnr_score = psnr.psnr(fake_A, real_A)

    save_image(
        image_grid,
        "pix2pix/outputs/%s/eval/out.png" % (model_config["dataset_name"]),
        normalize=False,
    )
    print("save image success, ssim: %f, psnr: %f" % (ssim_score, psnr_score))
    return image_grid, ssim_score, psnr_score


def gene():
    # sample_images()
    generator.eval()


if __name__ == "__main__":
    gene()
