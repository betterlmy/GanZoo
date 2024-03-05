from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from mydualgan.models import GeneratorUNet
from mydualgan.datasets import ImageDataset
from utils import config
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from evalution import ssim, psnr

config_file = "config_default.yaml"
configs = config.update_project_dir(config_file)
model_config = configs['model']['mydualgan']
train_config = configs['train']
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
input_shape = (1, 256, 256)

generator = GeneratorUNet(model_config['channels'], model_config['channels'])

if cuda:
    generator = generator.to(device)
    print("current device:" + str(device))

generator.load_state_dict(
    torch.load("mydualgan/saved_models/%s/generator_%d.pth" % (model_config['dataset_name'], 1900)))
transforms_ = [
    # 数据增强
    transforms.Resize(int(model_config['img_size'] * 1.12), Image.Resampling.BICUBIC),
    transforms.RandomCrop((model_config['img_size'], model_config['img_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]

val_data = ImageDataset(os.path.join(configs['project_dir'], 'dataset', model_config['dataset_name']),
                        transforms_=transforms_,
                        unaligned=False)
val_dataloader = DataLoader(
    val_data,
    batch_size=5,
    shuffle=True,
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

    save_image(image_grid, "pix2pix/outputs/%s/eval/out.png" % (model_config['dataset_name']), normalize=False)
    print("save image success, ssim: %f, psnr: %f" % (ssim_score, psnr_score))
    return image_grid, ssim_score, psnr_score


def gene():
    sample_images()


if __name__ == '__main__':
    gene()
