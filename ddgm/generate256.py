from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models import GeneratorUNet
from datasets import GeneDataset
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

device = torch.device("cuda:4")
input_shape = (1, 256, 256)
path_to_images = "/root/lmy/aapm256/high"

transforms_ = [
    transforms.transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]
batch_size = 32
dataset = GeneDataset(path_to_images, transforms_)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=4,
)


def geneULDCT():
    # sample_images()
    b_generator = GeneratorUNet(1, 1).to(device)
    b_generator.load_state_dict(
        torch.load(
            "mydualgan/saved_models/256/b_generator_1200.pth", map_location=device
        )
    )
    b_generator.eval()
    print("model loaded")

    output_folder = "/root/lmy/aapm256/gene_ULDCT"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, images in enumerate(dataloader):
        print(f"processing batch {i}")
        images = images.to(device)
        with torch.no_grad():
            reconstructed_imgs = b_generator(images)
            reconstructed_imgs = (reconstructed_imgs + 1) / 2
            reconstructed_imgs = reconstructed_imgs.clamp(0, 1).detach().cpu()
        for j, img in enumerate(reconstructed_imgs):
            # 将0到1之间的值转换为0到255之间的整数
            img_array_255 = (img.squeeze(0).numpy() * 255).astype(np.uint8)

            # 将numpy数组转换为PIL图像
            img_pil = Image.fromarray(img_array_255)
            img_pil.save(os.path.join(output_folder, f"{i*batch_size+j}.png"))
            # save_image(img, os.path.join(output_folder, f"{i*batch_size+j}.png"),cmap='gray')


def geneNDCT():
    # sample_images()
    dataset = GeneDataset("/root/lmy/aapm256/gene_ULDCT", transforms_)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=4,
    )

    db_generator = GeneratorUNet(1, 1).to(device)
    db_generator.load_state_dict(
        torch.load(
            "mydualgan/saved_models/256/db_generator_1500.pth", map_location=device
        )
    )
    db_generator.eval()
    print("model loaded")

    output_folder = "/root/lmy/aapm256/gene_NDCT"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, images in enumerate(dataloader):
        print(f"processing batch {i}")
        images = images[:, 0, :, :].unsqueeze(1).to(device)
        with torch.no_grad():
            reconstructed_imgs = db_generator(images)
            reconstructed_imgs = (reconstructed_imgs + 1) / 2
            reconstructed_imgs = reconstructed_imgs.clamp(0, 1).detach().cpu()

        for j, img in enumerate(reconstructed_imgs):
            # 将0到1之间的值转换为0到255之间的整数
            img_array_255 = (img.squeeze(0).numpy() * 255).astype(np.uint8)

            # 将numpy数组转换为PIL图像
            img_pil = Image.fromarray(img_array_255)
            img_pil.save(os.path.join(output_folder, f"{i*batch_size+j}.png"))


if __name__ == "__main__":
    geneULDCT()
    geneNDCT()
    print("done")
