from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models import GeneratorUNet
from datasets import GeneDataset
import torch
import torchvision.transforms as transforms
import os


device = torch.device("cuda:4")
input_shape = (1, 256, 256)
path_to_images = "/root/lmy/aapm256/high"

transforms_ = [
    transforms.Resize((256, 256)),  # 确保图像是256x256
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]
batch_size = 32
dataset = GeneDataset(path_to_images, transforms_)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


def geneULDCT():
    # sample_images()
    b_generator = GeneratorUNet(1, 1).to(device)
    b_generator.load_state_dict(
        torch.load("mydualgan/saved_models/256/b_generator_100.pth")
    )
    b_generator.eval()

    output_folder = "/root/lmy/aapm256/gene_ULDCT"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, images in enumerate(dataloader):
        print(f"processing batch {i}")
        images = images.to(device)
        with torch.no_grad():
            reconstructed_imgs = b_generator(images)
        for j, img in enumerate(reconstructed_imgs):
            save_image(img, os.path.join(output_folder, f"{i*batch_size+j}.png"))


if __name__ == "__main__":
    geneULDCT()
    print("done")
