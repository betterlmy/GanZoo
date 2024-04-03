from datetime import datetime

import torch
from PIL import Image
from DCUnet import DualChannelUnet
from ddpm import DCDDPM


def save_images_grid(tensor, filename):
    images = []
    for i in range(tensor.size(0)):
        # 处理每个图像
        img_tensor = tensor[i].mul(255).byte().cpu().numpy().squeeze(0)
        img = Image.fromarray(img_tensor, mode='L')
        images.append(img)

    total_width = sum([img.width for img in images])
    max_height = max([img.height for img in images])
    combined_img = Image.new('L', (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width
    combined_img.save(filename)


def sample(model=None):
    device = torch.device("cuda:0")
    if model is None:
        model = DCDDPM(DualChannelUnet(in_channels=1, out_channels=1, device=device))
        model.load_state_dict(torch.load(r"ddpm_model.pth"))
    else:
        device = model.unet.device
    noise_image = torch.randn(5, 1, 224, 224).to(device)
    generated_image = model.reverse_diffusion(noise_image)

    # 格式化日期和时间
    save_images_grid(generated_image, f'output/generated_image{datetime.now().strftime("%m-%d-%H-%M-%S")}.png')


if __name__ == "__main__":
    sample()
