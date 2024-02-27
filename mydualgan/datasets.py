import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    """Converts image to rgb if it is grayscale"""
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    """p2p要求成对的数据集"""

    def __init__(self, rootA, transforms_=None, unaligned=False, mode="train", rgb=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(rootA, "high") + "/*.png"))
        self.files_B = sorted(glob.glob(os.path.join(rootA, "low") + "/*.png"))
        self.rgb = rgb

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB" and self.rgb:
            image_A = to_rgb(image_A)
        else:
            image_A = image_A.convert("L")
        if image_B.mode != "RGB" and self.rgb:
            image_B = to_rgb(image_B)
        else:
            image_B = image_B.convert("L")
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
