import threading
import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image


def to_rgb(image):
    """Converts image to rgb if it is grayscale"""
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def add_poisson_noise(image, scale=1.0):
    """
    Apply Poisson noise to the input image.
    :param image: input image
    :param scale: scale factor for the input image
    :return: noisy image
    """
    image = (image + 1) / 2  # 归一化到[0,1]

    scaled_image = image * scale
    noisy_image = torch.poisson(scaled_image) / scale
    # save_image(noisy_image, 'noisy_image.png')

    return noisy_image * 2 - 1  # 反归一化到(-1,1)


class ImageDataset(Dataset):
    """p2p要求成对的数据集"""

    def __init__(
        self,
        rootA,
        transforms_=[
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # 单通道
        ],
        unaligned=False,
        rgb=False,
        max_nums=None,
    ):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(rootA, "high") + "/*.png"))
        self.files_B = sorted(glob.glob(os.path.join(rootA, "low") + "/*.png"))
        if max_nums:
            indices = np.random.choice(len(self.files_A), max_nums, replace=False)
            self.files_A = [self.files_A[i] for i in indices]
            self.files_B = [self.files_B[i] for i in indices]

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

        TrueSDCT = self.transform(image_A)
        TrueLDCT = self.transform(image_B)
        # save_image(TrueSDCT, 'image1.png')

        FakeLDCT = add_poisson_noise(TrueSDCT, 70)
        FakeULDCT = add_poisson_noise(TrueSDCT, 50)

        return {
            "TrueSDCT": TrueSDCT,
            "TrueLDCT": TrueLDCT,
            "FakeLDCT": FakeLDCT,
            "FakeULDCT": FakeULDCT,
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDatasetGPU(Dataset):
    """p2p要求成对的数据集"""

    def __init__(
        self,
        rootA,
        transforms_=[
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # 单通道
        ],
        device="cuda:6",
        max_nums=None,
    ):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(rootA, "high") + "/*.png"))
        self.files_B = sorted(glob.glob(os.path.join(rootA, "low") + "/*.png"))
        self.TrueSDCTs = []
        self.TrueLDCTs = []
        self.FakeLDCTs = []
        self.FakeULDCTs = []

        if max_nums:
            indices = np.random.choice(len(self.files_A), max_nums, replace=False)
            self.files_A = [self.files_A[i] for i in indices]
            self.files_B = [self.files_B[i] for i in indices]

        for index in range(len(self.files_A)):
            image_A = Image.open(self.files_A[index % len(self.files_A)])
            image_B = Image.open(self.files_B[index % len(self.files_B)])

            image_A = image_A.convert("L")
            image_B = image_B.convert("L")

            TrueSDCT = self.transform(image_A).to(device)
            TrueLDCT = self.transform(image_B).to(device)
            FakeLDCT = add_poisson_noise(TrueSDCT, 70).to(device)
            FakeULDCT = add_poisson_noise(TrueSDCT, 50).to(device)
            self.TrueSDCTs.append(TrueSDCT)
            self.TrueLDCTs.append(TrueLDCT)
            self.FakeLDCTs.append(FakeLDCT)
            self.FakeULDCTs.append(FakeULDCT)
        print("数据集加载到gpu完成")

    def __getitem__(self, index):

        return {
            "TrueSDCT": self.TrueSDCTs[index],
            "TrueLDCT": self.TrueLDCTs[index],
            "FakeLDCT": self.FakeLDCTs[index],
            "FakeULDCT": self.FakeULDCTs[index],
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDatasetGPU1(Dataset):

    def __init__(
        self,
        rootA,
        transforms_=[
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), 
        ],
        device="cuda",
        max_nums=None,
    ):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(rootA, "high") + "/*.png"))
        self.files_B = sorted(glob.glob(os.path.join(rootA, "low") + "/*.png"))
        self.TrueSDCTs = []
        self.TrueLDCTs = []
        self.FakeLDCTs = []
        self.FakeULDCTs = []

        if max_nums:
            print(max_nums)
            indices = np.random.choice(len(self.files_A), max_nums, replace=False)
            self.files_A = [self.files_A[i] for i in indices]
            self.files_B = [self.files_B[i] for i in indices]

        # Create and start threads
        threads = []
        for index in range(len(self.files_A)):
            thread = threading.Thread(target=self.process_images, args=(index, device))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        print(max_nums,"dataset load to GPU successful")

    def process_images(self, index, device):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        image_A = image_A.convert("L")
        image_B = image_B.convert("L")

        TrueSDCT = self.transform(image_A).to(device)
        TrueLDCT = self.transform(image_B).to(device)
        FakeLDCT = add_poisson_noise(TrueSDCT, 70).to(device)
        FakeULDCT = add_poisson_noise(TrueSDCT, 50).to(device)

        # Append processed images to lists
        self.TrueSDCTs.append(TrueSDCT)
        self.TrueLDCTs.append(TrueLDCT)
        self.FakeLDCTs.append(FakeLDCT)
        self.FakeULDCTs.append(FakeULDCT)

    def __getitem__(self, index):
        return {
            "TrueSDCT": self.TrueSDCTs[index],
            "TrueLDCT": self.TrueLDCTs[index],
            "FakeLDCT": self.FakeLDCTs[index],
            "FakeULDCT": self.FakeULDCTs[index],
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
if __name__ == "__main__":
    root = "/root/lmy/aapm256"
    dataset = ImageDatasetGPU(root, max_nums=100)
    print(len(dataset))
