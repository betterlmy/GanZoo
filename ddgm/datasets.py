import threading
import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import time
from concurrent.futures import ThreadPoolExecutor
import re


# from torchvision.utils import save_image


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


class ImageDatasetGPU1(Dataset):

    def __init__(
        self,
        rootA,
        transforms_=None,
        device="cuda",
        max_nums=0,
        max_workers=100,
    ):
        if transforms_ is None:
            transforms_ = [
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(rootA, "high") + "/*.png"))
        self.files_B = sorted(glob.glob(os.path.join(rootA, "low") + "/*.png"))
        self.TrueSDCTs = []
        self.TrueLDCTs = []
        self.FakeLDCTs = []
        self.FakeULDCTs = []
        max_nums = (
            len(self.files_A)
            if max_nums == 0 or max_nums > len(self.files_A)
            else max_nums
        )
        self.max_nums = max_nums
        print("数据集路径:", rootA)
        print("数据集:", self.max_nums)
        if self.max_nums > 0:
            indices = np.sort(
                np.random.choice(len(self.files_A), self.max_nums, replace=False)
            )
            self.files_A = [self.files_A[i] for i in indices]
            self.files_B = [self.files_B[i] for i in indices]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index in range(len(self.files_A)):
                executor.submit(self.process_images, index, device)
        time.sleep(1)
        print("dataset load to GPU successful")

    def process_images(self, index, device):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        image_A = image_A.convert("L")
        image_B = image_B.convert("L")

        TrueSDCT = self.transform(image_A).to(device)
        TrueLDCT = self.transform(image_B).to(device)
        FakeLDCT = add_poisson_noise(TrueSDCT, 90)
        FakeULDCT = add_poisson_noise(TrueSDCT, 50)

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
        return self.max_nums


class GeneDataset(Dataset):
    def __init__(
        self,
        root,
        transforms_,
    ):
        self.transforms_ = transforms.Compose(transforms_)
        files = glob.glob(os.path.join(root) + "/*.png")

        # 输出files_A 查看顺序
        def extract_number(file_path):
            basename = os.path.basename(file_path)  # 获取文件名
            number = re.search(r"\d+", basename)  # 使用正则表达式查找数字
            if number:
                return int(number.group())  # 返回数字部分转换成的整数
            return 0  # 如果没有找到数字，返回0

        self.files_A = sorted(files, key=extract_number)

        # for files in self.files_A:
        #     print(files)
        #     time.sleep(0.02)

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_A = self.transforms_(image_A)
        return image_A

    def __len__(self):
        return len(self.files_A)


# if __name__ == "__main__":
#     root = "/root/lmy/aapm256"
#     dataset = ImageDatasetGPU(root, max_nums=100)
#     print(len(dataset))
