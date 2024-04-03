import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CTImagesDataset(Dataset):
    def __init__(self, high_dir, low_dir, transform=None):
        self.high_dir = high_dir
        self.low_dir = low_dir
        self.transform = transform

        self.high_images = [file for file in os.listdir(high_dir) if file.endswith('.png')]
        self.low_images = [file for file in os.listdir(low_dir) if file.endswith('.png')]
        self.length_dataset = min(len(self.high_images), len(self.low_images))  # 保证high和low数量一致
        if self.length_dataset == 0:
            raise Exception("\ndata not found in the directory. \n"
                            "Please check your directory input. \n"
                            "high_dir: {}\n"
                            "low_dir: {}\n".format(self.high_dir, self.low_dir))

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        high_img_path = os.path.join(self.high_dir, "LD" + str(idx+1) + ".png")
        low_img_path = os.path.join(self.low_dir, "ULD" + str(idx+1) + ".png")

        high_img = Image.open(high_img_path)
        low_img = Image.open(low_img_path)

        if self.transform:
            high_img = self.transform(high_img)
            low_img = self.transform(low_img)

        return high_img, low_img


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    high_dir = 'B301MM/high'
    low_dir = 'B301MM/low'

    ct_dataset = CTImagesDataset(high_dir, low_dir, transform)
    ct_dataloader = DataLoader(ct_dataset, batch_size=4, shuffle=True)
    for high_img, low_img in ct_dataloader:
        pass
