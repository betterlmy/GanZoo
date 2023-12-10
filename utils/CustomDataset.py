from torch.utils.data import Dataset
from PIL import Image
import os


class CDataset(Dataset):
    def __init__(self, image_dir, transform=None,channels=1):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.images.sort()
        self.channels = channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        if self.channels == 1:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
