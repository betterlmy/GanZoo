from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

from utils.CustomDataset import CDataset
from utils.config import load_config
import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def calculate_fid(act1, act2):
    # 计算两组激活的均值和协方差
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # 计算 Fréchet 距离
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    # 防止复数值
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_activations(images, model, batch_size=100):
    n_batches = len(images) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        batch = images[start:end].to(images[0].device)
        with torch.no_grad():
            pred = model(batch)
        pred_arr[start:end] = pred.cpu().numpy().reshape(batch_size, -1)

    return pred_arr


def get_activations_dataloader(dataloader, model, dims=2048):
    """从dataloader中提取数据计算激活层"""
    activations = []
    for imgs in dataloader:
        if torch.cuda.is_available():
            imgs = imgs.to(model.device)

        with torch.no_grad():
            features = model(imgs)
            # 如果模型包含辅助输出，则仅使用主输出
            if isinstance(features, tuple):
                features = features[0]

            # 您可能需要根据您的模型架构调整这里
            # 以确保特征的维度是正确的
            activations.append(features.view(imgs.size(0), -1).cpu())

    # 将所有批次的特征合并成一个大的 numpy 数组
    activations = torch.cat(activations, dim=0).numpy()
    return activations


def fid(gen_images, real_dataset_path, real_activations_path='real_ct_activations.pt'):
    # 加载预训练的 Inception v3 模型
    model = inception_v3(pretrained=True, transform_input=False).to(gen_images[0].device)
    model.fc = nn.Identity()  # 将最后的全连接层替换为恒等映射
    model.eval()
    if not os.path.exists(real_activations_path):
        # 计算并保存真实图像的激活
        transform = transforms.Compose(
            [

                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        real_images = CDataset(real_dataset_path, transform, channels=3)  # inceptionv3要求输入3通道
        dataloader = torch.utils.data.DataLoader(
            real_images,
            batch_size=len(gen_images),
            shuffle=False,
        )
        real_activations = get_activations_dataloader(dataloader, model)
        torch.save(real_activations, real_activations_path)
    else:
        # 后续使用时，只需加载已保存的激活
        real_activations = torch.load(real_activations_path)

    # 对于生成的图像，每次评估时计算激活并计算 FID
    gen_activations = get_activations(gen_images, model)
    fid_value = calculate_fid(real_activations, gen_activations)
    return fid_value


if __name__ == "__main__":
    gen_images = torch.randn(205, 3, 256, 256)
    config = load_config()
    real_dataset_path = os.path.join(config['project_dir'], config['dataset_relative_path'])
    fid(gen_images, real_dataset_path)
