import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch
import os

from utils.CustomDataset import CDataset
from utils import config
from evalution import ssim, psnr, fid

config_file = "config_default.yaml"
configs = config.update_project_dir(config_file)
model_config = configs['model']['gan']
train_config = configs['train']
img_shape = (model_config['channels'], model_config['img_size'], model_config['img_size'])

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:" + configs['train'].gpu_id if cuda else "cpu")
print("current device:" + str(device))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(model_config['latent_dim'], 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

if cuda:
    device = torch.device("cuda:0")
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

transform = transforms.Compose(
    [
        # transforms.Grayscale(),  # 如果图像不是灰度图像，请添加此行
        transforms.Resize((model_config['img_size'], model_config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
image_dir = os.path.join(configs['project_dir'], configs['dataset_relative_path'])
dataset = CDataset(
    image_dir, transform
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=train_config['batch_size'],
    shuffle=True,
)

optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=model_config['lr'], betas=(model_config['b1'], model_config['b2'])
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=model_config['lr'], betas=(model_config['b1'], model_config['b2'])
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train():
    out_path = os.path.join(configs['project_dir'], "gan/output/")
    for epoch in range(train_config['n_epochs']):
        for i, imgs in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 1, (imgs.shape[0], model_config['latent_dim'])))
            )

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % train_config['sample_interval'] == 0:
                print(
                    "[Epoch %d/%d]  [D loss: %f] [G loss: %f]"
                    % (
                        epoch,
                        train_config['n_epochs'],

                        d_loss.item(),
                        g_loss.item(),
                    )
                )
                save_image(
                    gen_imgs.data[:25],
                    out_path + "%d.png" % batches_done,
                    nrow=5,
                    normalize=True,
                )

                # 计算SSIM和PSNR
                dataset2 = CDataset(image_dir, transform)
                psnr_value = 0
                ssim_value = 0
                for i in range(10):
                    max_psnr = 0
                    max_ssim = 0
                    for j in range(100):
                        real_imgs = dataset2[i]
                        ssim_value = ssim(gen_imgs.data[i], real_imgs.to(device))
                        psnr_value = psnr(gen_imgs.data[i], real_imgs.to(device))
                        if psnr_value > max_psnr:
                            max_psnr = psnr_value
                        if ssim_value > max_ssim:
                            max_ssim = ssim_value

                    psnr_value += max_psnr
                    ssim_value += max_ssim

                print("SSIM: %f, PSNR: %f" % (ssim_value / 10, psnr_value / 10))


if __name__ == "__main__":
    train()
