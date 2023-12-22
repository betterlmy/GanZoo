from datetime import datetime

import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from torch.autograd import Variable
import torch
import os
import wandb
from gan.model import Generator, Discriminator
from utils.CustomDataset import CDataset
from utils import config
from evalution import ssim, psnr, fid

config_file = "config_default.yaml"
configs = config.update_project_dir(config_file)
model_config = configs['model']['gan']
train_config = configs['train']
img_shape = (model_config['channels'], model_config['img_size'], model_config['img_size'])

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:" + configs['train']['gpu_id'] if cuda else "cpu")

adversarial_loss = torch.nn.BCELoss()  # 二分类交叉熵损失函数

generator = Generator(model_config['latent_dim'], img_shape)  # 生成器
discriminator = Discriminator(img_shape)  # 判别器
formatted_date = datetime.now().strftime("%m-%d-%H-%M")
use_wandb = configs['train']['use_wandb']
if use_wandb:
    wandb.init(project='gans', name='gan' + formatted_date, config=configs)

if cuda:
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    print("current device:" + str(device))

transform = transforms.Compose(
    [
        # transforms.Grayscale(),  # 如果图像不是灰度图像，添加此行
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

Tensor = torch.FloatTensor


def train():
    out_path = os.path.join(configs['project_dir'], "gan/output/")
    for epoch in range(train_config['n_epochs']):
        for i, imgs in enumerate(dataloader):
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device)

            real_imgs = Variable(imgs.type(Tensor)).to(device)
            optimizer_G.zero_grad()
            z = Variable(
                Tensor(np.random.normal(0, 1, (imgs.shape[0], model_config['latent_dim'])))
            ).to(device)
            gen_imgs = generator(z)

            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % train_config['print_interval'] == 0:
                print(
                    "[Epoch %d/%d]  [D loss: %f] [G loss: %f]"
                    % (
                        epoch,
                        train_config['n_epochs'],
                        d_loss.item(),
                        g_loss.item(),
                    )
                )

                if use_wandb:
                    wandb.log({
                        "Epoch": epoch,
                        "D loss": d_loss.item(),
                        "G loss": g_loss.item(),
                        "generated_images": [wandb.Image(img) for img in gen_imgs.data[:16]]
                    })

            if batches_done % train_config['model_save_interval'] == 0:
                save_image(
                    gen_imgs.data[:16],
                    out_path + "%d.png" % batches_done,
                    nrow=4,
                    normalize=True,
                )
                # 保存模型
                torch.save(generator.state_dict(), out_path + "gan_generator.pth")
                torch.save(discriminator.state_dict(), out_path + "gan_discriminator.pth")
                if use_wandb:
                    wandb.save(out_path + "gan_generator.pth", base_path=out_path)
                    wandb.save(out_path + "gan_discriminator.pth", base_path=out_path)
                # 计算SSIM和PSNR
                # dataset2 = CDataset(image_dir, transform)
                # psnr_value = 0
                # ssim_value = 0
                # for i in range(10):
                #     max_psnr = 0
                #     max_ssim = 0
                #     for j in range(100):
                #         real_imgs = dataset2[i]
                #         ssim_value = ssim(gen_imgs.data[i], real_imgs.to(device))
                #         psnr_value = psnr(gen_imgs.data[i], real_imgs.to(device))
                #         if psnr_value > max_psnr:
                #             max_psnr = psnr_value
                #         if ssim_value > max_ssim:
                #             max_ssim = ssim_value
                #
                #     psnr_value += max_psnr
                #     ssim_value += max_ssim
                #
                # print("SSIM: %f, PSNR: %f" % (ssim_value / 10, psnr_value / 10))


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("中断捕获，清理并关闭 wandb...")
        if use_wandb:
            wandb.finish()
    except Exception as e:
        print("发生错误:", e)
        if use_wandb:
            wandb.finish()
    if use_wandb:
        wandb.finish()
