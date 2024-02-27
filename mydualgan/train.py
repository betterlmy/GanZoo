import itertools
from datetime import datetime, timedelta
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from mydualgan.datasets import ImageDataset
from mydualgan.models import GeneratorUNet, Discriminator, weights_init_normal
from evalution import ssim, psnr
from utils import config
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import wandb

"""！！！训练时 将workdir设置为GanZoo的根目录 而非mydualgan的根目录"""

config_file = "config_default.yaml"
configs = config.update_project_dir(config_file)
model_config = configs['model']['mydualgan']
train_config = configs['train']
use_wandb = train_config['use_wandb']
formatted_date = datetime.now().strftime("%m-%d-%H-%M")

if use_wandb:
    wandb.init(project='gans', name='mydualgan' + formatted_date, config=configs)

os.makedirs("mydualgan/outputs/%s" % model_config['dataset_name'],
            exist_ok=True)
os.makedirs("mydualgan/saved_models/%s" % model_config['dataset_name'], exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, model_config['img_size'] // 2 ** 4, model_config['img_size'] // 2 ** 4)

db_generator = GeneratorUNet(model_config['channels'], model_config['channels'])
db_discriminator = Discriminator(model_config['channels'])

b_generator = GeneratorUNet(model_config['channels'], model_config['channels'])
b_discriminator = Discriminator(model_config['channels'])

cuda = torch.cuda.is_available()
device = torch.device("cuda:" + train_config['gpu_id'] if cuda else "cpu")

if cuda:
    db_generator = db_generator.to(device)
    db_discriminator = db_discriminator.to(device)
    
    b_generator = b_generator.to(device)
    b_discriminator = b_discriminator.to(device)
    criterion_GAN.to(device)
    criterion_pixelwise.to(device)
    print("current device:" + str(device))

if model_config['epoch'] != 0:
    # Load pretrained models
    db_generator.load_state_dict(
        torch.load("saved_models/%s/db_generator_%d.pth" % (train_config['dataset_name'], train_config['epoch'])))
    db_discriminator.load_state_dict(
        torch.load("saved_models/%s/db_discriminator_%d.pth" % (train_config['dataset_name'], train_config['epoch'])))
    b_generator.load_state_dict(
        torch.load("saved_models/%s/b_generator_%d.pth" % (train_config['dataset_name'], train_config['epoch'])))
    b_discriminator.load_state_dict(
        torch.load("saved_models/%s/b_discriminator_%d.pth" % (train_config['dataset_name'], train_config['epoch'])))

else:
    # Initialize weights
    db_generator.apply(weights_init_normal)
    db_discriminator.apply(weights_init_normal)
    b_generator.apply(weights_init_normal)
    b_discriminator.apply(weights_init_normal)

# Optimizers
optimizer_DBG = torch.optim.Adam(db_generator.parameters(), lr=model_config['lr'],
                               betas=(model_config['b1'], model_config['b2']))
optimizer_DBD = torch.optim.Adam(db_discriminator.parameters(), lr=model_config['lr'],
                               betas=(model_config['b1'], model_config['b2']))
optimizer_BG = torch.optim.Adam(b_generator.parameters(), lr=model_config['lr'],
                               betas=(model_config['b1'], model_config['b2']))
optimizer_BD = torch.optim.Adam(b_discriminator.parameters(), lr=model_config['lr'],
                               betas=(model_config['b1'], model_config['b2']))

transforms_ = [
    transforms.Resize((model_config['img_size'], model_config['img_size']), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # 单通道
]

dataloader = DataLoader(
    ImageDataset(os.path.join(configs['project_dir'], 'dataset', model_config['dataset_name']),
                 transforms_=transforms_),
    batch_size=model_config['batch_size'],
    shuffle=True,
)

val_dataloader = DataLoader(
    ImageDataset(os.path.join(configs['project_dir'], 'dataset', model_config['dataset_name']),
                 transforms_=transforms_),
    batch_size=5,
    shuffle=False,
)

test_dataloader = DataLoader(
    ImageDataset(os.path.join(configs['project_dir'], 'dataset', model_config['dataset_name']),
                 transforms_=transforms_),
    batch_size=5,
    shuffle=False,
)



## 待修改
def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    train_imgs = next(iter(val_dataloader))
    real_A = train_imgs["A"].to(device)
    real_B = train_imgs["B"].to(device)
    fake_A = db_generator(real_B)

    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    train_image_grid = torch.cat((real_B, fake_A, real_A), 1)


    save_image(train_image_grid, "mydualgan/outputs/%s/%s.png" % (model_config['dataset_name'], batches_done), nrow=5,
               normalize=True)
    ssim_score = ssim.ssim(fake_A, real_A)
    psnr_score = psnr.psnr(fake_A, real_A)
    return train_image_grid, ssim_score, psnr_score


# ----------
#  Training
# ----------

def train():
    prev_time = time.time()
    for epoch in range(model_config['epoch'], train_config['n_epochs']):
        for i, batch in enumerate(dataloader):
 
            # Model inputs
            real_A = batch["B"].to(device) # 真实的LDCT
            real_B = batch["A"].to(device) # 真实的SDCT

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *patch), dtype=torch.float32, requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), *patch), dtype=torch.float32, requires_grad=False).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_DBG.zero_grad()

            # GAN loss
            fake_B = db_generator(real_A)  # 
            pred_fake = db_discriminator(fake_B, real_A)
            # print(pred_fake.shape)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_DBDBG.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_DBD.zero_grad()

            # Real loss
            pred_real = db_discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = db_discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_DBD.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i

            batches_left = train_config['n_epochs'] * len(dataloader) - batches_done
            time_left = timedelta(seconds=batches_left * (time.time() - prev_time))  # 计算剩余时间
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    train_config['n_epochs'],
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % model_config['sample_interval'] == 0:
                outimages, ssim_score, psnr_score = sample_images(batches_done)

                if use_wandb:
                    wandb.log({
                        "Epoch": epoch,
                        "D loss": loss_D.item(),
                        "G loss": loss_G.item(),
                        "pixel loss": loss_pixel.item(),
                        "GAN loss": loss_GAN.item(),
                        "ssim_score": ssim_score,
                        "psnr_score": psnr_score,
                        "generated_images": [wandb.Image(outimages)]
                    })

        if model_config['checkpoint_interval'] != -1 and epoch % model_config['checkpoint_interval'] == 0:
            # Save model checkpoints
            torch.save(db_generator.state_dict(),
                       "mydualgan/saved_models/%s/db_generator_%d.pth" % (model_config['dataset_name'], epoch))
            torch.save(db_discriminator.state_dict(),
                       "mydualgan/saved_models/%s/db_discriminator_%d.pth" % (model_config['dataset_name'], epoch))


if __name__ == '__main__':
    train()
