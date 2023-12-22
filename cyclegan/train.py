import itertools
import datetime
import time
import sys
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from models import *
from datasets import *
from utils import config
from utils.CustomDataset import CDataset
from utils.cyclegan_utils import ReplayBuffer, LambdaLR
import torch

config_file = "config_default.yaml"
configs = config.update_project_dir(config_file)
model_config = configs['model']['cyclegan']
train_config = configs['train']

os.makedirs("outputs/%s" % model_config['dataset_name'],
            exist_ok=True)  # exist_ok = True: 如果目录存在，什么都不做，如果不存在，则创建该目录
os.makedirs("saved_models/%s" % model_config['dataset_name'], exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
input_shape = (model_config['channels'], model_config['img_size'], model_config['img_size'])

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, model_config['n_residual_blocks'])
G_BA = GeneratorResNet(input_shape, model_config['n_residual_blocks'])
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if model_config['epoch'] != 0:
    # Load pretrained models
    G_AB.load_state_dict(
        torch.load("saved_models/%s/G_AB_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))
    G_BA.load_state_dict(
        torch.load("saved_models/%s/G_BA_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))
    D_A.load_state_dict(
        torch.load("saved_models/%s/D_A_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))
    D_B.load_state_dict(
        torch.load("saved_models/%s/D_B_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=model_config['lr'],
    betas=(model_config['b1'], model_config['b2'])
)
# 两个生成器 G_AB 和 G_BA 的参数合并成一个参数迭代器。

optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=model_config['lr'],
                                 betas=(model_config['b1'], model_config['b2']))  # betas用于计算梯度和梯度平方的运行平均值，
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=model_config['lr'],
                                 betas=(model_config['b1'], model_config['b2']))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(train_config['n_epochs'], model_config['epoch'], model_config['decay_epoch']).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(train_config['n_epochs'], model_config['epoch'], model_config['decay_epoch']).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(train_config['n_epochs'], model_config['epoch'], model_config['decay_epoch']).step
)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    # 数据增强
    transforms.Resize(int(model_config['img_size'] * 1.12), Image.BICUBIC),
    transforms.RandomCrop((model_config['img_size'], model_config['img_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("cyclegan/dogsvscats/dogs", "cyclegan/dogsvscats/cats", transforms_=transforms_, unaligned=True),
    batch_size=train_config['batch_size'],
    shuffle=True,
    num_workers=model_config['n_cpu'],
)

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("cyclegan/dogsvscats/dogs", "cyclegan/dogsvscats/cats", transforms_=transforms_, unaligned=True,
                 mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=model_config['n_cpu'],
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = imgs["A"].to(device)
    fake_B = G_AB(real_A)
    real_B = imgs["B"].to(device)
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "outputs/%s/%s.png" % (model_config['dataset_name'], batches_done), normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(model_config['epoch'], train_config['n_epochs']):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)

        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), *D_A.output_shape), dtype=torch.float32, requires_grad=False).to(device)
        fake = torch.zeros((real_A.size(0), *D_A.output_shape), dtype=torch.float32, requires_grad=False).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + model_config['lambda_cyc'] * loss_cycle + model_config['lambda_id'] * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = train_config['n_epochs'] * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                train_config['n_epochs'],
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % model_config['sample_interval'] == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if model_config['checkpoint_interval'] != -1 and epoch % model_config['checkpoint_interval'] == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (model_config['dataset_name'], epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (model_config['dataset_name'], epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (model_config['dataset_name'], epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (model_config['dataset_name'], epoch))
