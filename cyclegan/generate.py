from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from models import GeneratorResNet, Discriminator
from datasets import ImageDataset
from utils import config
import torch

config_file = "config_default.yaml"
configs = config.update_project_dir(config_file)
model_config = configs['model']['cyclegan']
train_config = configs['train']
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
input_shape = (3, 256, 256)

G_AB = GeneratorResNet(input_shape, model_config['n_residual_blocks'])
G_BA = GeneratorResNet(input_shape, model_config['n_residual_blocks'])
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)
    print("current device:" + str(device))

G_AB.load_state_dict(
    torch.load("cyclegan/saved_models/%s/G_AB_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))
G_BA.load_state_dict(
    torch.load("cyclegan/saved_models/%s/G_BA_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))
D_A.load_state_dict(
    torch.load("cyclegan/saved_models/%s/D_A_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))
D_B.load_state_dict(
    torch.load("cyclegan/saved_models/%s/D_B_%d.pth" % (model_config['dataset_name'], model_config['epoch'])))

val_data = ImageDataset("cyclegan/dogsvscats/dogs", "cyclegan/dogsvscats/cats", transforms_=transforms_,
                        unaligned=True, mode="test")
val_dataloader = DataLoader(
    val_data,
    batch_size=5,
    shuffle=True,
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
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "cyclegan/outputs/%s/%s.png" % (model_config['dataset_name'], batches_done), normalize=False)
    return image_grid


def gene():
    sample_images(0)


if __name__ == '__main__':
    gene()
