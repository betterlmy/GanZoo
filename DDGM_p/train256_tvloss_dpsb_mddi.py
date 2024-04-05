from datetime import datetime, timedelta
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from DDGM_p.datasets import ImageDatasetGPU1
from DDGM_p.models import GeneratorUNet, Discriminator, weights_init_normal, DPSB
from evalution import ssim, psnr, rmse
from utils import config
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import wandb

"""！！！训练时 将workdir设置为GanZoo的根目录 而非ddgm的根目录"""
config_file = "config_ddgm_256.yaml"
configs = config.update_project_dir(config_file)
model_config = configs["model"]["ddgm"]
train_config = configs["train_dpsb_mddi"]
use_wandb = train_config["use_wandb"]
formatted_date = datetime.now().strftime("%m-%d-%H-%M")
batch_size = train_config["batch_size"]

os.makedirs("ddgm/outputs/256", exist_ok=True)
os.makedirs("ddgm/saved_models/256", exist_ok=True)


def tv_loss(img):
    # 计算图像在水平方向上的变化
    w_var = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    # 计算图像在垂直方向上的变化
    h_var = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    # 将变化量的和作为总变异
    loss = torch.sum(w_var) + torch.sum(h_var)
    batch_size, channel, height, width = img.shape
    return loss / (batch_size * channel * height * width)


# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 20

# Calculate output of image discriminator (PatchGAN)
# patch = (1, train_config["img_size"] // 2**4, train_config["img_size"] // 2**4)
patch = (1, 16, 16)
db_generator = GeneratorUNet(model_config["channels"], model_config["channels"])
db_discriminator = DPSB(model_config["channels"])

b_generator = GeneratorUNet(model_config["channels"], model_config["channels"])
b_discriminator = DPSB(model_config["channels"])

cuda = torch.cuda.is_available()
device = torch.device("cuda:" + train_config["gpu_id"] if cuda else "cpu")
# device = torch.device("mps")
if device.type != "cpu":
    db_generator = db_generator.to(device)
    db_discriminator = db_discriminator.to(device)

    b_generator = b_generator.to(device)
    b_discriminator = b_discriminator.to(device)
    criterion_GAN.to(device)
    criterion_pixelwise.to(device)
    print("current device:" + str(device))
else:
    # Initialize weights
    db_generator.apply(weights_init_normal)
    db_discriminator.apply(weights_init_normal)
    b_generator.apply(weights_init_normal)
    b_discriminator.apply(weights_init_normal)

# Optimizers
optimizer_DBG = torch.optim.Adam(
    db_generator.parameters(),
    lr=model_config["lr"],
    betas=(model_config["b1"], model_config["b2"]),
)
optimizer_DBD = torch.optim.Adam(
    db_discriminator.parameters(),
    lr=model_config["lr"],
    betas=(model_config["b1"], model_config["b2"]),
)
optimizer_BG = torch.optim.Adam(
    b_generator.parameters(),
    lr=model_config["lr"],
    betas=(model_config["b1"], model_config["b2"]),
)
optimizer_BD = torch.optim.Adam(
    b_discriminator.parameters(),
    lr=model_config["lr"],
    betas=(model_config["b1"], model_config["b2"]),
)

transforms_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # 单通道
]

max_nums = train_config["max_nums"]

aapm_data = ImageDatasetGPU1(
    os.path.join(os.path.dirname(configs["project_dir"]), "aapm256"),
    transforms_=transforms_,
    device=device,
    max_nums=max_nums,
)

train_size = int(0.9 * len(aapm_data))
test_size = len(aapm_data) - train_size

train_dataset, test_dataset = random_split(aapm_data, [train_size, test_size])

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataloader = DataLoader(
    test_dataset,
    batch_size=5,
    shuffle=False,
)


# ----------
#  Training
# ----------


def train():
    prev_time = time.time()
    if use_wandb:
        wandb.init(
            project="gans",
            name="ddgm-256-TVloss-dpsb-" + formatted_date,
            config=configs,
        )
    for epoch in range(model_config["epoch"], train_config["n_epochs"]):
        for i, batch in enumerate(dataloader):
            trueSDCT = batch["TrueSDCT"].to(device)  # 真实的SDCT
            trueLDCT = batch["TrueLDCT"].to(device)  # 真实的LDCT
            fakeLDCT = batch["FakeLDCT"].to(device)  # 加噪0.9SDCT得到的假LDCT
            fakeULDCT = batch["FakeULDCT"].to(device)  # 加噪0.8SDCT得到的假ULDCT

            # Adversarial ground truths
            valid = torch.ones(
                (trueLDCT.size(0), *patch), dtype=torch.float32, requires_grad=False
            ).to(device)
            fake = torch.zeros(
                (trueLDCT.size(0), *patch), dtype=torch.float32, requires_grad=False
            ).to(device)
            valid_global = torch.ones(
                (trueLDCT.size(0), 1), dtype=torch.float32, requires_grad=False
            ).to(device)
            fake_global = torch.zeros(
                (trueLDCT.size(0), 1), dtype=torch.float32, requires_grad=False
            ).to(device)

            # BGAN的训练使用到模拟的LDCT和真实的LDCT  学习模拟的LDCT到真实LDCT的映射关系 因此
            b_generator.train()
            b_discriminator.eval()
            db_generator.eval()
            db_discriminator.eval()
            optimizer_BG.zero_grad()
            g_LDCT = b_generator(fakeLDCT)  # 给出加噪得到的LDCT图像 用来模拟真实的LDCT
            pred_fake1, pred_fake1_global = b_discriminator(g_LDCT, fakeLDCT)
            loss_GAN1 = criterion_GAN(pred_fake1, valid)
            loss_GAN1_global = criterion_GAN(pred_fake1_global, valid_global)
            # Pixel-wise loss
            loss_B_pixel = criterion_pixelwise(g_LDCT, trueLDCT)
            loss_tv1 = tv_loss(g_LDCT)
            loss_BG = (
                loss_GAN1_global
                + loss_GAN1
                + lambda_pixel * loss_B_pixel
                + lambda_pixel * loss_tv1
            )
            # Total loss

            loss_BG.backward()
            optimizer_BG.step()

            # 训练BGAN_D
            b_generator.eval()
            b_discriminator.train()
            db_generator.eval()
            db_discriminator.eval()
            optimizer_BD.zero_grad()
            # Real loss
            pred_real2, pred_real2_global = b_discriminator(fakeLDCT, trueLDCT)
            loss_real2 = criterion_GAN(pred_real2, valid)
            loss_real2_global = criterion_GAN(pred_real2_global, valid_global)
            # Fake loss
            pred_fake2, pred_fake2_global = b_discriminator(g_LDCT.detach(), fakeLDCT)
            loss_fake2 = criterion_GAN(pred_fake2, fake)
            loss_fake2_global = criterion_GAN(pred_fake2_global, fake_global)

            # Total loss
            loss_BD = 0.25 * (
                loss_real2 + loss_fake2 + loss_real2_global + loss_fake2_global
            )

            loss_BD.backward()
            optimizer_BD.step()

            # DBGAN的训练使用到了训练好的BGAN_G来生成模拟的ULDCT和真实的SDCT数据 之间的映射  学习模拟的ULDCT到真实SDCT的映射关系 因此
            b_generator.eval()
            b_discriminator.eval()
            db_generator.train()
            db_discriminator.eval()

            ULDCT = b_generator(
                fakeULDCT
            ).detach()  # 生成虚假的ULDCT 也就是模拟出来的ULDCT

            optimizer_DBG.zero_grad()
            g_SDCT = db_generator(ULDCT)  # g_SDCT为去噪后的清晰CT
            pred_fake3, pred_fake3_global = db_discriminator(g_SDCT, ULDCT)
            loss_GAN3 = criterion_GAN(pred_fake3, valid)
            loss_GAN3_global = criterion_GAN(pred_fake3_global, valid_global)

            # Pixel-wise loss
            loss_DB_pixel = criterion_pixelwise(g_SDCT, trueSDCT)
            loss_tv2 = tv_loss(g_SDCT)

            # Total loss
            loss_DBG = (
                loss_GAN3_global
                + loss_GAN3
                + lambda_pixel * loss_DB_pixel
                + lambda_pixel * loss_tv2
            )

            loss_DBG.backward()
            optimizer_DBG.step()

            b_generator.eval()
            b_discriminator.eval()
            db_generator.eval()
            db_discriminator.train()
            # 训练DBGAN_D
            optimizer_DBD.zero_grad()
            # Real loss
            pred_real4, pred_real4_global = db_discriminator(trueSDCT, ULDCT)
            loss_real4 = criterion_GAN(pred_real4, valid)
            loss_real4_global = criterion_GAN(pred_real4_global, valid_global)

            # Fake loss
            pred_fake4, pred_fake4_global = db_discriminator(g_SDCT.detach(), ULDCT)
            loss_fake4 = criterion_GAN(pred_fake4, fake)
            pred_fake4_global = criterion_GAN(pred_fake4_global, fake_global)
            # Total loss
            loss_DBD = 0.25 * (
                loss_real4 + loss_fake4 + pred_fake4_global + loss_real4_global
            )

            loss_DBD.backward()
            optimizer_DBD.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i

            batches_left = train_config["n_epochs"] * len(dataloader) - batches_done
            time_left = timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )  # 计算剩余时间
            prev_time = time.time()
            # Print log
            if batches_done % train_config["print_interval"] == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [SSIM:%f][PSNR:%f][BD loss: %f] [BG loss: %f], [Bpixel: %f][DBD loss: %f] [DBG loss: %f], [DBpixel: %f] ETA: %s\n"
                    % (
                        epoch,
                        train_config["n_epochs"],
                        i,
                        len(dataloader),
                        ssim.ssim(g_SDCT, trueSDCT),
                        psnr.psnr(g_SDCT, trueSDCT),
                        loss_BD.item(),
                        loss_BG.item(),
                        loss_B_pixel.item(),
                        loss_DBD.item(),
                        loss_DBG.item(),
                        loss_DB_pixel.item(),
                        time_left,
                    )
                )

            # If at sample interval save image
            if batches_done % train_config["sample_interval"] == 0:
                outimages, ssim_score, psnr_score, rmse_score = sample_images(
                    batches_done
                )

                print(
                    "eval:%.4f,%.4f,%.4f"
                    % (ssim_score.item(), psnr_score.item(), rmse_score.item())
                )
                if use_wandb:
                    wandb.log(
                        {
                            "Epoch": epoch,
                            "BGAN_D loss": loss_BD.item(),
                            "BGAN_G loss": loss_BG.item(),
                            "BGAN pixel loss": loss_B_pixel.item(),
                            "DBGAN_D loss": loss_BD.item(),
                            "DBGAN_G loss": loss_BG.item(),
                            "DBGAN pixel loss": loss_B_pixel.item(),
                            "ssim_score": ssim_score,
                            "psnr_score": psnr_score,
                            "rmse_score": rmse_score,
                            "generated_images": [
                                wandb.Image(
                                    outimages,
                                    caption="加噪模拟的ULDCT, 使用BGAN生成的ULDCT, 对BGAN生成的ULDCT进行重建的SDCT, 真实的SDCT, 真实的LDCT, 对SDCT加噪得到的LDCT",
                                )
                            ],
                        }
                    )

        if (
            train_config["checkpoint_interval"] != -1
            and epoch % train_config["checkpoint_interval"] == 0
        ):
            print("save model %d" % epoch)
            # Save model checkpoints
            torch.save(
                db_generator.state_dict(),
                "ddgm/saved_models/256/db_generator_tv+dp+md_%d.pth" % epoch,
            )
            torch.save(
                db_discriminator.state_dict(),
                "ddgm/saved_models/256/db_discriminator_tv+dp+md_%d.pth" % epoch,
            )
            torch.save(
                b_generator.state_dict(),
                "ddgm/saved_models/256/b_generator_tv+dp+md_%d.pth" % epoch,
            )
            torch.save(
                b_discriminator.state_dict(),
                "ddgm/saved_models/256/b_discriminator_tv+dp+md_%d.pth" % epoch,
            )


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    train_imgs = next(iter(val_dataloader))
    trueSDCT = train_imgs["TrueSDCT"].to(device)  # 真实的SDCT
    fakeULDCT = train_imgs["FakeULDCT"].to(device)  # 加噪0.8SDCT得到的假ULDCT
    trueLDCT = train_imgs["TrueLDCT"].to(device)  # 真实的LDCT
    fakeLDCT = train_imgs["FakeLDCT"].to(device)  # 加噪0.9SDCT得到的假LDCT
    ULDCT = b_generator(fakeULDCT)
    g_SDCT = db_generator(ULDCT)
    rmse_score = rmse.rmse(g_SDCT, trueSDCT)
    trueSDCT = make_grid(trueSDCT, nrow=5, normalize=True)
    fakeULDCT = make_grid(fakeULDCT, nrow=5, normalize=True)
    ULDCT = make_grid(ULDCT, nrow=5, normalize=True)
    g_SDCT = make_grid(g_SDCT, nrow=5, normalize=True)
    trueLDCT = make_grid(trueLDCT, nrow=5, normalize=True)
    fakeLDCT = make_grid(fakeLDCT, nrow=5, normalize=True)

    train_image_grid = torch.cat(
        (fakeULDCT, ULDCT, g_SDCT, trueSDCT, trueLDCT, fakeLDCT), 1
    )

    save_image(
        train_image_grid,
        "ddgm/outputs/256/tv+dp+md_%s.png" % batches_done,
        nrow=5,
        normalize=True,
    )
    ssim_score = ssim.ssim(g_SDCT, trueSDCT)
    psnr_score = psnr.psnr(g_SDCT, trueSDCT)
    return train_image_grid, ssim_score, psnr_score, rmse_score


if __name__ == "__main__":
    train()
