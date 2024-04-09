import os, sys
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import ImageDatasetGPU1
from ddpm import DCDDPM
from mpunet import *
import time
import torch
from tqdm import tqdm
from sample import sample

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evalution import rmse, ssim, psnr

if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # 获取 GPU 数量
        print(f"CUDA 可用")
        device = torch.device(f"cuda:6")
    print("当前正在使用设备:", device)

    # 超参数设置
    learning_rate = 1e-4
    batch_size = 16
    num_epochs = 1000

    high_dir = "B301MM/high"
    low_dir = "B301MM/low"
    max_nums = 30
    aapm_data = ImageDatasetGPU1(
        os.path.join("/root/lmy/GanZoo/MPDDPM_p/B301MM"),
        device=device,
        # max_nums=max_nums,
    )
    ckpt_save_iter = 100
    train_size = int(0.9 * len(aapm_data))
    test_size = len(aapm_data) - train_size

    train_dataset, test_dataset = random_split(aapm_data, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        test_dataset,
        batch_size=5,
        shuffle=False,
    )

    # 模型初始化
    unet_model = MPUnet(in_channels=1, out_channels=1, device=device)
    model = DCDDPM(unet_model).to(device)
    print("模型初始化完成")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as t:
            for high_img, low_img in t:
                pred_x_0, pred_noise, scaled_noise = model(low_img)
                loss_noise = F.mse_loss(pred_noise, scaled_noise)
                loss_content = F.mse_loss(pred_x_0, high_img)
                loss = (loss_noise + loss_content) / high_img.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    psnr_score = psnr.psnr(pred_x_0, high_img)
                    ssim_score = ssim.ssim(pred_x_0, high_img)
                    # rmse_score = rmse(pred_x_0, high_img)
                    t.set_postfix(
                        loss=f"{loss.item():.4f}",
                        psnr=f"{psnr_score:.4f}",
                        ssim=f"{ssim_score:.4f}",
                    )

        if (epoch + 1) % ckpt_save_iter == 0 and epoch != 0:
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs} finished, Avg Time per epoch: {epoch_time:.2f}s"
            )
            sample(model)
            torch.save(model.state_dict(), "ddpm_model.pth")
