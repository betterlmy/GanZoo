from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DCDDPM(nn.Module):
    def __init__(self, unet_model, beta_schedule="dns", T=50):
        super(DCDDPM, self).__init__()
        self.seed = 1
        self.unet = unet_model
        self.device = self.unet.device
        self.T = T  # Number of diffusion steps
        self.beta_schedule = self._get_beta_schedule(beta_schedule, T).to(self.device)

    def _get_beta_schedule(self, schedule, T, beta_start=0.0001, beta_end=0.1):
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T)
        elif schedule == "quadratic":
            # Quadratic growth from beta_start to beta_end
            return torch.linspace(0, 1, T) ** 2 * (beta_end - beta_start) + beta_start
        elif schedule == "exponential":
            # Exponential growth from beta_start to beta_end
            return (beta_end / beta_start) ** (torch.linspace(0, 1, T)) * beta_start
        elif schedule == "dns":
            # Exponential growth from beta_start to beta_end
            return self.generate_DNS_schedule(T, beta_start, beta_end, 5)
        else:
            raise ValueError("Unknown beta schedule")

    @abstractmethod
    def generate_DNS_schedule(self, T, low, high, off=5):
        # Dynamic negative square.
        print("generate_DNS_schedule, T =", off)
        t = [value for value in range(0, T)]
        wt = []
        for i in t:
            w = -(((i + off) / (T + 1 + off)) ** 2) + 1  # T+1防止最后失效
            wt.append(w)
        wt = torch.tensor(wt)
        assert (
            high > low and low >= 0 and high <= 1
        ), "high > low and low >= 0 and high <= 1"
        betas = (1 - wt) * (high - low) + low
        return betas

    def control_seed(self):
        torch.manual_seed(self.seed)
        self.seed = self.seed + 1

    def add_noise_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        self.control_seed()
        if noise is None:
            noise = torch.randn_like(x_start)  # 默认噪声是正态分布
        beta_t = self.beta_schedule[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return (
            torch.sqrt(1 - beta_t) * x_start + torch.sqrt(beta_t) * noise,
            torch.sqrt(beta_t) * noise,
        )

    def p_sample(self, x_t, t):
        """Sample from p(x_{t-1} | x_t)"""
        if t == 0:
            return x_t
        else:
            beta_t = self.beta_schedule[t]
            pred_noise = self.unet(x_t)
            print("pred_noise.shape", pred_noise.shape)
            mean = (1 / torch.sqrt(1 - beta_t)) * (
                x_t - (beta_t / torch.sqrt(1 - beta_t)) * pred_noise
            )
            if t > 1:  # 如果不是最后一步，添加下一步的噪声
                noise = torch.randn_like(x_t)
                return mean + torch.sqrt(beta_t) * noise
            else:
                return mean

    def p_sample_batch(self, x_t, t):
        """Sample from p(x_{t-1} | x_t)"""
        if torch.any(t == 0):
            return x_t  # 如果时间步t为0，则直接返回x_t

        beta_t = self.beta_schedule[t].view(
            -1, 1, 1, 1
        )  # beta_schedule需要能处理tensor索引
        pred_noise = self.unet(x_t)  # 确保unet可以处理批量数据

        one_minus_beta_t = 1 - beta_t
        sqrt_one_minus_beta_t = torch.sqrt(one_minus_beta_t)
        mean = (
            x_t - beta_t / sqrt_one_minus_beta_t * pred_noise
        ) / sqrt_one_minus_beta_t

        # 添加噪声
        noise = torch.randn_like(x_t)
        sqrt_beta_t = torch.sqrt(beta_t)
        noisy_mean = mean + sqrt_beta_t * noise
        mask = t > 1  # 创建一个布尔掩码，表示哪些元素的时间步大于1
        mean[mask] = noisy_mean[mask]
        return mean

    def reverse_diffusion(self, x_start, t):
        x_t = x_start
        if t == None:
            t = torch.full(
                (x_start.size(0),),
                self.T,
                dtype=torch.long,
                device=self.device,
            )  # 为每个样本创建一个全是self.T的Tensor
        for current_t in reversed(range(1, self.T + 1)):
            mask = t >= current_t  # 找出所有当前需要处理的样本
            if mask.any():
                x_t[mask] = self.p_sample_batch(
                    x_t[mask], torch.full_like(t[mask], current_t, device=self.device)
                )
        return x_t

    def forward(self, x_start, t=None):
        self.control_seed()
        if t is None:
            t = torch.randint(0, self.T, (x_start.size(0),), device=self.device)

        x_t, scaled_noise = self.add_noise_sample(
            x_start, t
        )  # 得到前向加噪的图像和噪声
        pred_noise = self.unet(x_t)  # self.unet(x_t, t)
        # pred_x_0 = self.reverse_diffusion(x_t, t)  # 推理的高清图像
        return x_t, t, pred_noise, scaled_noise

    def pred(self, x_t, t=None):
        pred_x_0 = self.reverse_diffusion(x_t, t)
        return pred_x_0


if __name__ == "__main__":
    from mpunet import MPUnet

    unet_model = MPUnet(in_channels=1, out_channels=1, device="cpu")
    model = DCDDPM(unet_model)
    x = torch.randn(1, 1, 224, 224).to("cpu")
    x2 = torch.randn(1, 1, 224, 224).to("cpu")
    print(model(x, x2))
