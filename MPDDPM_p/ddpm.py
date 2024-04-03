from DCUnet import *
import torch.nn.functional as F
from tqdm import tqdm


class DCDDPM(nn.Module):
    def __init__(self, unet_model, beta_schedule="linear", T=1000):
        super(DCDDPM, self).__init__()
        self.seed = 1
        self.unet = unet_model
        self.T = T  # Number of diffusion steps
        self.beta_schedule = self._get_beta_schedule(beta_schedule, T)

    def _get_beta_schedule(self, schedule, T, beta_start=0.0001, beta_end=0.1):
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T).to(self.unet.device)
        elif schedule == "quadratic":
            # Quadratic growth from beta_start to beta_end
            return torch.linspace(0, 1, T) ** 2 * (beta_end - beta_start) + beta_start
        elif schedule == "exponential":
            # Exponential growth from beta_start to beta_end
            return (beta_end / beta_start) ** (torch.linspace(0, 1, T)) * beta_start
        else:
            raise ValueError("Unknown beta schedule")

    def control_seed(self):
        torch.manual_seed(self.seed)
        self.seed += 1

    def add_noise_sample(self, x_start, t, noise=None):
        """ Sample from q(x_t | x_0) """
        self.control_seed()
        if noise is None:
            noise = torch.randn_like(x_start)
        beta_t = self.beta_schedule[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return torch.sqrt(1 - beta_t) * x_start + torch.sqrt(beta_t) * noise, torch.sqrt(beta_t) * noise

    def p_sample(self, x_t, t):
        """ Sample from p(x_{t-1} | x_t) """
        if t == 0:
            return x_t
        else:
            beta_t = self.beta_schedule[t]
            pred_noise = self.unet(x_t)
            mean = (1 / torch.sqrt(1 - beta_t)) * (x_t - (beta_t / torch.sqrt(1 - beta_t)) * pred_noise)
            if t > 1:  # 如果不是最后一步，添加下一步的噪声
                noise = torch.randn_like(x_t)
                return mean + torch.sqrt(beta_t) * noise
            else:
                return mean

    def reverse_diffusion(self, x_start):
        with torch.no_grad():
            self.eval()
            x_t = x_start
            for t in tqdm(reversed(range(self.T)), desc="Reverse diffusion"):
                x_t = self.p_sample(x_t, t)
            self.train()
            return x_t

    def get_loss(self, x_t, scaled_noise, t):
        """ Get loss for training """
        self.control_seed()

        pred_noise = self.unet(x_t)  # self.unet(x_t, t)
        loss = F.mse_loss(pred_noise, scaled_noise, reduction="sum")
        # loss = torch.mean((x_recon - x_start) ** 2)
        return loss / x_t.size(0)

    def forward(self, x_start):
        self.control_seed()
        t = torch.randint(0, self.T, (x_start.size(0),), device=self.unet.device)
        x_t, scaled_noise = self.add_noise_sample(x_start, t)
        return self.get_loss(x_t, scaled_noise, t)
