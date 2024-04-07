from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DCDDIM(nn.Module):
    def __init__(self, unet_model, beta_schedule="dns", T=1000):
        super(DCDDIM, self).__init__()
        self.seed = 1
        self.unet = unet_model
        self.T = T  # Number of diffusion steps
        self.beta_schedule = self._get_beta_schedule(beta_schedule, T).to(self.unet.device)
        self.alpha_ts = self._compute_alpha_ts(T)
        print("beta_schedule:", self.beta_schedule.size())
        print("alpha_ts:", self.alpha_ts.size())
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

    def _compute_alpha_ts(self, T):
        """Precompute alpha values for all time steps."""
        alphas = torch.cumprod(1 - self.beta_schedule, dim=0)
        alphas = alphas.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Add necessary dims for image processing
        return alphas

    @staticmethod
    def generate_DNS_schedule(T, low, high, off=5):
        # Dynamic negative square.
        print("generate_DNS_schedule, T =", off)
        t = [value for value in range(0, T)]
        wt = []
        for i in t:
            w = -((i + off) / (T + 1 + off)) ** 2 + 1  # T+1防止最后失效
            wt.append(w)
        wt = torch.tensor(wt)
        assert (high > low and low >= 0 and high <= 1), "high > low and low >= 0 and high <= 1"
        betas = (1 - wt) * (high - low) + low
        return betas

    def control_seed(self):
        torch.manual_seed(self.seed)
        self.seed += 1

    def add_noise_sample(self, x_start, t, noise=None):
        """Modify to compute noise based on both the current and initial state."""
        if noise is None:
            noise = torch.randn_like(x_start)  # Default noise remains normal distribution
        print("t:", t)
        alpha_t = torch.cumprod(1 - self.beta_schedule[:t + 1], dim=0)[-1]

        alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return torch.sqrt(alpha_t) * x_start + torch.sqrt(1 - alpha_t) * noise, noise

    def p_sample(self, x_t, t):
        """Sample from p(x_{t-1} | x_t)"""
        if t == 0:
            return x_t
        else:
            beta_t = self.beta_schedule[t]
            pred_noise = self.unet(x_t)
            mean = (1 / torch.sqrt(1 - beta_t)) * (
                    x_t - (beta_t / torch.sqrt(1 - beta_t)) * pred_noise
            )
            if t > 1:  # 如果不是最后一步，添加下一步的噪声
                noise = torch.randn_like(x_t)
                return mean + torch.sqrt(beta_t) * noise
            else:
                return mean

    def reverse_diffusion(self, x_start, x_original, num_steps=100):
        """Incorporate x_original into the reverse diffusion process without using beta_t."""
        with torch.no_grad():
            self.eval()
            x_t = x_start
            for i in reversed(range(num_steps)):
                t = torch.full((x_start.size(0),), i, device=self.unet.device, dtype=torch.long)
                alpha_t = torch.cumprod(1 - self.beta_schedule[:t + 1], dim=0)[-1]
                alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                pred_noise = self.unet(x_t)
                # Calculate mu_t using alpha_t only
                mu_t = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                if i > 0:
                    x_t = mu_t + torch.sqrt(alpha_t * (1 - alpha_t)) * torch.randn_like(x_t)
                else:
                    x_t = mu_t
            self.train()
            return x_t

    def forward(self, x_start, x_high):
        """Assume initial noisy state is generated in the forward process."""
        noise = torch.randn_like(x_start)
        x_noisy, _ = self.add_noise_sample(x_start, self.T, noise)
        x_reconstructed = self.reverse_diffusion(x_noisy, x_start, num_steps=50)
        loss = F.mse_loss(x_reconstructed, x_high, reduction='mean')

        return x_reconstructed, loss


if __name__ == "__main__":
    from mpunet import MPUnet

    unet_model = MPUnet(in_channels=1, out_channels=1, device="mps")
    model = DCDDIM(unet_model)
    # x = torch.randn(1, 1, 224, 224).to("mps")
    # x2 = torch.randn(1, 1, 224, 224).to("mps")
    # print(model(x, x2))
