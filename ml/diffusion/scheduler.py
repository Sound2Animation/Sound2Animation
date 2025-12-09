"""DDPM Scheduler for diffusion"""
import torch
import math


class DDPMScheduler:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For p(x_{t-1} | x_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add noise to x_0 at timestep t: q(x_t | x_0)"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def step(self, pred_x0: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        """Single denoising step using predicted x0 directly (MDM style)"""
        coef1 = self.alphas_cumprod_prev[t].sqrt() * self.betas[t] / (1 - self.alphas_cumprod[t])
        coef2 = self.alphas[t].sqrt() * (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])
        posterior_mean = coef1 * pred_x0 + coef2 * x_t

        if t > 0:
            noise = torch.randn_like(x_t)
            return posterior_mean + self.posterior_variance[t].sqrt() * noise
        return posterior_mean
