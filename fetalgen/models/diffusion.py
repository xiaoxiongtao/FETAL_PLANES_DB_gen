"""Diffusion 模型：DDPM + U-Net（含时间步与类别条件）。

参考论文:
    Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", 2021
    (cosine schedule)。
    Song et al., "Denoising Diffusion Implicit Models", 2021 (DDIM 加速采样)。

本项目用于胎儿超声图像的类别条件生成（6 个切面类别）。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- U-Net ---------------------------------- #

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int,
                         max_period: int = 10000) -> torch.Tensor:
    """正弦位置编码，用于把时间步 t 映射为特征向量。"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, dtype=torch.float32) / half)
    freqs = freqs.to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # dim 为奇数时补零
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AttentionBlock(nn.Module):
    """多头自注意力（单头实现已足够，通道维度上做 qkv）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H * W).unbind(1)  # 各 [B, C, HW]
        scale = C ** -0.5
        attn = torch.bmm(q.transpose(1, 2) * scale, k)          # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)  # [B, C, HW]
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class ResBlock(nn.Module):
    """带时间/类别嵌入的残差块（嵌入以逐通道加法注入）。"""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb(emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """DDPM 使用的 U-Net，带时间步嵌入与可选类别嵌入。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        time_emb_dim = base_channels * 4
        emb_dim = time_emb_dim * 4  # 时间 MLP 与类别嵌入的输出维度

        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes, emb_dim) if num_classes is not None else None

        # 各层通道数
        num_levels = len(channel_mults)
        chs = [base_channels * m for m in channel_mults]
        self.attention_resolutions = set(attention_resolutions)

        self.init_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        # 下采样
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        for level in range(num_levels):
            in_ch = chs[level - 1] if level > 0 else chs[0]
            out_ch = chs[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, emb_dim, dropout))
                in_ch = out_ch
            self.down_blocks.append(blocks)
            self.down_attns.append(AttentionBlock(out_ch))
            if level < num_levels - 1:
                self.down_samples.append(Downsample(out_ch))

        # 中间层（最底层）
        self.mid_block1 = ResBlock(chs[-1], chs[-1], emb_dim, dropout)
        self.mid_attn = AttentionBlock(chs[-1])
        self.mid_block2 = ResBlock(chs[-1], chs[-1], emb_dim, dropout)

        # 上采样
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        prev_ch = chs[-1]  # 进入当前层的通道数（来自更深层 / 中间层）
        for level in reversed(range(num_levels)):
            out_ch = chs[level]
            in_ch = prev_ch + out_ch  # 拼接跳跃连接
            blocks = nn.ModuleList()
            cur = in_ch
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(cur, out_ch, emb_dim, dropout))
                cur = out_ch
            self.up_blocks.append(blocks)
            self.up_attns.append(AttentionBlock(out_ch))
            prev_ch = out_ch
            if level > 0:
                self.up_samples.append(Upsample(out_ch))

        self.out_norm = nn.GroupNorm(32, chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                c: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 时间嵌入 + 类别嵌入
        emb = sinusoidal_embedding(t, self.time_emb_dim).to(x.device)
        emb = self.time_mlp(emb)
        if self.class_emb is not None:
            if c is None:
                raise ValueError("类别条件模型需要传入类别索引 c")
            emb = emb + self.class_emb(c)

        skips = []
        h = self.init_conv(x)
        skips.append(h)

        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, emb)
            if h.shape[-1] in self.attention_resolutions:
                h = self.down_attns[level](h)
            skips.append(h)
            if level < len(self.down_samples):
                h = self.down_samples[level](h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for level, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, emb)
            if h.shape[-1] in self.attention_resolutions:
                h = self.up_attns[level](h)
            if level < len(self.up_samples):
                h = self.up_samples[level](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------- Diffusion --------------------------------- #

def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    """构造噪声方差表 beta_t。

    - linear: 线性从 beta_start 到 beta_end。
    - cosine: Nichol & Dhariwal 的余弦调度。
    """
    if schedule == "linear":
        beta_start, beta_end = 1e-4, 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, min=0.0, max=0.999)
    else:
        raise ValueError(f"未知 beta_schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    """DDPM 扩散过程，封装前向加噪、训练损失与采样。"""

    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_schedule: str = "cosine"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向加噪：q(x_t | x_0)。"""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * x0 + b * noise

    def training_loss(self, x0: torch.Tensor,
                      c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """简化目标：预测噪声的 MSE。"""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t, c)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor,
                 c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DDPM 单步去噪采样。"""
        pred_noise = self.model(x, t, c)
        alpha = self.alphas[t][:, None, None, None]
        alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]

        mean = (x - beta / torch.sqrt(1.0 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha)
        noise = torch.randn_like(x)
        noise = torch.where((t > 0).view(-1, 1, 1, 1), noise, torch.zeros_like(noise))
        return mean + torch.sqrt(beta) * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], c: Optional[torch.Tensor] = None,
               device: str = "cuda", progress: bool = False) -> torch.Tensor:
        """完整 DDPM 采样（timesteps 步）。"""
        b = shape[0]
        x = torch.randn(shape, device=device)
        iterator = reversed(range(self.timesteps))
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="DDPM 采样")
        for t in iterator:
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, c)
        return x

    @torch.no_grad()
    def ddim_sample(self, shape: Tuple[int, ...], c: Optional[torch.Tensor] = None,
                    steps: int = 100, eta: float = 0.0, device: str = "cuda",
                    progress: bool = False) -> torch.Tensor:
        """DDIM 加速采样（eta=0 时完全确定）。"""
        b = shape[0]
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)

        iterator = range(steps)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="DDIM 采样")

        for i in iterator:
            t = times[i]
            t_prev = times[i + 1] if i < steps - 1 else torch.tensor(-1, device=device)
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)

            pred_noise = self.model(x, t_tensor, c)
            a_cum = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # 预测 x0
            pred_x0 = (x - torch.sqrt(1.0 - a_cum) * pred_noise) / torch.sqrt(a_cum)
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_cum) *
                                     (1.0 - a_cum / a_prev))
            noise = torch.randn_like(x)
            direction = torch.sqrt(1.0 - a_prev - sigma ** 2) * pred_noise
            x = torch.sqrt(a_prev) * pred_x0 + direction + sigma * noise
        return x
