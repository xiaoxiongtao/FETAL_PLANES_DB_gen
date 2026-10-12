"""生成模型评估指标：FID、SSIM、PSNR、LPIPS。

- FID: 无配对分布距离，是生成质量的主指标，对 CycleGAN 与 Diffusion 通用。
- SSIM / PSNR / LPIPS: 逐像素/感知质量指标，需要图像一一对应，
  在 CycleGAN 中用于衡量循环一致性（输入 vs 重建）。
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# InceptionV3 输入的 ImageNet 归一化参数
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _InceptionV3Features(nn.Module):
    """预训练 InceptionV3，去掉分类头，输出 2048 维特征用于 FID。"""

    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        out = self.model(x)
        # eval 模式下 aux_logits 为 None，返回主输出；这里做防御性处理
        if isinstance(out, tuple):
            out = out[0]
        return out


def to_inception_input(x: torch.Tensor) -> torch.Tensor:
    """把 [N, 1, H, W] 且取值 [-1, 1] 的灰度图转成 InceptionV3 的输入。"""
    x = (x + 1.0) / 2.0                      # [-1,1] -> [0,1]
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)             # 灰度转三通道
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = _IMAGENET_MEAN.to(x.device)
    std = _IMAGENET_STD.to(x.device)
    return (x - mean) / std


@torch.no_grad()
def get_inception_features(images: torch.Tensor, model: _InceptionV3Features,
                           batch_size: int = 64, device: str = "cuda") -> np.ndarray:
    """提取一组图像的 InceptionV3 特征，返回 [N, 2048] numpy 数组。"""
    feats = []
    model.to(device)
    for i in range(0, len(images), batch_size):
        batch = to_inception_input(images[i:i + batch_size].to(device))
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    try:
        from scipy.linalg import sqrtm
    except ImportError as e:
        raise ImportError("FID 计算需要 scipy，请执行 `pip install scipy`") from e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


class FIDComputer:
    """封装 InceptionV3 与 FID 计算，避免重复加载模型。"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = _InceptionV3Features().to(device).eval()

    @torch.no_grad()
    def compute(self, real_images: torch.Tensor, fake_images: torch.Tensor,
                batch_size: int = 64) -> float:
        act_real = get_inception_features(real_images, self.model, batch_size, self.device)
        act_fake = get_inception_features(fake_images, self.model, batch_size, self.device)
        mu1, mu2 = act_real.mean(0), act_fake.mean(0)
        # 加 eps*I 正则化，避免样本数少于特征维度(2048)时协方差奇异
        d = act_real.shape[1]
        sigma1 = np.cov(act_real, rowvar=False) + 1e-6 * np.eye(d)
        sigma2 = np.cov(act_fake, rowvar=False) + 1e-6 * np.eye(d)
        return _frechet_distance(mu1, sigma1, mu2, sigma2)


# ------------------------- SSIM / PSNR / LPIPS ------------------------- #

def _gaussian_kernel(size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.expand(channels, 1, size, size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
         val_range: float = 2.0) -> float:
    """结构相似度（SSIM），输入为 [N, C, H, W]、取值范围 [-1, 1]。"""
    assert img1.shape == img2.shape
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    channels = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, channels).to(img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 2.0) -> float:
    """峰值信噪比（PSNR），输入取值 [-1, 1]，故 max_val = 2。"""
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return float("inf")
    return float(10.0 * torch.log10(max_val ** 2 / mse).item())


_LPIPS_MODEL = None


def lpips(img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
    """感知相似度（LPIPS），依赖可选包 lpips；未安装时返回 None。"""
    global _LPIPS_MODEL
    try:
        import lpips as _lpips
    except ImportError:
        return None

    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = _lpips.LPIPS(net="alex").to(img1.device).eval()
    # LPIPS 期望 [0, 1] 的 RGB 图
    a = (img1 + 1) / 2
    b = (img2 + 1) / 2
    if a.shape[1] == 1:
        a = a.repeat(1, 3, 1, 1)
        b = b.repeat(1, 3, 1, 1)
    with torch.no_grad():
        return float(_LPIPS_MODEL(a, b).mean().item())


def compute_pairwise_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Optional[float]]:
    """计算两批一一对应图像之间的 SSIM / PSNR / LPIPS。"""
    return {
        "ssim": ssim(a, b),
        "psnr": psnr(a, b),
        "lpips": lpips(a, b),
    }
