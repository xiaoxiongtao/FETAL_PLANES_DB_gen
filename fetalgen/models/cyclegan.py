"""CycleGAN 模型：ResNet 生成器 + PatchGAN 判别器。

参考论文:
    Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent
    Adversarial Networks", ICCV 2017.

本项目用于两个胎儿超声切面之间的无配对翻译，例如
"Fetal brain" <-> "Fetal femur"。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def init_weights(m: nn.Module):
    """CycleGAN 标准初始化：Conv 用 N(0, 0.02)，BatchNorm 用 N(1, 0.02)。"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    """InstanceNorm + ReLU 残差块（ReflectionPad 保持分辨率）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet-9 生成器（Johnson et al.）。

    结构: c7s1-64 -> d128 -> d256 -> 9x ResBlock -> u128 -> u64 -> c7s1-1 -> Tanh
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 1,
                 ngf: int = 64, n_resblocks: int = 9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_resblocks):
            layers.append(ResidualBlock(ngf * 4))
        layers += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN 判别器。

    输出为一张 patch 置信图，而非单个标量。
    """

    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev, nf = nf, min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    """对抗损失。

    - lsgan: MSE 对齐到 1（真）/ 0（假），训练更稳定（本项目默认）。
    - vanilla: BCEWithLogitsLoss。
    """

    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        assert mode in ("lsgan", "vanilla"), f"未知 gan_mode: {mode}"
        self.mode = mode
        self.loss = nn.MSELoss() if mode == "lsgan" else nn.BCEWithLogitsLoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)
