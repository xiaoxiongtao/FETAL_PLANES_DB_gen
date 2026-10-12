"""通用工具：随机种子、日志、检查点、EMA、学习率调度、图像保存。"""
from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    """固定随机种子以提高可复现性。

    注意：这里开启 cudnn.benchmark 以换取训练速度（自动选择最快卷积算法），
    因此卷积运算本身不做 bit 级确定性保证；数据打乱、权重初始化等仍受种子控制。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    """维护一个滑动平均值，用于统计 loss 等。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CSVLogger:
    """把训练过程逐行写入 CSV，供 Jupyter 绘图与离线分析。"""

    def __init__(self, path: str, columns: Iterable[str]):
        self.path = path
        self.columns = list(columns)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.columns)

    def log(self, **kwargs):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow({k: kwargs.get(k, "") for k in self.columns})

    def read(self) -> list:
        import pandas as pd

        return pd.read_csv(self.path).to_dict("records")


class LinearDecayLR:
    """线性衰减学习率调度器（CycleGAN 常用）。

    在 decay_start_epoch 之前保持初始学习率，之后线性衰减到 0。
    """

    def __init__(self, optimizer, epochs: int, decay_start_epoch: int):
        self.optimizer = optimizer
        self.initial_lrs = [g["lr"] for g in optimizer.param_groups]
        self.epochs = epochs
        self.decay_start = decay_start_epoch

    def step(self, epoch: int):
        if epoch >= self.decay_start:
            frac = 1.0 - (epoch - self.decay_start) / max(self.epochs - self.decay_start, 1)
            frac = max(frac, 0.0)
            for g, lr0 in zip(self.optimizer.param_groups, self.initial_lrs):
                g["lr"] = lr0 * frac

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class EMA:
    """指数滑动平均（Exponential Moving Average），用于 Diffusion 采样。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Any]):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, epoch: int = 0,
                    metrics: Optional[Dict[str, float]] = None, ema: Optional[EMA] = None,
                    **extra) -> None:
    """保存检查点（模型权重 + 优化器 + 指标）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler
    if metrics is not None:
        ckpt["metrics"] = metrics
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, device: str = "cpu",
                    strict: bool = True) -> Dict[str, Any]:
    """加载检查点，返回元信息。"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def save_image_grid(tensor: torch.Tensor, path: str, nrow: int = 8) -> None:
    """把 [N, C, H, W]（取值范围 [-1, 1]）保存为一张网格图。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def denorm(x: torch.Tensor) -> torch.Tensor:
    """把 [-1, 1] 反归一化到 [0, 1]。"""
    return (x + 1.0) / 2.0


def count_parameters(model: nn.Module) -> int:
    """返回可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_exp_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)
