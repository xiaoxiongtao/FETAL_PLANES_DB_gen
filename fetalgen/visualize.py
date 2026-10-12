"""可视化工具：图像网格、训练曲线、性能对比图。"""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from .utils import denorm


def _to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """[C, H, W] 或 [1, C, H, W] 张量 -> [H, W] 或 [H, W, C] uint8 numpy。"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = denorm(tensor.detach().cpu()).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img if img.shape[-1] == 3 else img[..., 0]


def show_grid(images: torch.Tensor, nrow: int = 8, title: str = "",
              save_path: Optional[str] = None, cmap: str = "gray"):
    """把一批 [N, C, H, W] 图像画成网格。"""
    grid = make_grid(denorm(images).clamp(0, 1), nrow=nrow)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    if grid.shape[-1] == 1:
        grid = grid[..., 0]

    plt.figure(figsize=(nrow, max(1, len(images) // nrow) * 1.2))
    plt.imshow(grid, cmap=cmap if grid.ndim == 2 else None)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


def show_dataset_samples(images_by_class: dict, n_per_class: int = 8,
                         save_path: Optional[str] = None):
    """每个类别一行，展示若干样例。"""
    n_classes = len(images_by_class)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 1.5, n_classes * 1.6))
    if n_classes == 1:
        axes = axes[None, :]
    for i, (cls, imgs) in enumerate(images_by_class.items()):
        imgs = imgs[:n_per_class]
        axes[i, 0].set_ylabel(cls, fontsize=10, rotation=0, labelpad=60, va="center")
        for j in range(n_per_class):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(imgs):
                ax.imshow(_to_numpy_img(imgs[j]), cmap="gray")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


def plot_training_curves(log_path: str, columns: List[str], title: str = "训练曲线",
                         x_col: str = "iter", save_path: Optional[str] = None):
    """读取训练 CSV 并绘制损失/指标曲线。"""
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 2.6 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        if col in df.columns:
            ax.plot(df[x_col], df[col], lw=1.2)
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)
    axes[-1].set_xlabel(x_col)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


def plot_metric_comparison(metrics: dict, metric_name: str = "FID",
                           save_path: Optional[str] = None, lower_is_better: bool = True):
    """绘制两个模型的指标对比柱状图。metrics: {model_name: value}。"""
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(names, values, color=colors[:len(names)], width=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} 对比（{'越低越好' if lower_is_better else '越高越好'}）")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()
