"""训练过程共享的评估辅助函数。"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import torch


def prepare_real_reference(dataset, n: int, seed: int = 0,
                           device: str = "cpu") -> torch.Tensor:
    """确定性采样 n 张真实图像作为 FID 的固定参考集。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    imgs = [dataset[i][0] for i in idx]
    return torch.stack(imgs).to(device)


def prepare_balanced_reference(dataset, n_per_class: int, seed: int = 0,
                               device: str = "cpu") -> torch.Tensor:
    """每类采样 n_per_class 张真实图像，用于类别条件模型的 FID 参考。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    buckets = defaultdict(list)
    for i in order:
        x, lbl = dataset[int(i)]
        if len(buckets[lbl]) < n_per_class:
            buckets[lbl].append(x)
    imgs = []
    for lbl in sorted(buckets):
        imgs.extend(buckets[lbl][:n_per_class])
    return torch.stack(imgs).to(device)


def translate_collect(generator, loader, n: int, device: str) -> torch.Tensor:
    """用生成器翻译测试集图像，收集最多 n 张结果（CPU tensor）。"""
    out = []
    for x, _ in loader:
        with torch.no_grad():
            fake = generator(x.to(device)).detach().cpu()
        out.append(fake)
        if sum(t.shape[0] for t in out) >= n:
            break
    return torch.cat(out, dim=0)[:n]
