"""FETAL_PLANES_DB 数据集加载。

数据组织方式（来自 Zenodo 记录 3904280）：
    data/FETAL_PLANES_ZENODO/
        Images/                         # 12400 张灰度 PNG，平铺存放
            Patient00001_Plane1_1_of_15.png
            ...
        FETAL_PLANES_DB_data.csv        # 元数据，分号分隔
        FETAL_PLANES_DB_data.xlsx

CSV 列（注意部分列名带尾随空格，读取时统一 strip）：
    Image_name; Patient_num; Plane; Brain_plane; Operator; US_Machine; Train
其中 Plane 为 6 大类，Train=1 表示训练集、0 表示测试集（论文划分）。
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import PLANES


def load_metadata(csv_path: str) -> pd.DataFrame:
    """读取元数据 CSV，统一去除列名与字符串字段的尾随空格。"""
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def _square_pad(img: Image.Image) -> Image.Image:
    """在短边补零（黑色背景），把任意尺寸图像变成正方形。

    超声图像背景为黑色，因此用 0 填充不会引入伪影，且避免直接拉伸导致形变。
    """
    w, h = img.size
    s = max(w, h)
    canvas = Image.new(img.mode, (s, s), 0)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


class FetalPlanesDataset(Dataset):
    """按 CSV 元数据加载图像。

    参数:
        csv_path:   元数据 CSV 路径
        img_dir:    图像目录
        split:      "train" / "test" / "all"
        planes:     仅保留这些 Plane 类别（None 表示全部）
        brain_planes: 仅保留这些 Brain_plane 子类别（None 表示不过滤）
        img_size:   统一缩放到 img_size x img_size
        channels:   1（灰度）
        augment:    训练时是否做随机水平翻转增强
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        split: str = "train",
        planes: Optional[List[str]] = None,
        brain_planes: Optional[List[str]] = None,
        img_size: int = 128,
        channels: int = 1,
        augment: bool = False,
    ):
        super().__init__()
        assert split in ("train", "test", "all"), f"未知 split: {split}"
        assert os.path.isfile(csv_path), f"找不到 CSV: {csv_path}"
        assert os.path.isdir(img_dir), f"找不到图像目录: {img_dir}"

        df = load_metadata(csv_path)
        if split != "all":
            keep = 1 if split == "train" else 0
            df = df[df["Train"] == keep]
        if planes is not None:
            df = df[df["Plane"].isin(planes)]
        if brain_planes is not None:
            df = df[df["Brain_plane"].isin(brain_planes)]
        self.df = df.reset_index(drop=True)

        self.img_dir = img_dir
        self.img_size = img_size
        self.channels = channels
        self.augment = augment

        # 建立类别索引（用于 Diffusion 类别条件）
        self.planes = list(planes) if planes is not None else list(PLANES)
        self.plane_to_idx = {p: i for i, p in enumerate(self.planes)}

        transform_list = [
            _square_pad,
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),                              # [0, 1]
            T.Normalize((0.5,) * channels, (0.5,) * channels),  # [-1, 1]
        ]
        if augment:
            transform_list.insert(0, T.RandomHorizontalFlip(p=0.5))
        self.transform = T.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        name = row["Image_name"]
        # CSV 中的 Image_name 不含扩展名，实际文件为 <name>.png
        if not os.path.splitext(name)[1]:
            name = name + ".png"
        path = os.path.join(self.img_dir, name)
        img = Image.open(path).convert("L")
        img = self.transform(img)              # [C, H, W] in [-1, 1]
        label = self.plane_to_idx[row["Plane"]]
        return img, label

    def class_counts(self) -> dict:
        return self.df["Plane"].value_counts().to_dict()


def build_dataloader(csv_path: str, img_dir: str, split: str = "train",
                     planes: Optional[List[str]] = None, img_size: int = 128,
                     batch_size: int = 8, num_workers: int = 4,
                     augment: bool = False, shuffle: bool = True,
                     drop_last: bool = True) -> DataLoader:
    """便捷函数：构建 DataLoader。"""
    ds = FetalPlanesDataset(
        csv_path=csv_path, img_dir=img_dir, split=split,
        planes=planes, img_size=img_size, augment=augment,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last, pin_memory=True,
    )


def collect_images_from_loader(loader: DataLoader, max_n: int,
                               device: str = "cpu") -> torch.Tensor:
    """从 DataLoader 收集最多 max_n 张图像，返回 [N, C, H, W] tensor。"""
    images = []
    for x, _ in loader:
        images.append(x.to(device))
        if sum(x.shape[0] for x in images) >= max_n:
            break
    return torch.cat(images, dim=0)[:max_n]


if __name__ == "__main__":
    # 自检：打印各类别数量
    root = "data/FETAL_PLANES_ZENODO"
    ds = FetalPlanesDataset(os.path.join(root, "FETAL_PLANES_DB_data.csv"),
                            os.path.join(root, "Images"), split="all")
    print("总样本数:", len(ds))
    print("类别分布:", ds.class_counts())
    img, label = ds[0]
    print("样本形状:", tuple(img.shape), "标签:", label, "取值范围:", float(img.min()), float(img.max()))
