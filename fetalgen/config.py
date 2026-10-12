"""统一配置模块。

所有超参数均以 dataclass 组织，可通过 YAML 配置文件覆盖默认值。
这样训练脚本、评估脚本与 Jupyter 都能复用同一套参数定义。
"""
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

# FETAL_PLANES_DB 的 6 个类别（与 CSV 中 Plane 字段一致）
PLANES = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]

# 胎儿脑部的 3 个子切面 + 非脑部标记
BRAIN_PLANES = [
    "Trans-thalamic",
    "Trans-cerebellum",
    "Trans-ventricular",
    "Not A Brain",
    "Other",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataConfig:
    """数据相关配置。"""

    data_root: str = "data/FETAL_PLANES_ZENODO"      # 数据集根目录
    img_dir: str = "data/FETAL_PLANES_ZENODO/Images" # 图像目录
    csv_path: str = "data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv"  # 元数据 CSV
    img_size: int = 128          # 统一缩放后的图像尺寸
    channels: int = 1            # 超声图像为灰度，1 通道
    num_workers: int = 4         # DataLoader 进程数
    # CycleGAN 做两个切面之间的无配对翻译：source -> target
    source_plane: str = "Fetal brain"
    target_plane: str = "Fetal femur"
    # Diffusion 类别条件生成时使用的类别；None 表示全部 6 类
    planes: Optional[List[str]] = None

    @property
    def all_planes(self) -> List[str]:
        return list(PLANES)

    def __post_init__(self):
        # 允许用户把相对路径解释为相对于项目根目录
        for attr in ("data_root", "img_dir", "csv_path"):
            v = getattr(self, attr)
            if v and not os.path.isabs(v):
                setattr(self, attr, os.path.join(PROJECT_ROOT, v))


@dataclass
class CycleGANConfig:
    """CycleGAN 模型 + 训练配置。"""

    name: str = "cyclegan"
    data: DataConfig = field(default_factory=DataConfig)

    # 通用
    device: str = "auto"         # auto / cuda / cpu
    seed: int = 42

    # 训练
    batch_size: int = 8
    epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lr_decay_epoch: int = 25     # 从该 epoch 开始线性衰减学习率

    # 模型结构
    input_nc: int = 1
    output_nc: int = 1
    ngf: int = 64                # 生成器基础通道数
    ndf: int = 64                # 判别器基础通道数
    n_resblocks: int = 9         # 生成器残差块数量
    n_layers_D: int = 3          # 判别器下采样层数

    # 损失权重
    gan_mode: str = "lsgan"      # lsgan / vanilla
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    use_identity: bool = True

    # 日志 / 保存
    log_dir: str = "logs/cyclegan"
    checkpoint_dir: str = "checkpoints/cyclegan"
    sample_dir: str = "outputs/cyclegan"
    log_interval: int = 20       # 每多少个 iter 打印一次
    eval_interval: int = 5       # 每多少个 epoch 评估一次 FID
    n_eval_samples: int = 256    # FID 评估采样数量
    n_save_samples: int = 16     # 可视化网格数量
    max_iters: Optional[int] = None  # 调试用：最多训练多少个 iter（None 表示不限）

    def __post_init__(self):
        self.data.__post_init__()
        for attr in ("log_dir", "checkpoint_dir", "sample_dir"):
            v = getattr(self, attr)
            if v and not os.path.isabs(v):
                setattr(self, attr, os.path.join(PROJECT_ROOT, v))


@dataclass
class DiffusionConfig:
    """Diffusion (DDPM) 模型 + 训练配置。"""

    name: str = "diffusion"
    data: DataConfig = field(default_factory=DataConfig)

    # 通用
    device: str = "auto"
    seed: int = 42

    # 训练
    batch_size: int = 16
    epochs: int = 100
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    # 扩散过程
    timesteps: int = 1000
    beta_schedule: str = "cosine"   # linear / cosine
    ddim_timesteps: int = 100       # 采样时用 DDIM 加速的步数（<= timesteps）

    # 模型结构（U-Net）
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.0
    class_cond: bool = True         # 是否做类别条件生成
    num_classes: int = 6

    # EMA（指数滑动平均，提升采样质量）
    ema_decay: float = 0.9999

    # 日志 / 保存
    log_dir: str = "logs/diffusion"
    checkpoint_dir: str = "checkpoints/diffusion"
    sample_dir: str = "outputs/diffusion"
    log_interval: int = 20
    eval_interval: int = 10
    n_eval_samples: int = 256
    n_save_samples: int = 16
    max_iters: Optional[int] = None  # 调试用：最多训练多少个 iter（None 表示不限）

    def __post_init__(self):
        self.data.__post_init__()
        for attr in ("log_dir", "checkpoint_dir", "sample_dir"):
            v = getattr(self, attr)
            if v and not os.path.isabs(v):
                setattr(self, attr, os.path.join(PROJECT_ROOT, v))


def _construct(cls, d: Dict[str, Any]):
    """递归地把 dict 构造成对应 dataclass，忽略未知字段。"""
    if not isinstance(d, dict):
        return d
    # 注意：本文件开启了 `from __future__ import annotations`，
    # 字段类型会以字符串形式存储，必须用 get_type_hints 还原真实类型。
    from typing import get_type_hints

    try:
        field_types = get_type_hints(cls)
    except Exception:
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in field_types:
            continue
        ftype = field_types[k]
        if dataclasses.is_dataclass(ftype):
            kwargs[k] = _construct(ftype, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str):
    """从 YAML 文件加载配置，自动根据文件内容判断是 CycleGAN 还是 Diffusion。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model = raw.get("model", raw.get("name", ""))
    if "diffusion" in model.lower() or "timesteps" in raw or "beta_schedule" in raw:
        cls = DiffusionConfig
    else:
        cls = CycleGANConfig
    return _construct(cls, raw)


def resolve_device(device: str = "auto") -> str:
    """解析设备字符串。"""
    if device in ("cuda", "gpu"):
        return "cuda"
    if device == "cpu":
        return "cpu"
    # auto：优先 GPU
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # 简单自检：打印默认配置
    print(yaml.safe_dump(dataclasses.asdict(CycleGANConfig()), allow_unicode=True, sort_keys=False))
    print(yaml.safe_dump(dataclasses.asdict(DiffusionConfig()), allow_unicode=True, sort_keys=False))
