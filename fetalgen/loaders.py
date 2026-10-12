"""检查点加载工具，供命令行与 Jupyter 复用。"""
from __future__ import annotations

import copy
import dataclasses
from typing import Dict, Optional, Tuple

import torch

from .config import CycleGANConfig, DiffusionConfig, resolve_device
from .models import Discriminator, GaussianDiffusion, Generator, UNet


def load_cyclegan(checkpoint_path: str, device: str = "auto",
                  cfg: Optional[CycleGANConfig] = None) -> Tuple[Dict[str, torch.nn.Module], dict]:
    """加载 CycleGAN 检查点，返回模型字典与检查点元信息。

    返回:
        models: {"G_AB", "G_BA", "D_A", "D_B"}
        meta:   检查点原始内容（含 config / fid / epoch）
    """
    device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 优先用检查点内保存的配置，其次用外部传入的配置，最后用默认
    arch = cfg
    if arch is None and "config" in ckpt:
        raw = ckpt["config"]
        arch = construct_config(CycleGANConfig, raw)
    if arch is None:
        arch = CycleGANConfig()

    G_AB = Generator(arch.input_nc, arch.output_nc, arch.ngf, arch.n_resblocks)
    G_BA = Generator(arch.input_nc, arch.output_nc, arch.ngf, arch.n_resblocks)
    D_A = Discriminator(arch.input_nc, arch.ndf, arch.n_layers_D)
    D_B = Discriminator(arch.input_nc, arch.ndf, arch.n_layers_D)

    G_AB.load_state_dict(ckpt["G_AB"])
    G_BA.load_state_dict(ckpt["G_BA"])
    D_A.load_state_dict(ckpt.get("D_A", {}))
    D_B.load_state_dict(ckpt.get("D_B", {}))

    models = {"G_AB": G_AB.to(device), "G_BA": G_BA.to(device),
              "D_A": D_A.to(device), "D_B": D_B.to(device)}
    for m in models.values():
        m.eval()
    return models, ckpt


def load_diffusion(checkpoint_path: str, device: str = "auto",
                   use_ema: bool = True,
                   cfg: Optional[DiffusionConfig] = None) -> Tuple[GaussianDiffusion, dict]:
    """加载 Diffusion 检查点，返回可采样的 GaussianDiffusion 与元信息。"""
    device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 解析架构配置
    arch = cfg
    saved_cfg = ckpt.get("config", {})
    if arch is None and saved_cfg:
        arch = construct_config(DiffusionConfig, saved_cfg)
    if arch is None:
        arch = DiffusionConfig()

    num_classes = ckpt.get("num_classes", arch.num_classes if arch.class_cond else None)
    if not arch.class_cond:
        num_classes = None

    unet = UNet(
        in_channels=arch.in_channels, out_channels=arch.out_channels,
        base_channels=arch.base_channels, channel_mults=tuple(arch.channel_mults),
        num_res_blocks=arch.num_res_blocks,
        attention_resolutions=tuple(arch.attention_resolutions),
        dropout=arch.dropout, num_classes=num_classes,
    )

    # 权重：优先 EMA；latest.pt 中 ema 是 {"decay","shadow"}，best_fid_model.pt 是裸 state_dict
    if use_ema and "ema" in ckpt:
        shadow = ckpt["ema"].get("shadow", ckpt["ema"])
        unet.load_state_dict(shadow)
    elif use_ema and "ema" not in ckpt and "model" in ckpt:
        unet.load_state_dict(ckpt["model"])
    else:
        state = ckpt.get("model", ckpt.get("ema", ckpt))
        if "shadow" in state:
            state = state["shadow"]
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        unet.load_state_dict(state)

    diffusion = GaussianDiffusion(unet, timesteps=arch.timesteps,
                                  beta_schedule=arch.beta_schedule).to(device)
    diffusion.eval()
    return diffusion, ckpt


def construct_config(cls, raw: dict):
    """把（可能不完整的）配置 dict 构造成 dataclass，缺失字段用默认值。

    可直接用 `construct_config(DiffusionConfig, ckpt["config"])` 还原训练时的配置。
    """
    defaults = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            defaults[f.name] = f.default_factory()
    merged = {**defaults, **{k: v for k, v in raw.items() if k in defaults}}
    # 嵌套 data 配置（注意 future annotations 会把类型变成字符串，需用 get_type_hints 解析）
    try:
        from typing import get_type_hints
        hints = get_type_hints(cls)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}
    for name, ftype in hints.items():
        if dataclasses.is_dataclass(ftype) and isinstance(merged.get(name), dict):
            merged[name] = construct_config(ftype, merged[name])
    return cls(**merged)
