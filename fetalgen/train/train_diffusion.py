"""Diffusion (DDPM) 训练脚本。

对胎儿超声图像做类别条件生成（6 个切面类别）。训练目标为预测噪声的 MSE。

训练过程中会周期性:
    - 用 EMA 权重做 DDIM 加速采样，计算 FID（生成图 vs 真实图）
    - 保存可视化网格与检查点
"""
from __future__ import annotations

import copy
import dataclasses
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import PLANES, DiffusionConfig, resolve_device
from ..dataset import FetalPlanesDataset
from ..metrics import FIDComputer
from ..models import GaussianDiffusion, UNet
from ..utils import (AverageMeter, CSVLogger, EMA, make_exp_dirs,
                     save_image_grid, set_seed)
from ._common import prepare_balanced_reference


def train_diffusion(cfg: DiffusionConfig):
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    make_exp_dirs(cfg.log_dir, cfg.checkpoint_dir, cfg.sample_dir)

    d = cfg.data
    planes = d.planes if d.planes is not None else list(PLANES)
    n_classes = len(planes)
    print(f"[Diffusion] 设备={device}, 类别={planes}, "
          f"beta_schedule={cfg.beta_schedule}, timesteps={cfg.timesteps}")

    # ---------- 数据 ----------
    train_ds = FetalPlanesDataset(d.csv_path, d.img_dir, split="train",
                                  planes=planes, img_size=d.img_size, augment=True)
    test_ds = FetalPlanesDataset(d.csv_path, d.img_dir, split="test",
                                 planes=planes, img_size=d.img_size)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=d.num_workers, drop_last=True, pin_memory=True)
    print(f"训练样本数: {len(train_ds)}, 测试样本数: {len(test_ds)}")

    # ---------- 模型 ----------
    num_classes = n_classes if cfg.class_cond else None
    unet = UNet(
        in_channels=cfg.in_channels, out_channels=cfg.out_channels,
        base_channels=cfg.base_channels, channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout, num_classes=num_classes,
    )
    diffusion = GaussianDiffusion(unet, timesteps=cfg.timesteps,
                                  beta_schedule=cfg.beta_schedule).to(device)

    opt = torch.optim.Adam(diffusion.parameters(), lr=cfg.lr,
                           betas=cfg.betas, weight_decay=cfg.weight_decay)
    ema = EMA(unet, cfg.ema_decay)

    # ---------- 评估准备 ----------
    fid_computer = None  # 惰性创建，避免未评估时也下载 InceptionV3
    n_per_class = max(1, cfg.n_eval_samples // n_classes)
    real_ref = prepare_balanced_reference(test_ds, n_per_class, seed=cfg.seed)

    train_log = CSVLogger(os.path.join(cfg.log_dir, "train_log.csv"),
                          ["epoch", "iter", "loss", "lr"])
    eval_log = CSVLogger(os.path.join(cfg.log_dir, "fid_history.csv"),
                         ["epoch", "fid"])

    # ---------- 训练 ----------
    global_iter = 0
    best_fid = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        diffusion.train()
        loss_meter = AverageMeter()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for x, c in pbar:
            x = x.to(device)
            c = c.to(device) if cfg.class_cond else None

            loss = diffusion.training_loss(x, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(unet)

            global_iter += 1
            loss_meter.update(loss.item())

            if cfg.max_iters is not None and global_iter >= cfg.max_iters:
                break

            if global_iter % cfg.log_interval == 0:
                train_log.log(epoch=epoch, iter=global_iter,
                              loss=round(loss_meter.avg, 6), lr=opt.param_groups[0]["lr"])
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if cfg.max_iters is not None and global_iter >= cfg.max_iters:
            break

        # ---------- 评估 ----------
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            if fid_computer is None:
                fid_computer = FIDComputer(device)
            # 用 EMA 权重构造一个用于采样/评估的模型副本
            ema_model = copy.deepcopy(unet)
            ema.copy_to(ema_model)
            ema_diffusion = GaussianDiffusion(ema_model, timesteps=cfg.timesteps,
                                              beta_schedule=cfg.beta_schedule).to(device)
            ema_diffusion.eval()

            # 每类生成 n_per_class 张图（类别均衡）
            classes = torch.arange(n_classes).repeat_interleave(n_per_class).to(device)
            shape = (classes.shape[0], cfg.in_channels, d.img_size, d.img_size)
            fake_ref = ema_diffusion.ddim_sample(
                shape, c=classes if cfg.class_cond else None,
                steps=min(cfg.ddim_timesteps, cfg.timesteps), eta=0.0, device=device,
            ).detach().cpu()

            fid = fid_computer.compute(real_ref.cpu(), fake_ref)
            eval_log.log(epoch=epoch, fid=round(fid, 3))
            print(f"[评估] epoch={epoch} FID={fid:.3f}")

            # 保存可视化网格（前 n_save_samples 张）
            save_image_grid(fake_ref[:cfg.n_save_samples],
                            os.path.join(cfg.sample_dir, f"epoch_{epoch:03d}.png"),
                            nrow=8)

            # 保存检查点（EMA 权重 + 原始权重 + 完整配置）
            torch.save({
                "epoch": epoch,
                "model": unet.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": opt.state_dict(),
                "fid": fid,
                "config": dataclasses.asdict(cfg),
                "num_classes": num_classes,
            }, os.path.join(cfg.checkpoint_dir, "latest.pt"))
            if fid < best_fid:
                best_fid = fid
                torch.save(unet.state_dict(),
                           os.path.join(cfg.checkpoint_dir, "best_fid_model.pt"))

    print(f"训练完成，最佳 FID = {best_fid:.3f}")
    return best_fid


if __name__ == "__main__":
    import sys
    from ..config import load_config

    cfg = load_config(sys.argv[1]) if len(sys.argv) > 1 else DiffusionConfig()
    train_diffusion(cfg)
