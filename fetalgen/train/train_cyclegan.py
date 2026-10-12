"""CycleGAN 训练脚本。

在两个胎儿超声切面之间做无配对翻译：
    source_plane (A) <-> target_plane (B)

损失组成:
    - 对抗损失 (LSGAN)
    - 循环一致性损失 (L1, 权重 lambda_cycle)
    - 恒等映射损失 (L1, 权重 lambda_identity * lambda_cycle)

训练过程中会周期性:
    - 计算 FID（生成的目标切面 vs 真实目标切面）
    - 计算循环一致性 SSIM/PSNR
    - 保存可视化网格与检查点
"""
from __future__ import annotations

import dataclasses
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import CycleGANConfig, resolve_device
from ..dataset import FetalPlanesDataset, build_dataloader
from ..metrics import FIDComputer, compute_pairwise_metrics
from ..models import Discriminator, GANLoss, Generator
from ..utils import (CSVLogger, LinearDecayLR, AverageMeter, make_exp_dirs,
                     save_image_grid, set_seed)
from ._common import prepare_real_reference, translate_collect


def train_cyclegan(cfg: CycleGANConfig):
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    make_exp_dirs(cfg.log_dir, cfg.checkpoint_dir, cfg.sample_dir)

    d = cfg.data
    print(f"[CycleGAN] 设备={device}, 翻译方向: {d.source_plane} -> {d.target_plane}")

    # ---------- 数据 ----------
    loader_A = build_dataloader(d.csv_path, d.img_dir, split="train",
                                planes=[d.source_plane], img_size=d.img_size,
                                batch_size=cfg.batch_size, num_workers=d.num_workers,
                                augment=True, shuffle=True)
    loader_B = build_dataloader(d.csv_path, d.img_dir, split="train",
                                planes=[d.target_plane], img_size=d.img_size,
                                batch_size=cfg.batch_size, num_workers=d.num_workers,
                                augment=True, shuffle=True)
    # 测试集（评估用）
    test_A = FetalPlanesDataset(d.csv_path, d.img_dir, split="test",
                                planes=[d.source_plane], img_size=d.img_size)
    test_B = FetalPlanesDataset(d.csv_path, d.img_dir, split="test",
                                planes=[d.target_plane], img_size=d.img_size)
    loader_A_test = DataLoader(test_A, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=d.num_workers, drop_last=False)
    loader_B_test = DataLoader(test_B, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=d.num_workers, drop_last=False)

    print(f"训练集: {d.source_plane}={len(loader_A.dataset)}, "
          f"{d.target_plane}={len(loader_B.dataset)}")

    # ---------- 模型 ----------
    G_AB = Generator(cfg.input_nc, cfg.output_nc, cfg.ngf, cfg.n_resblocks).to(device)
    G_BA = Generator(cfg.input_nc, cfg.output_nc, cfg.ngf, cfg.n_resblocks).to(device)
    D_A = Discriminator(cfg.input_nc, cfg.ndf, cfg.n_layers_D).to(device)
    D_B = Discriminator(cfg.input_nc, cfg.ndf, cfg.n_layers_D).to(device)

    # ---------- 优化器与调度器 ----------
    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                             lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                             lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    sched_G = LinearDecayLR(opt_G, cfg.epochs, cfg.lr_decay_epoch)
    sched_D = LinearDecayLR(opt_D, cfg.epochs, cfg.lr_decay_epoch)

    criterion_gan = GANLoss(cfg.gan_mode)
    criterion_l1 = nn.L1Loss()

    # ---------- 评估准备 ----------
    fid_computer = None  # 惰性创建，避免未评估时也下载 InceptionV3
    real_ref_B = prepare_real_reference(test_B, cfg.n_eval_samples, seed=cfg.seed)

    train_log = CSVLogger(os.path.join(cfg.log_dir, "train_log.csv"),
                          ["epoch", "iter", "loss_D", "loss_G_gan", "loss_G_cycle",
                           "loss_G_id", "loss_G_total", "lr"])
    eval_log = CSVLogger(os.path.join(cfg.log_dir, "fid_history.csv"),
                         ["epoch", "fid", "cycle_ssim", "cycle_psnr"])

    # ---------- 训练 ----------
    global_iter = 0
    best_fid = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        meter = {k: AverageMeter() for k in
                 ["D", "G_gan", "G_cycle", "G_id", "G_total"]}
        pbar = tqdm(zip(loader_A, loader_B), total=min(len(loader_A), len(loader_B)),
                    desc=f"Epoch {epoch}/{cfg.epochs}")

        for (real_A, _), (real_B, _) in pbar:
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # ---- 前向 ----
            fake_B = G_AB(real_A)          # A -> B
            rec_A = G_BA(fake_B)           # B -> A（循环）
            fake_A = G_BA(real_B)          # B -> A
            rec_B = G_AB(fake_A)           # A -> B（循环）

            # 恒等映射
            id_A = G_BA(real_A)
            id_B = G_AB(real_B)

            # ---- 生成器损失 ----
            loss_G_gan = criterion_gan(D_B(fake_B), True) + criterion_gan(D_A(fake_A), True)
            loss_G_cycle = criterion_l1(rec_A, real_A) + criterion_l1(rec_B, real_B)
            loss_G_id = criterion_l1(id_A, real_A) + criterion_l1(id_B, real_B)
            loss_G = (loss_G_gan
                      + cfg.lambda_cycle * loss_G_cycle
                      + cfg.lambda_identity * cfg.lambda_cycle * loss_G_id)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # ---- 判别器损失 ----
            loss_D_A = 0.5 * (criterion_gan(D_A(real_A), True)
                              + criterion_gan(D_A(fake_A.detach()), False))
            loss_D_B = 0.5 * (criterion_gan(D_B(real_B), True)
                              + criterion_gan(D_B(fake_B.detach()), False))
            loss_D = loss_D_A + loss_D_B

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            global_iter += 1
            meter["D"].update(loss_D.item())
            meter["G_gan"].update(loss_G_gan.item())
            meter["G_cycle"].update(loss_G_cycle.item())
            meter["G_id"].update(loss_G_id.item())
            meter["G_total"].update(loss_G.item())

            if cfg.max_iters is not None and global_iter >= cfg.max_iters:
                break

            if global_iter % cfg.log_interval == 0:
                train_log.log(
                    epoch=epoch, iter=global_iter, loss_D=meter["D"].avg,
                    loss_G_gan=meter["G_gan"].avg, loss_G_cycle=meter["G_cycle"].avg,
                    loss_G_id=meter["G_id"].avg, loss_G_total=meter["G_total"].avg,
                    lr=sched_G.get_lr(),
                )
                pbar.set_postfix(D=f"{meter['D'].avg:.3f}",
                                 G=f"{meter['G_total'].avg:.3f}")

        sched_G.step(epoch)
        sched_D.step(epoch)

        if cfg.max_iters is not None and global_iter >= cfg.max_iters:
            break

        # ---------- 评估 ----------
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            G_AB.eval(); G_BA.eval()
            if fid_computer is None:
                fid_computer = FIDComputer(device)
            with torch.no_grad():
                fake_ref = translate_collect(G_AB, loader_A_test, cfg.n_eval_samples, device)
                fid = fid_computer.compute(real_ref_B.cpu(), fake_ref.cpu())

                # 循环一致性指标（输入 A -> B -> A 重建）
                xA = next(iter(loader_A_test))[0].to(device)
                rec = G_BA(G_AB(xA))
                pair_metrics = compute_pairwise_metrics(xA.cpu(), rec.cpu())

                # 可视化：第一行真实 A，第二行翻译后的 B
                vis_A = next(iter(loader_A_test))[0][:cfg.n_save_samples // 2]
                vis_fake_B = G_AB(vis_A.to(device))
                grid = torch.cat([vis_A.cpu(), vis_fake_B.cpu()], dim=0)
                save_image_grid(grid, os.path.join(
                    cfg.sample_dir, f"epoch_{epoch:03d}.png"),
                    nrow=cfg.n_save_samples // 2)

            eval_log.log(epoch=epoch, fid=round(fid, 3),
                         cycle_ssim=round(pair_metrics["ssim"], 4),
                         cycle_psnr=round(pair_metrics["psnr"], 3))
            print(f"[评估] epoch={epoch} FID={fid:.3f} "
                  f"cycle_SSIM={pair_metrics['ssim']:.4f} "
                  f"cycle_PSNR={pair_metrics['psnr']:.2f}")

            # 保存最新与最佳（一个文件包含全部四个网络 + 配置）
            ckpt = {
                "epoch": epoch,
                "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                "optimizer_G": opt_G.state_dict(), "optimizer_D": opt_D.state_dict(),
                "fid": fid,
                "config": dataclasses.asdict(cfg),
            }
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "latest.pt"))
            if fid < best_fid:
                best_fid = fid
                torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "best_fid.pt"))

    print(f"训练完成，最佳 FID = {best_fid:.3f}")
    return best_fid


if __name__ == "__main__":
    import sys
    from ..config import load_config

    cfg = load_config(sys.argv[1]) if len(sys.argv) > 1 else CycleGANConfig()
    train_cyclegan(cfg)
