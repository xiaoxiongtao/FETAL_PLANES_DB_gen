"""命令行入口：训练 / 采样 / 评估。

用法示例:
    python main.py train-cyclegan --config configs/cyclegan.yaml
    python main.py train-diffusion --config configs/diffusion.yaml

    python main.py sample-cyclegan --checkpoint checkpoints/cyclegan/best_fid.pt --n 16
    python main.py sample-diffusion --checkpoint checkpoints/diffusion/latest.pt --per-class 8

    python main.py evaluate --model cyclegan --checkpoint checkpoints/cyclegan/best_fid.pt
    python main.py evaluate --model diffusion --checkpoint checkpoints/diffusion/latest.pt
"""
from __future__ import annotations

import os

# Windows 下 PyTorch 与 scipy 的 OpenMP 运行库冲突的规避（需在导入 torch 前设置）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os

import torch
from torch.utils.data import DataLoader

from fetalgen.config import (PLANES, CycleGANConfig, DiffusionConfig,
                             load_config, resolve_device)
from fetalgen.dataset import FetalPlanesDataset
from fetalgen.loaders import construct_config, load_cyclegan, load_diffusion
from fetalgen.metrics import FIDComputer, compute_pairwise_metrics
from fetalgen.train import train_cyclegan, train_diffusion
from fetalgen.train._common import (prepare_balanced_reference,
                                    prepare_real_reference, translate_collect)
from fetalgen.utils import save_image_grid, set_seed


# --------------------------- 训练命令 --------------------------- #

def cmd_train_cyclegan(args):
    cfg = load_config(args.config) if args.config else CycleGANConfig()
    train_cyclegan(cfg)


def cmd_train_diffusion(args):
    cfg = load_config(args.config) if args.config else DiffusionConfig()
    train_diffusion(cfg)


# --------------------------- 采样命令 --------------------------- #

def cmd_sample_cyclegan(args):
    device = resolve_device("auto")
    cfg = load_config(args.config) if args.config else CycleGANConfig()
    models, ckpt = load_cyclegan(args.checkpoint, device, cfg)
    G_AB = models["G_AB"]
    d = cfg.data

    ds = FetalPlanesDataset(d.csv_path, d.img_dir, split="test",
                            planes=[d.source_plane], img_size=d.img_size)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    src = []
    for x, _ in loader:
        src.append(x)
        if sum(t.shape[0] for t in src) >= args.n:
            break
    src = torch.cat(src, dim=0)[:args.n]

    with torch.no_grad():
        fake = G_AB(src.to(device)).detach().cpu()

    out = args.out or os.path.join("outputs", "cyclegan", "sample.png")
    save_image_grid(torch.cat([src, fake], dim=0), out, nrow=args.n)
    print(f"已保存 {args.n} 组翻译结果（上行输入，下行翻译）到 {out}")


def cmd_sample_diffusion(args):
    device = resolve_device("auto")
    diffusion, ckpt = load_diffusion(args.checkpoint, device, use_ema=True)
    arch = construct_config(DiffusionConfig, ckpt.get("config", {}) or {})
    img_size = arch.data.img_size
    n_classes = ckpt.get("num_classes", arch.num_classes)

    per = args.per_class
    classes = torch.arange(n_classes).repeat_interleave(per).to(device)
    shape = (classes.shape[0], arch.in_channels, img_size, img_size)
    samples = diffusion.ddim_sample(shape, c=classes if arch.class_cond else None,
                                    steps=min(args.steps, arch.timesteps),
                                    eta=0.0, device=device).detach().cpu()

    out = args.out or os.path.join("outputs", "diffusion", "sample.png")
    save_image_grid(samples, out, nrow=per)
    print(f"已保存 {samples.shape[0]} 张生成图像（每类 {per} 张）到 {out}")


# --------------------------- 评估命令 --------------------------- #

def cmd_evaluate(args):
    device = resolve_device("auto")
    fid_computer = FIDComputer(device)

    if args.model == "cyclegan":
        cfg = load_config(args.config) if args.config else CycleGANConfig()
        models, ckpt = load_cyclegan(args.checkpoint, device, cfg)
        G_AB = models["G_AB"]
        d = cfg.data
        test_A = FetalPlanesDataset(d.csv_path, d.img_dir, split="test",
                                    planes=[d.source_plane], img_size=d.img_size)
        test_B = FetalPlanesDataset(d.csv_path, d.img_dir, split="test",
                                    planes=[d.target_plane], img_size=d.img_size)
        loader_A = DataLoader(test_A, batch_size=8, shuffle=False, num_workers=0)
        loader_B = DataLoader(test_B, batch_size=8, shuffle=False, num_workers=0)

        real_ref = prepare_real_reference(test_B, args.n, seed=cfg.seed)
        fake_ref = translate_collect(G_AB, loader_A, args.n, device)

        # 循环一致性
        xA = next(iter(loader_A))[0].to(device)
        rec = models["G_BA"](G_AB(xA))
        pair = compute_pairwise_metrics(xA.cpu(), rec.cpu())

        fid = fid_computer.compute(real_ref.cpu(), fake_ref.cpu())
        print(f"[CycleGAN] FID={fid:.3f} | cycle SSIM={pair['ssim']:.4f} "
              f"PSNR={pair['psnr']:.2f} LPIPS={pair['lpips']}")

    elif args.model == "diffusion":
        diffusion, ckpt = load_diffusion(args.checkpoint, device, use_ema=True)
        arch = construct_config(DiffusionConfig, ckpt.get("config", {}) or {})
        img_size = arch.data.img_size
        n_classes = ckpt.get("num_classes", arch.num_classes)
        planes = arch.data.planes or list(PLANES)

        test_ds = FetalPlanesDataset(arch.data.csv_path, arch.data.img_dir, split="test",
                                     planes=planes, img_size=img_size)
        n_per = max(1, args.n // n_classes)
        real_ref = prepare_balanced_reference(test_ds, n_per, seed=arch.seed)

        classes = torch.arange(n_classes).repeat_interleave(n_per).to(device)
        shape = (classes.shape[0], arch.in_channels, img_size, img_size)
        fake_ref = diffusion.ddim_sample(
            shape, c=classes if arch.class_cond else None,
            steps=min(args.steps, arch.timesteps), eta=0.0, device=device,
        ).detach().cpu()

        fid = fid_computer.compute(real_ref.cpu(), fake_ref)
        print(f"[Diffusion] FID={fid:.3f}")

    else:
        raise ValueError(f"未知模型: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="胎儿多切面超声图像生成（CycleGAN / Diffusion）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train-cyclegan", help="训练 CycleGAN")
    p.add_argument("--config", default="configs/cyclegan.yaml")
    p.set_defaults(func=cmd_train_cyclegan)

    p = sub.add_parser("train-diffusion", help="训练 Diffusion")
    p.add_argument("--config", default="configs/diffusion.yaml")
    p.set_defaults(func=cmd_train_diffusion)

    p = sub.add_parser("sample-cyclegan", help="用 CycleGAN 生成翻译样本")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_sample_cyclegan)

    p = sub.add_parser("sample-diffusion", help="用 Diffusion 生成样本")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--steps", type=int, default=100, help="DDIM 采样步数")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_sample_diffusion)

    p = sub.add_parser("evaluate", help="在测试集上评估 FID 等指标")
    p.add_argument("--model", choices=["cyclegan", "diffusion"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
