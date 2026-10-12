# 胎儿多切面超声图像生成（FETAL_PLANES_DB）

基于 **CycleGAN** 与 **Diffusion (DDPM)** 两种生成模型，在公开数据集
[FETAL_PLANES_DB](https://zenodo.org/records/3904280) 上进行胎儿超声标准切面图像的
生成 / 跨切面翻译，并对两种方法的性能（FID、SSIM、PSNR、LPIPS、训练损失曲线）进行
对比。

---

## 1. 项目简介

超声是产前筛查最常用的成像手段，但高质量标注数据稀缺、不同切面（脑、胸、股骨、
腹部、宫颈等）的图像风格差异大。本项目实现并比较两类主流生成模型：

| 模型 | 任务 | 关键思想 |
| --- | --- | --- |
| **CycleGAN** | 两个切面之间的**无配对图像翻译**（如 `Fetal brain ↔ Fetal femur`） | 循环一致性损失 + PatchGAN 对抗损失 + 恒等损失 |
| **Diffusion (DDPM)** | 对 6 个切面类别做**条件生成** | 逐步去噪的马尔可夫过程 + U-Net 预测噪声，DDIM 加速采样 |

两者都在训练过程中实时记录损失，并周期性计算 **FID**（Fréchet Inception Distance）
作为生成质量主指标，最终在 Jupyter 中统一可视化与比较。

---

## 2. 数据集

- **来源**：Zenodo 记录 [3904280](https://zenodo.org/records/3904280)
- **下载文件**：`FETAL_PLANES_ZENODO.zip`（约 2.1 GB，MD5 `2a5fcc2cefb789bcc0f6c1f73e0ea43f`）
- **内容**：12400 张灰度超声图像 + 元数据 CSV

### 2.1 目录结构

```
data/FETAL_PLANES_ZENODO/
├── Images/                          # 12400 张 PNG，平铺存放
│   ├── Patient00001_Plane1_1_of_15.png
│   └── ...
├── FETAL_PLANES_DB_data.csv         # 元数据（分号分隔）
└── FETAL_PLANES_DB_data.xlsx
```

CSV 列为：`Image_name; Patient_num; Plane; Brain_plane; Operator; US_Machine; Train`
（注意部分列名带尾随空格，代码已做 `strip` 处理；`Image_name` 不含 `.png` 后缀）。

### 2.2 类别分布

| Plane（切面类别） | 样本数 | 训练 | 测试 |
| --- | --- | --- | --- |
| Fetal abdomen（胎儿腹部） | 711 | 353 | 358 |
| Fetal brain（胎儿脑部） | 3092 | 1620 | 1472 |
| Fetal femur（胎儿股骨） | 1040 | 516 | 524 |
| Fetal thorax（胎儿胸腔） | 1718 | 1058 | 660 |
| Maternal cervix（母体宫颈） | 1626 | 981 | 645 |
| Other（其他） | 4213 | 2601 | 1612 |
| **合计** | **12400** | **7129** | **5271** |

脑部还有子切面分类 `Brain_plane`：Trans-thalamic / Trans-cerebellum /
Trans-ventricular / Not A Brain / Other。

### 2.3 下载

```bash
python scripts/download_data.py
# 或手动下载解压到 data/ 目录：
# https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1
```

> 本项目默认假设数据位于 `data/FETAL_PLANES_ZENODO/`，如需修改请在
> `configs/*.yaml` 中调整 `data_root / img_dir / csv_path`。

---

## 3. 项目结构

```
FETAL_PLANES_DB_gen/
├── data/FETAL_PLANES_ZENODO/         # 数据集（需自行下载）
├── fetalgen/                         # 核心包
│   ├── config.py                     # dataclass 配置 + YAML 加载
│   ├── dataset.py                    # 数据加载 / 预处理 / 训练测试划分
│   ├── metrics.py                    # FID / SSIM / PSNR / LPIPS
│   ├── utils.py                      # 种子 / 日志 / 检查点 / EMA / 调度器
│   ├── visualize.py                  # 可视化工具
│   ├── loaders.py                    # 检查点加载（命令行与 Jupyter 共用）
│   ├── models/
│   │   ├── cyclegan.py               # ResNet 生成器 + PatchGAN 判别器 + GANLoss
│   │   └── diffusion.py              # U-Net + DDPM + DDIM
│   └── train/
│       ├── _common.py                # FID 参考集等评估辅助
│       ├── train_cyclegan.py         # CycleGAN 训练循环
│       └── train_diffusion.py        # Diffusion 训练循环
├── configs/
│   ├── cyclegan.yaml                 # CycleGAN 超参数
│   └── diffusion.yaml                # Diffusion 超参数
├── notebooks/
│   └── project_overview.ipynb        # 数据展示 / 模型加载 / 可视化 / 性能对比
├── scripts/download_data.py          # 数据集下载脚本
├── main.py                           # 命令行入口（训练 / 采样 / 评估）
├── requirements.txt
└── README.md
```

---

## 4. 环境安装

```bash
conda create -n fetalgen python=3.10 -y
conda activate fetalgen

# 先安装 PyTorch（按你的 CUDA 版本选择，见 pytorch.org）
# 例如 CUDA 12.x：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 再安装其余依赖
pip install -r requirements.txt
```

> 本项目已在 Python 3.14 + PyTorch 2.14 (CUDA) 上验证可运行。

---

## 5. 快速开始

### 5.1 训练 CycleGAN（切面翻译）

```bash
python main.py train-cyclegan --config configs/cyclegan.yaml
```

默认配置：`Fetal brain -> Fetal femur` 翻译，128×128 灰度图，50 epoch。
训练日志写入 `logs/cyclegan/train_log.csv` 与 `logs/cyclegan/fid_history.csv`，
检查点保存在 `checkpoints/cyclegan/`，可视化网格保存在 `outputs/cyclegan/`。

### 5.2 训练 Diffusion（类别条件生成）

```bash
python main.py train-diffusion --config configs/diffusion.yaml
```

默认配置：6 类条件生成，1000 步 cosine 调度，DDIM 100 步采样，EMA 权重。

### 5.3 采样

```bash
# CycleGAN 翻译可视化（上行输入、下行翻译）
python main.py sample-cyclegan --checkpoint checkpoints/cyclegan/best_fid.pt --n 16

# Diffusion 生成（每类 8 张）
python main.py sample-diffusion --checkpoint checkpoints/diffusion/latest.pt --per-class 8
```

### 5.4 评估

```bash
python main.py evaluate --model cyclegan --checkpoint checkpoints/cyclegan/best_fid.pt
python main.py evaluate --model diffusion --checkpoint checkpoints/diffusion/latest.pt
```

### 5.5 交互式对比

打开 `notebooks/project_overview.ipynb`，依次运行即可：展示数据集、加载网络与参数、
可视化结果、绘制训练曲线并对比两个模型的指标。

---

## 6. 模型与损失说明

### 6.1 CycleGAN

- **生成器**：ResNet-9（`c7s1-64 → d128 → d256 → 9×ResBlock → u128 → u64 → c7s1-1`），
  使用 InstanceNorm，输出 `Tanh` 到 `[-1, 1]`。
- **判别器**：70×70 PatchGAN，输出 patch 置信图。
- **损失**（默认 LSGAN）：

$$ \mathcal{L}_{G} = \underbrace{\mathcal{L}_{GAN}(G_{AB}) + \mathcal{L}_{GAN}(G_{BA})}_{\text{对抗损失}}
+ \underbrace{\lambda_{cyc}\,\big(\|G_{BA}(G_{AB}(A))-A\|_1 + \|G_{AB}(G_{BA}(B))-B\|_1\big)}_{\text{循环一致性损失}}
+ \underbrace{\lambda_{id}\lambda_{cyc}\,\big(\|G_{BA}(A)-A\|_1 + \|G_{AB}(B)-B\|_1\big)}_{\text{恒等损失}} $$

- 判别器损失为真实/生成各 0.5 权重的 MSE；学习率从 `lr_decay_epoch` 起线性衰减。

### 6.2 Diffusion (DDPM)

- **前向**：$q(x_t|x_0)=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$。
- **损失**：预测噪声的 MSE $\|\epsilon_\theta(x_t,t,c)-\epsilon\|^2$，
  `c` 为类别嵌入（与时间步嵌入相加注入 U-Net）。
- **结构**：U-Net（GroupNorm + SiLU 残差块 + 自注意力 + 类别/时间嵌入），
  支持 `linear` / `cosine` 两种 beta 调度。
- **采样**：标准 DDPM 逐步去噪，或 **DDIM** 加速（`ddim_timesteps` 步，`eta=0` 确定）。
- **EMA**：训练中维护指数滑动平均权重，采样/评估时使用，显著提升质量。

---

## 7. 评估指标

| 指标 | 含义 | 用途 |
| --- | --- | --- |
| **FID** | 生成分布与真实分布在 InceptionV3 特征空间的 Fréchet 距离，越低越好 | 主指标，两模型通用 |
| **SSIM** | 结构相似度 | 配对图像质量（CycleGAN 循环一致性） |
| **PSNR** | 峰值信噪比 | 配对图像质量 |
| **LPIPS** | 感知相似度（可选，需装 `lpips`） | 配对感知质量 |

CycleGAN 额外评估**循环一致性**：输入 $A \to B \to A$ 重建后与原图的 SSIM/PSNR，
衡量翻译的可逆性与内容保持能力。Diffusion 以 FID 为主（无条件生成无自然配对）。

---

## 8. 复现与调参建议

- **切面选择**：改 `configs/cyclegan.yaml` 的 `source_plane / target_plane`。
  推荐样本量相近的组合，如 `Fetal thorax ↔ Maternal cervix`。
- **显存不足**：调小 `batch_size`、`img_size`，或减小 `base_channels`。
- **Diffusion 加速采样**：增大 `ddim_timesteps` 质量更好、更慢；`eta=0` 为确定性采样。
- **日志可视化**：`train_log.csv`（损失曲线）与 `fid_history.csv`（FID 历史）可直接用
  pandas 读取，Jupyter 中已提供绘图代码。
- 若只想快速验证管线，可在配置里加 `max_iters: 100` 只跑 100 个迭代。

---

## 9. 常见问题

- **Windows 下 FID 计算崩溃 / 报 `libiomp5md.dll` 错误**：PyTorch 与 scipy 的 OpenMP
  运行库冲突所致。本项目已在 `fetalgen/__init__.py` 与 `main.py` 中自动设置
  `KMP_DUPLICATE_LIB_OK=TRUE`；若你在其他脚本中自行调用 scipy 与 torch，请手动
  `import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`（需在导入 torch 前）。
- **FID 数值不稳定**：FID 需要足够的样本量（特征维度为 2048）。默认
  `n_eval_samples=256` 是快速评估的折中，结果偏有偏；建议评估最终模型时调大到
  1000 以上以获得更可靠的结果（可通过 `python main.py evaluate --n 1000`）。
- **找不到数据**：请确认已按第 2 节下载并解压，且 CSV 位于 `data/FETAL_PLANES_ZENODO/`。
  本项目已处理 CSV 中 `Image_name` 不含 `.png` 后缀、列名带尾随空格等细节。
- **显存不足 (OOM)**：调小 `batch_size` / `img_size`，或降低 `base_channels`。

## 10. 参考

- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, 2021.
- Song et al., *Denoising Diffusion Implicit Models*, 2021.
- Burgos-Artizzu et al., *Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes*, Sci Rep 2020.（数据集论文）

## 11. 作者与版权

- **作者**：肖雄涛
- **版权**：Copyright © 2026 肖雄涛，保留所有权利。
- **用途许可**：本项目（含代码、训练脚本与结果）仅供**学习、教学、科研等非商业用途**使用，**禁止用于任何商业目的**；未经作者书面许可，不得将本项目用于商业用途，或进行销售、再分发、商业部署。