"""胎儿多切面超声图像生成项目 (FETAL_PLANES_DB)。

包含 CycleGAN 与 Diffusion (DDPM) 两种生成模型的搭建、训练、评估与可视化。
"""
import os

# Windows 下 PyTorch 与 scipy 会各自加载 OpenMP 运行库（libiomp5md.dll），
# 二者冲突会导致 FID 计算时进程崩溃。提前设置该变量以允许共存（需在 scipy
# 初始化 OpenMP 之前生效）。参见 Intel OpenMP 官方说明。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
