# 骨龄评估迁移实验 — LIM 工作区完整指南

> 版本：v1.1 | 创建日期：2026-06-01 | 适用范围：LIM/ 工作区全部实验
> 本文档是项目执行的重要参考，所有脚本、数据、结果均存放于 LIM/ 目录下。
> LIM/ 与 骨龄/、bone/ 平级（均在项目根目录下，即 骨龄项目/ 目录）。

---

## 目录

1. [项目概述与研究目标](#1-项目概述与研究目标)
2. [目录结构](#2-目录结构)
3. [环境配置](#3-环境配置)
4. [数据准备](#4-数据准备)
5. [Embedding 提取](#5-embedding-提取)
6. [MLP 训练](#6-mlp-训练)
7. [实验矩阵](#7-实验矩阵)
8. [分析与可视化](#8-分析与可视化)
9. [可视化规范](#9-可视化规范)
10. [输出文件规范](#10-输出文件规范)
11. [审查节点](#11-审查节点)
12. [禁止事项](#12-禁止事项)

---

## 1. 项目概述与研究目标

### 1.1 研究背景

本项目基于大规模视觉预训练模型（DINOv2）构建骨龄自动评估框架，
在两个公开数据集（RSNA 2017、RHPE）上系统评估不同 Backbone 规模、
损失函数设计与性别特征融合对预测精度的影响。

**核心研究问题：**
1. 不同规模的 DINOv2（ViT/S、ViT/B、ViT/L、ViT/g）与传统 CNN
   （ResNet-50、EfficientNet-B4）在骨龄评估任务上的性能差异是什么？
2. 临床对齐的加权损失函数（GW-MAE）相比标准 MAE 是否能改善关键
   发育阶段的预测精度？
3. 在 RSNA 上训练的模型能否跨数据集泛化至 RHPE（跨数据集泛化实验）？

### 1.2 数据集概览

| 数据集 | 规模 | 标签单位 | 用途 |
|---|---|---|---|
| RSNA Bone Age 2017 | 12,611 训练 + 200 测试 | 月（12–228） | 主训练/测试集 |
| RHPE | 5,491 训练 + 713 验证 + 80 测试 | 月（12–228） | 跨数据集泛化测试集 |

WeChat/weixin_imgs 数据集：数据来源不清晰，**本项目完全不使用**。

### 1.3 方法框架

```
原始 X 光图像
      │
      ▼
[阶段一] Backbone 特征提取（build_vision_library.py）
      │  支持 6 种 Backbone，分两组脚本：
      │  · build_vision_library.py：DINOv2-ViT/S / ViT/B / ViT/L / ViT/g
      │  · extract_cnn_embeddings.py（另建）：ResNet-50 / EfficientNet-B4
      │
      ▼
[阶段二] MLP 回归头训练（train_mlp_from_embeddings.py）
      │  消融变量：
      │  · 损失函数：MAE / GW-MAE / MAE+GW-MAE
      │  · 性别特征：有 / 无
      │
      ▼
[阶段三] 跨数据集泛化测试
      │  在 RSNA 上训练的最优模型 → 直接在 RHPE 上推理
      │
      ▼
[阶段四] 分析与可视化
         · 分年龄段误差热力图
         · 分性别误差分析
         · GW-MAE 权重函数可视化
         · Backbone 性能对比图
```

---

## 2. 目录结构

**所有内容必须存放于 LIM/ 目录下，禁止在原始位置（bone/）写入任何新文件。**
LIM/ 与 骨龄/ （代码仓库根目录）、bone/（数据目录）平级。

```
LIM/
│
├── code/                          ← 所有脚本（从 bone/code/ 迁移并修改）
│   ├── build_vision_library.py    ← 迁移自 bone/code/vision/build_vision_library.py
│   ├── train_mlp.py               ← 迁移自 bone/code/train/train_mlp_from_embeddings.py
│   ├── infer_rhpe.py              ← 新建：RHPE 跨数据集推理脚本
│   ├── analyze_results.py         ← 新建：结果分析与可视化主脚本
│   └── utils/
│       ├── gw_mae.py              ← GW-MAE 损失函数实现
│       ├── vis_style.py           ← 可视化样式配置（见第 9 节）
│       └── data_utils.py          ← 数据加载工具函数
│
├── embeddings/                    ← 所有 Backbone 提取的 embedding
│   ├── rsna/
│   │   ├── resnet50/
│   │   │   ├── embeddings/        ← *.npy 文件，每张图一个
│   │   │   └── metadata/
│   │   │       └── rsna_metadata.csv
│   │   ├── efficientnet_b4/
│   │   │   ├── embeddings/
│   │   │   └── metadata/
│   │   ├── dinov2_vits/
│   │   │   ├── embeddings/
│   │   │   └── metadata/
│   │   ├── dinov2_vitb/
│   │   │   ├── embeddings/
│   │   │   └── metadata/
│   │   ├── dinov2_vitl/
│   │   │   ├── embeddings/
│   │   │   └── metadata/
│   │   └── dinov2_vitg/
│   │       ├── embeddings/
│   │       └── metadata/
│   └── rhpe/
│       ├── resnet50/
│       │   ├── embeddings/
│       │   └── metadata/
│       ├── efficientnet_b4/
│       │   ├── embeddings/
│       │   └── metadata/
│       ├── dinov2_vits/
│       │   ├── embeddings/
│       │   └── metadata/
│       ├── dinov2_vitb/
│       │   ├── embeddings/
│       │   └── metadata/
│       ├── dinov2_vitl/
│       │   ├── embeddings/
│       │   └── metadata/
│       └── dinov2_vitg/
│           ├── embeddings/
│           └── metadata/
│
├── experiments/                   ← 所有训练实验结果
│   ├── E01_resnet50_mae_nogender/
│   ├── E02_resnet50_mae_gender/
│   ├── E03_resnet50_gwmae_gender/
│   ├── E04_resnet50_combined_gender/
│   ├── E05_effb4_mae_nogender/
│   ├── E06_effb4_mae_gender/
│   ├── E07_effb4_gwmae_gender/
│   ├── E08_effb4_combined_gender/
│   ├── E09_vits_mae_nogender/
│   ├── E10_vits_mae_gender/
│   ├── E11_vits_gwmae_gender/
│   ├── E12_vits_combined_gender/
│   ├── E13_vitb_mae_nogender/
│   ├── E14_vitb_mae_gender/
│   ├── E15_vitb_gwmae_gender/
│   ├── E16_vitb_combined_gender/
│   ├── E17_vitl_mae_nogender/
│   ├── E18_vitl_mae_gender/
│   ├── E19_vitl_gwmae_gender/
│   ├── E20_vitl_combined_gender/
│   ├── E21_vitg_mae_nogender/
│   ├── E22_vitg_mae_gender/
│   ├── E23_vitg_gwmae_gender/
│   └── E24_vitg_combined_gender/
│   （每个实验目录内容见第 10 节）
│
├── generalization/                ← 跨数据集泛化实验结果
│   ├── best_model_on_rhpe/        ← 最优 RSNA 模型在 RHPE 上的推理结果
│   └── summary_generalization.csv
│
└── figures/                       ← 所有输出图表
    ├── backbone_comparison.png
    ├── loss_ablation.png
    ├── gender_analysis.png
    ├── age_error_heatmap.png
    ├── gw_mae_weight_curve.png
    ├── generalization_comparison.png
    └── error_distribution_by_age_group.png
```

---

## 3. 环境配置

使用 `boneage` conda 环境。

### 3.1 验证基础依赖

```powershell
conda run -n boneage python -c "
import torch
import numpy
import pandas
import sklearn
import tqdm
import timm
import PIL
print('all ok')
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
"
```

### 3.2 安装额外依赖（如缺失）

```powershell
conda run -n boneage pip install timm efficientnet_pytorch matplotlib seaborn scipy
```

### 3.3 路径常量（所有脚本顶部统一定义）

所有脚本必须在顶部定义以下路径常量，**不得硬编码绝对路径到其他位置**。

LIM/ 与 骨龄/、bone/ 平级，因此 `LIM_ROOT/../骨龄/bone/` 即为数据目录。
所有数据路径统一通过 `LIM_ROOT/../骨龄/bone/...` 引用。

```python
import os

# ── 项目根目录（相对于脚本位置自动推导，无需修改）──
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # LIM/code/utils/
LIM_ROOT     = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))   # LIM/
DATA_ROOT    = os.path.join(LIM_ROOT, "..", "骨龄", "bone")        # 骨龄/bone/

# ── 原始数据（只读，禁止写入）──
RSNA_IMG_DIR = os.path.join(DATA_ROOT, "boneage-training-dataset",
                             "boneage-training-dataset")
RSNA_TRAIN_CSV = os.path.join(DATA_ROOT, "boneage-training-dataset.csv")
RSNA_TEST_CSV  = os.path.join(DATA_ROOT, "boneage-test-dataset.csv")

RHPE_TRAIN_IMG = os.path.join(DATA_ROOT, "RHPE_train", "images")
RHPE_VAL_IMG   = os.path.join(DATA_ROOT, "RHPE_val",   "images")
RHPE_TEST_IMG  = os.path.join(DATA_ROOT, "RHPE_test",  "images")
RHPE_TRAIN_CSV = os.path.join(DATA_ROOT, "RHPE_Annotations",
                               "RHPE_Annotations", "BONEAGE", "boneage_train.csv")
RHPE_VAL_CSV   = os.path.join(DATA_ROOT, "RHPE_Annotations",
                               "RHPE_Annotations", "BONEAGE", "boneage_val.csv")

# ── LIM 工作区输出目录（可写）──
EMB_ROOT     = os.path.join(LIM_ROOT, "embeddings")
EXP_ROOT     = os.path.join(LIM_ROOT, "experiments")
GEN_ROOT     = os.path.join(LIM_ROOT, "generalization")
FIG_ROOT     = os.path.join(LIM_ROOT, "figures")
```

---

## 4. 数据准备

### 4.1 RSNA 数据集

**图片路径规则：**
```
骨龄/bone/boneage-training-dataset/boneage-training-dataset/{id}.png
```
ID 范围：10000–22611，共 12,611 张。

**标签 CSV（boneage-training-dataset.csv）列说明：**

| 列名 | 类型 | 说明 |
|---|---|---|
| id | int | 图片 ID，对应文件名（无需补零） |
| boneage | int | 骨龄（月），范围 12–228 |
| male | bool | True=男，False=女 |

**注意：** RSNA 测试集（boneage-test-dataset.csv）共 200 张，
无 boneage 标签，仅用于最终推理展示，不参与训练/验证/测试分割。

### 4.2 RHPE 数据集

**图片路径规则：**
```
骨龄/bone/RHPE_train/images/{ID:05d}.png   （ID 从 1 开始，补零至 5 位）
骨龄/bone/RHPE_val/images/{ID:05d}.png
骨龄/bone/RHPE_test/images/{ID:05d}.png
```

**标签 CSV 列说明（boneage_train.csv / boneage_val.csv）：**

| 列名 | 类型 | 说明 |
|---|---|---|
| ID | int | 病例 ID，对应图片文件名 |
| Male | bool | True=男，False=女 |
| Boneage | int | 骨龄（月），范围 12–228 |
| Chronological | int | 实际日历年龄（月） |

**重要说明：**
- 本项目预测目标为 `Boneage`（骨龄），不是 `Chronological`（日历年龄）。
- RHPE 测试集（RHPE_test/）无对应标注 CSV（boneage_test.csv 不存在），
  跨数据集泛化测试使用 RHPE 验证集（boneage_val.csv，713 条）。
- RHPE 训练集与验证集合并后共 6,204 条，
  在跨数据集泛化实验中**全部作为推理目标，不参与训练**。

### 4.3 数据完整性检查

在提取 embedding 前，运行以下检查脚本确认图片与标签一一对应：

```python
# LIM/code/utils/data_utils.py 中的 check_dataset_integrity 函数
import os, pandas as pd

def check_dataset_integrity(csv_path, img_dir, id_col, id_fmt="{}.png"):
    df = pd.read_csv(csv_path)
    missing = []
    for _, row in df.iterrows():
        fname = id_fmt.format(row[id_col])
        fpath = os.path.join(img_dir, fname)
        if not os.path.exists(fpath):
            missing.append(fpath)
    print(f"Total: {len(df)}, Missing: {len(missing)}")
    if missing:
        for p in missing[:10]:
            print("  MISSING:", p)
    return len(missing) == 0

# RSNA
check_dataset_integrity(RSNA_TRAIN_CSV, RSNA_IMG_DIR, "id", "{}.png")

# RHPE（ID 补零至 5 位）
check_dataset_integrity(RHPE_TRAIN_CSV, RHPE_TRAIN_IMG, "ID", "{:05d}.png")
check_dataset_integrity(RHPE_VAL_CSV,   RHPE_VAL_IMG,   "ID", "{:05d}.png")
```

---

## 5. Embedding 提取

### 5.0 服务器工作流（仅 DINOv2-ViT/B、ViT/L、ViT/g 需要）

#### 5.0.1 步骤在哪里运行

| 阶段 | 本地 CPU | 服务器 | 说明 |
|---|---|---|---|
| 数据完整性检查 | ✅ | — | 纯文件 IO |
| Embedding 提取（ResNet-50 / EfficientNet-B4） | ✅（需先实现 extract_cnn_embeddings.py） | — | 本地约 20–30 min（实现后） |
| Embedding 提取（DINOv2-ViT/S） | ✅ | — | 本地约 1–2 h，慢但可接受 |
| Embedding 提取（DINOv2-ViT/B） | ⚠️ 极慢 | ✅ 推荐 | 服务器约 30–60 min |
| Embedding 提取（DINOv2-ViT/L） | ❌ | ✅ | 服务器约 45–90 min |
| Embedding 提取（DINOv2-ViT/g） | ❌ | ✅ | 服务器约 75–150 min |
| MLP 训练（全部 24 个实验） | ✅ | — | embedding 就绪后，本地 5 min/实验 |
| 分析与可视化 | ✅ | — | 纯 numpy/matplotlib |

**服务器只需要使用一次，集中提取 ViT/B、ViT/L、ViT/g 的 embedding，
完成后下载至本地，后续所有工作在本地完成。**

#### 5.0.2 上传到服务器

在本地完成脚本开发与调试（用 ResNet-50 小批量验证逻辑正确）后，
将以下内容上传至服务器：

```
需要上传的内容：
├── LIM/code/build_vision_library.py   ← 提取脚本
├── LIM/code/utils/data_utils.py       ← 工具函数
├── 骨龄/bone/boneage-training-dataset/ ← RSNA 图片（若服务器尚无）
├── 骨龄/bone/boneage-training-dataset.csv
├── 骨龄/bone/RHPE_train/              ← RHPE 图片（若服务器尚无）
├── 骨龄/bone/RHPE_val/
├── 骨龄/bone/RHPE_Annotations/        ← RHPE 标签
└── LIM/embeddings/                    ← 空目录结构（提前建好）
```

推荐使用 scp 或 rsync 上传：

```bash
# 上传脚本（轻量，每次修改后重传）
scp LIM/code/build_vision_library.py user@server:/path/to/LIM/code/

# 上传数据（仅首次，数据量大，建议 rsync 断点续传）
rsync -avz --progress 骨龄/bone/boneage-training-dataset/ \
  user@server:/path/to/骨龄/bone/boneage-training-dataset/
```

#### 5.0.3 服务器上挂后台运行

SSH 连接后，使用 nohup 挂后台，防止断线中断：

```bash
# 在服务器项目根目录下执行
mkdir -p LIM/logs

# ViT/B — RSNA
nohup conda run -n boneage python LIM/code/build_vision_library.py \
  --dataset rsna \
  --output-dir LIM/embeddings/rsna/dinov2_vitb \
  --batch-size 16 \
  > LIM/logs/vitb_rsna.log 2>&1 &
echo "ViT/B RSNA PID: $!"

# ViT/L — RSNA
nohup conda run -n boneage python LIM/code/build_vision_library.py \
  --dataset rsna \
  --output-dir LIM/embeddings/rsna/dinov2_vitl \
  --batch-size 8 \
  > LIM/logs/vitl_rsna.log 2>&1 &
echo "ViT/L RSNA PID: $!"

# ViT/g — RSNA（最慢，单独跑）
nohup conda run -n boneage python LIM/code/build_vision_library.py \
  --dataset rsna \
  --output-dir LIM/embeddings/rsna/dinov2_vitg \
  --batch-size 8 \
  > LIM/logs/vitg_rsna.log 2>&1 &
echo "ViT/g RSNA PID: $!"
```

查看进度：

```bash
tail -f LIM/logs/vitg_rsna.log
```

注意：ViT/L 和 ViT/g 内存占用较大，**不建议同时运行**，
按顺序依次启动，确认上一个完成后再启动下一个。

#### 5.0.4 从服务器下载 Embedding

提取完成后，将 embedding 打包下载至本地 LIM/embeddings/ 对应目录：

```bash
# 在服务器上打包（减少传输文件数量）
cd LIM/embeddings/rsna
tar -czf dinov2_vitb.tar.gz dinov2_vitb/
tar -czf dinov2_vitl.tar.gz dinov2_vitl/
tar -czf dinov2_vitg.tar.gz dinov2_vitg/

# 同理打包 RHPE
cd 骨龄/LIM/embeddings/rhpe
tar -czf dinov2_vitb.tar.gz dinov2_vitb/
tar -czf dinov2_vitl.tar.gz dinov2_vitl/
tar -czf dinov2_vitg.tar.gz dinov2_vitg/
```

在本地接收：

```bash
scp user@server:/path/to/LIM/embeddings/rsna/dinov2_vitg.tar.gz \
    LIM/embeddings/rsna/

# 解压
tar -xzf LIM/embeddings/rsna/dinov2_vitg.tar.gz -C LIM/embeddings/rsna/
```

下载完成后，服务器使命结束，后续所有 MLP 训练、分析、可视化均在本地完成。

### 5.1 脚本说明

`LIM/code/build_vision_library.py` 迁移自 `bone/code/vision/build_vision_library.py`，
修改内容：
1. 路径常量替换为第 3.3 节定义的 LIM_ROOT 体系。
2. 输出目录改为 `LIM/embeddings/{dataset}/{backbone}/`。
3. `--weights` 参数默认值改为 `None`（不使用学长微调权重，全部使用原始预训练权重）。

**注意：`build_vision_library.py` 仅支持 DINOv2 系列 Backbone。**
ResNet-50 和 EfficientNet-B4 需要额外新建 `LIM/code/extract_cnn_embeddings.py`，
基于 `timm` 库实现。该脚本的接口设计应与 `build_vision_library.py` 保持一致
（相同的命令行参数风格、相同的输出目录结构、相同的 metadata 格式），
仅 Backbone 加载和预处理部分不同。详见第 5.2 节。

### 5.2 支持的 Backbone 及特征维度

Backbone 分两组，使用不同的提取脚本：

**A. DINOv2 系列 — 使用 `build_vision_library.py`（已就绪）**

| Backbone 标识 | 模型来源 | 特征维度 | 预训练权重 |
|---|---|---|---|
| `dinov2_vits` | torch.hub / HuggingFace | 384 | DINOv2 |
| `dinov2_vitb` | torch.hub / HuggingFace | 768 | DINOv2 |
| `dinov2_vitl` | torch.hub / HuggingFace | 1024 | DINOv2 |
| `dinov2_vitg` | torch.hub / HuggingFace | 1536 | DINOv2 |

**B. CNN 系列 — 使用 `extract_cnn_embeddings.py`（待实现，基于 timm）**

| Backbone 标识 | 模型来源 | 特征维度 | 预训练权重 |
|---|---|---|---|
| `resnet50` | timm | 2048 | ImageNet-1k |
| `efficientnet_b4` | timm | 1792 | ImageNet-1k |

**所有 Backbone 均使用原始公开预训练权重，不使用任何经过骨龄任务微调的权重。**

### 5.3 执行命令

**DINOv2 系列 — 使用 `build_vision_library.py`**

在项目根目录（骨龄/）下执行，`--cpu` 强制 CPU 推理（服务器无 GPU 时使用）。
`build_vision_library.py` 自动从预训练权重加载 DINOv2 模型，无需额外指定。

**RSNA — DINOv2-ViT/g（示例，其他 DINOv2 版本同理替换 --output-dir 中的路径名）：**

```powershell
conda run -n boneage python LIM/code/build_vision_library.py ^
  --dataset rsna ^
  --output-dir LIM/embeddings/rsna/dinov2_vitg ^
  --batch-size 16 ^
  --cpu
```

**RSNA — DINOv2-ViT/S（batch-size 可较大）：**

```powershell
conda run -n boneage python LIM/code/build_vision_library.py ^
  --dataset rsna ^
  --output-dir LIM/embeddings/rsna/dinov2_vits ^
  --batch-size 32 ^
  --cpu
```

**RSNA — DINOv2-ViT/B：**

```powershell
conda run -n boneage python LIM/code/build_vision_library.py ^
  --dataset rsna ^
  --output-dir LIM/embeddings/rsna/dinov2_vitb ^
  --batch-size 16 ^
  --cpu
```

**RSNA — DINOv2-ViT/L（内存占用大，batch-size 减小）：**

```powershell
conda run -n boneage python LIM/code/build_vision_library.py ^
  --dataset rsna ^
  --output-dir LIM/embeddings/rsna/dinov2_vitl ^
  --batch-size 8 ^
  --cpu
```

**RHPE — 所有 DINOv2 Backbone（合并 train+val 一次性提取，以 ViT/g 为例）：**

```powershell
conda run -n boneage python LIM/code/build_vision_library.py ^
  --dataset rhpe ^
  --output-dir LIM/embeddings/rhpe/dinov2_vitg ^
  --batch-size 16 ^
  --cpu
```

（其余 DINOv2 版本同理替换 --output-dir 中的路径名即可）

**CNN 系列 — 使用 `extract_cnn_embeddings.py`**

CNN embedding 提取脚本待实现。设计原则：
- 接口风格与 `build_vision_library.py` 保持一致
- 通过 `timm.create_model('resnet50', pretrained=True)` 加载
- 移除最后的分类头，取 global avg pooling 后的特征向量
- 输出目录结构、metadata 格式与 DINOv2 系列一致，便于后续 MLP 训练共用 `train_mlp.py`

### 5.4 输出文件格式

每次提取完成后，`output-dir` 下生成：

```
LIM/embeddings/rsna/dinov2_vitg/
├── embeddings/
│   ├── 10000.npy      ← shape: (feat_dim,)，float32
│   ├── 10001.npy
│   └── ...
└── metadata/
    └── rsna_metadata.csv
```

**rsna_metadata.csv 列：**

| 列名 | 说明 |
|---|---|
| case_id | 图片 ID |
| gender | M / F |
| true_age | 骨龄（月） |
| image_path | 原始图片绝对路径 |
| embedding_file | 对应 .npy 文件绝对路径 |

**RHPE metadata 同理，列名一致。**

### 5.5 断点续传

`build_vision_library.py` 支持断点续传：若 `embeddings/` 目录下已存在对应 `.npy`
文件，则跳过该样本。中断后重新运行同一命令即可继续。

### 5.6 预估时间（CPU，服务器）

| Backbone | RSNA 12,611 张 | RHPE 6,204 张 |
|---|---|---|
| ResNet-50 | 待实测（extract_cnn_embeddings.py 实现后补充） | — |
| EfficientNet-B4 | 待实测（extract_cnn_embeddings.py 实现后补充） | — |
| DINOv2-ViT/S | ~40 min | ~20 min |
| DINOv2-ViT/B | ~60 min | ~30 min |
| DINOv2-ViT/L | ~90 min | ~45 min |
| DINOv2-ViT/g | ~150 min | ~75 min |

以上为保守估计，实际取决于服务器 CPU 核心数。

---

## 6. MLP 训练

### 6.1 脚本说明

`LIM/code/train_mlp.py` 迁移自 `bone/code/train/train_mlp_from_embeddings.py`，
修改内容：
1. 路径常量替换为 LIM_ROOT 体系。
2. 输出目录改为 `LIM/experiments/{exp_id}/`。
3. `--feat-dim` 参数默认值为 1536（适用于 DINOv2-ViT/g），
   使用其他 backbone 时必须手动指定对应维度（见第 5.2 节维度表）。
4. 损失函数通过 `--alpha-mae` 和 `--beta-gw-mae` 参数控制（见 6.3 节）。
5. 新增 `--no-gender` 参数用于消融性别特征。

### 6.2 MLP 结构

MLP 模型定义位于 `mlp_age_service.py` 的 `BoneAgeEmbeddingMLP` 类中，结构如下：

```
输入层：feat_dim（+ 1，若使用性别特征）
  → LayerNorm(feat_dim+1)                   ← 层归一化
  → Linear(feat_dim+1, 512) → GELU → Dropout(0.2)
  → Linear(512, 128)        → GELU → Dropout(0.2)
  → Linear(128, 1)          → Softplus() → 输出骨龄（月，正值约束）
```

训练配置：
- Optimizer：AdamW，lr=1e-3，weight_decay=1e-4
- Scheduler：CosineAnnealingLR，T_max=80
- Gradient Clipping：max_norm=1.0
- Early Stopping：patience=15（监控 val MAE）
- Epochs：最多 80
- Batch Size：64
- 数据分割：train/val/test = 80/10/10，按 gender+age_bin 分层

### 6.3 损失函数实现

**设计说明：** 本文档定义的 GW-MAE 为连续高斯加权函数，是对现有代码中分段常数版本（`eval_utils.py` `GrowthWindowWeights`）的重新设计。两者临床动机相同（对关键发育阶段赋予更高权重），但实现不同；LIM 项目使用本文档定义的版本。

在 `LIM/code/utils/gw_mae.py` 中实现：

```python
import torch
import torch.nn as nn
import numpy as np


class GrowthWeightedMAELoss(nn.Module):
    """
    GW-MAE：生长加权平均绝对误差损失函数。

    临床动机：骨龄评估误差在不同发育阶段的临床代价不同。
    青春期前（60–180 月）是生长激素治疗的关键窗口期，
    该阶段的预测误差应受到更严格的惩罚。

    权重函数 w(age) 基于高斯混合，峰值位于 120 月（10 岁），
    对应青春期前发育加速阶段。

    参数：
        mu    : 高斯峰值位置（月），默认 120
        sigma : 高斯宽度（月），默认 36
        w_min : 最小权重（非关键年龄段的基础惩罚），默认 0.5
        w_max : 最大权重（峰值年龄段的惩罚倍数），默认 2.0
    """

    def __init__(self, mu=120.0, sigma=36.0, w_min=0.5, w_max=2.0):
        super().__init__()
        self.mu    = mu
        self.sigma = sigma
        self.w_min = w_min
        self.w_max = w_max

    def weight(self, age_months: torch.Tensor) -> torch.Tensor:
        """计算每个样本的权重，age_months 为真实骨龄（月）。"""
        gaussian = torch.exp(
            -0.5 * ((age_months - self.mu) / self.sigma) ** 2
        )
        return self.w_min + (self.w_max - self.w_min) * gaussian

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight(target)
        return (w * torch.abs(pred - target)).mean()


class CombinedLoss(nn.Module):
    """
    MAE + alpha * GW-MAE 联合损失。

    参数：
        alpha_mae   : MAE 项系数，默认 1.0
        beta_gw_mae : GW-MAE 项系数，默认 1.0
        gw_kwargs   : 传递给 GrowthWeightedMAELoss 的参数
    """

    def __init__(self, alpha_mae=1.0, beta_gw_mae=1.0, **gw_kwargs):
        super().__init__()
        self.alpha_mae   = alpha_mae
        self.beta_gw_mae = beta_gw_mae
        self.mae_loss    = nn.L1Loss()
        self.gw_mae_loss = GrowthWeightedMAELoss(**gw_kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.alpha_mae   * self.mae_loss(pred, target) +
                self.beta_gw_mae * self.gw_mae_loss(pred, target))
```

### 6.4 执行命令

**基本格式：**

```powershell
conda run -n boneage python LIM/code/train_mlp.py ^
  --embeddings-dir LIM/embeddings/rsna/{backbone}/embeddings ^
  --metadata-csv   LIM/embeddings/rsna/{backbone}/metadata/rsna_metadata.csv ^
  --labels-csv     骨龄/bone/boneage-training-dataset.csv ^
  --output-dir     LIM/experiments/{exp_id} ^
  --alpha-mae 1.0 --beta-gw-mae 0.0 ^
  [--no-gender] ^
  --epochs 80 ^
  --batch-size 64 ^
  --lr 1e-3 ^
  --seed 42 ^
  --cpu
```

损失模式通过 `--alpha-mae` 和 `--beta-gw-mae` 组合实现：
- 纯 MAE：`--alpha-mae 1.0 --beta-gw-mae 0.0`
- 纯 GW-MAE：`--alpha-mae 0.0 --beta-gw-mae 1.0`
- 联合损失：`--alpha-mae 1.0 --beta-gw-mae 1.0`

**示例 — E01（ResNet-50，MAE，无性别特征）：**

```powershell
conda run -n boneage python LIM/code/train_mlp.py ^
  --embeddings-dir LIM/embeddings/rsna/resnet50/embeddings ^
  --metadata-csv   LIM/embeddings/rsna/resnet50/metadata/rsna_metadata.csv ^
  --labels-csv     骨龄/bone/boneage-training-dataset.csv ^
  --output-dir     LIM/experiments/E01_resnet50_mae_nogender ^
  --alpha-mae 1.0 --beta-gw-mae 0.0 ^
  --no-gender ^
  --cpu
```

**示例 — E23（DINOv2-ViT/g，GW-MAE，有性别特征）：**

```powershell
conda run -n boneage python LIM/code/train_mlp.py ^
  --embeddings-dir LIM/embeddings/rsna/dinov2_vitg/embeddings ^
  --metadata-csv   LIM/embeddings/rsna/dinov2_vitg/metadata/rsna_metadata.csv ^
  --labels-csv     骨龄/bone/boneage-training-dataset.csv ^
  --output-dir     LIM/experiments/E23_vitg_gwmae_gender ^
  --alpha-mae 0.0 --beta-gw-mae 1.0 ^
  --cpu
```

---

## 7. 实验矩阵

### 7.1 完整实验列表（24 个实验）

| 实验 ID | Backbone | 损失函数 | 性别特征 | 核心目的 |
|---|---|---|---|---|
| E01 | ResNet-50 | MAE | 无 | CNN baseline（无性别） |
| E02 | ResNet-50 | MAE | 有 | CNN baseline（有性别） |
| E03 | ResNet-50 | GW-MAE | 有 | CNN + 临床损失 |
| E04 | ResNet-50 | MAE+GW-MAE | 有 | CNN + 联合损失 |
| E05 | EfficientNet-B4 | MAE | 无 | 轻量 CNN baseline |
| E06 | EfficientNet-B4 | MAE | 有 | 轻量 CNN + 性别 |
| E07 | EfficientNet-B4 | GW-MAE | 有 | 轻量 CNN + 临床损失 |
| E08 | EfficientNet-B4 | MAE+GW-MAE | 有 | 轻量 CNN + 联合损失 |
| E09 | DINOv2-ViT/S | MAE | 无 | 小 Transformer baseline |
| E10 | DINOv2-ViT/S | MAE | 有 | 小 Transformer + 性别 |
| E11 | DINOv2-ViT/S | GW-MAE | 有 | 小 Transformer + 临床损失 |
| E12 | DINOv2-ViT/S | MAE+GW-MAE | 有 | 小 Transformer + 联合损失 |
| E13 | DINOv2-ViT/B | MAE | 无 | 中 Transformer baseline |
| E14 | DINOv2-ViT/B | MAE | 有 | 中 Transformer + 性别 |
| E15 | DINOv2-ViT/B | GW-MAE | 有 | 中 Transformer + 临床损失 |
| E16 | DINOv2-ViT/B | MAE+GW-MAE | 有 | 中 Transformer + 联合损失 |
| E17 | DINOv2-ViT/L | MAE | 无 | 大 Transformer baseline |
| E18 | DINOv2-ViT/L | MAE | 有 | 大 Transformer + 性别 |
| E19 | DINOv2-ViT/L | GW-MAE | 有 | 大 Transformer + 临床损失 |
| E20 | DINOv2-ViT/L | MAE+GW-MAE | 有 | 大 Transformer + 联合损失 |
| E21 | DINOv2-ViT/g | MAE | 无 | 最大 Transformer baseline |
| E22 | DINOv2-ViT/g | MAE | 有 | 最大 Transformer + 性别 |
| E23 | DINOv2-ViT/g | GW-MAE | 有 | 最大 Transformer + 临床损失 |
| E24 | DINOv2-ViT/g | MAE+GW-MAE | 有 | 最大 Transformer + 联合损失（预期最优） |

### 7.2 跨数据集泛化实验

从 E01–E24 中选取 RSNA 测试集 MAE 最低的模型（预期为 E24），
直接在 RHPE 验证集（713 条）上推理，不做任何 RHPE 数据的训练。

```powershell
conda run -n boneage python LIM/code/infer_rhpe.py ^
  --model-path     LIM/experiments/E24_vitg_combined_gender/best.pt ^
  --embeddings-dir LIM/embeddings/rhpe/dinov2_vitg/embeddings ^
  --metadata-csv   LIM/embeddings/rhpe/dinov2_vitg/metadata/rhpe_metadata.csv ^
  --labels-csv     骨龄/bone/RHPE_Annotations/RHPE_Annotations/BONEAGE/boneage_val.csv ^
  --output-dir     LIM/generalization/best_model_on_rhpe
```

### 7.3 评估指标

每个实验输出以下指标：

| 指标 | 说明 |
|---|---|
| MAE | 全集平均绝对误差（月） |
| MAE(M) | 男性子集 MAE |
| MAE(F) | 女性子集 MAE |
| MAE_0_60 | 0–60 月年龄段 MAE |
| MAE_60_120 | 60–120 月年龄段 MAE |
| MAE_120_180 | 120–180 月年龄段 MAE |
| MAE_180_228 | 180–228 月年龄段 MAE |
| GW-MAE | 加权 MAE（临床对齐指标） |
| RMSE | 均方根误差 |
| R² | 决定系数 |

---

## 8. 分析与可视化

所有分析脚本位于 `LIM/code/analyze_results.py`，
图表输出至 `LIM/figures/`，格式为 PNG（300 DPI）。

### 8.1 图表清单

**图 1：Backbone 性能对比（backbone_comparison.png）**

- 类型：分组条形图（grouped bar chart）
- X 轴：6 个 Backbone
- Y 轴：MAE（月）
- 分组：有性别 vs 无性别（仅 MAE 损失，E01–E02、E05–E06、E09–E10、E13–E14、E17–E18、E21–E22）
- 附加：在每个 Backbone 组上方标注特征维度
- 尺寸：(12, 5)

**图 2：损失函数消融（loss_ablation.png）**

- 类型：分组条形图
- X 轴：6 个 Backbone
- Y 轴：MAE（月）
- 分组：MAE / GW-MAE / MAE+GW-MAE（均含性别特征，E02、E03、E04 等）
- 尺寸：(12, 5)

**图 3：性别分层误差（gender_analysis.png）**

- 类型：1×2 子图
  - 左：MAE(M) vs MAE(F) 各 Backbone 对比条形图
  - 右：散点图（真实骨龄 vs 预测骨龄），男女用不同颜色，对角线为理想预测线
- 尺寸：(12, 5)

**图 4：年龄段误差热力图（age_error_heatmap.png）**

- 类型：热力图（heatmap）
- 行：6 个 Backbone（最优损失函数配置）
- 列：4 个年龄段（0–60 / 60–120 / 120–180 / 180–228 月）
- 颜色：MAE 值，色图使用 Blues（值越深误差越大）
- 在每个格子内标注 MAE 数值
- 尺寸：(10, 6)

**图 5：GW-MAE 权重函数曲线（gw_mae_weight_curve.png）**

- 类型：折线图
- X 轴：骨龄（月），范围 0–240
- Y 轴：权重 w(age)
- 附加：用浅色背景区域标注 4 个年龄段
- 附加：标注峰值位置（120 月，10 岁）
- 尺寸：(10, 6)

**图 6：跨数据集泛化对比（generalization_comparison.png）**

- 类型：1×2 子图
  - 左：最优模型在 RSNA 测试集 vs RHPE 验证集的 MAE 对比（条形图）
  - 右：RHPE 上的年龄段误差分布（与 RSNA 同格式，便于对比）
- 尺寸：(12, 5)

**图 7：误差分布（error_distribution_by_age_group.png）**

- 类型：2×2 子图，每个子图为一个年龄段的误差分布直方图
- 每个子图叠加 MAE 和 GW-MAE 训练结果的误差分布（两条曲线）
- 尺寸：(10, 8)

---

## 9. 可视化规范

**本节规范强制适用于 LIM/figures/ 下所有图表，
所有绘图脚本必须在生成图像前导入并应用 LIM/code/utils/vis_style.py。**

### 9.1 字体系统

优先级链：Times New Roman → Georgia → DejaVu Serif → serif

| 场景 | 字体风格 | 字号 |
|---|---|---|
| 图题 | Times New Roman，Bold | 14 pt |
| 坐标轴标签 | Times New Roman，变量斜体+单位正体 | 12 pt |
| 刻度标签 | Times New Roman，正体 | 10 pt |
| 图例文字 | Times New Roman，正体 | 11 pt |
| 标注文字 | Times New Roman，正体 | 9 pt |
| 子图标签 (A)(B) | Times New Roman，Bold | 12 pt |

坐标轴标签格式示例：
```python
ax.set_xlabel(r'$\it{Age}$ (months)', fontsize=12)
ax.set_ylabel(r'$\it{MAE}$ (months)', fontsize=12)
```

### 9.2 配色方案（语义固定，不得随意更换）

```python
COLORS = {
    # Backbone 系列（按规模递增）
    "resnet50":        "#457B9D",   # 钢蓝
    "efficientnet_b4": "#2A9D8F",   # 青绿
    "dinov2_vits":     "#E9C46A",   # 琥珀黄
    "dinov2_vitb":     "#E64B35",   # 朱红
    "dinov2_vitl":     "#6A4C93",   # 深紫
    "dinov2_vitg":     "#333333",   # 深灰（最大模型，最显眼）

    # 损失函数系列
    "mae":             "#457B9D",   # 钢蓝（baseline）
    "gwmae":           "#E64B35",   # 朱红（临床损失）
    "combined":        "#2A9D8F",   # 青绿（联合损失）

    # 性别
    "male":            "#457B9D",   # 钢蓝
    "female":          "#E64B35",   # 朱红

    # 数据集
    "rsna":            "#457B9D",
    "rhpe":            "#E64B35",

    # 年龄段背景
    "age_0_60":        "#FFF5F5",
    "age_60_120":      "#F0F7FF",
    "age_120_180":     "#F0FFF4",
    "age_180_228":     "#FFFFF0",

    # 通用
    "neutral":         "#D3D3D3",
    "text_primary":    "#333333",
    "text_secondary":  "#666666",
    "axis_line":       "#444444",
    "grid_line":       "#E0E0E0",
}
```

### 9.3 图幅与 DPI

| 图表类型 | 尺寸（英寸） | DPI |
|---|---|---|
| 单图 | (6, 5) | 300 |
| 1×2 子图 | (12, 5) | 300 |
| 2×2 子图 | (10, 8) | 300 |
| 热力图 | (10, 6) | 300 |
| 宽幅对比图 | (12, 5) | 300 |

### 9.4 坐标轴规范

```python
# 统一应用于所有子图
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.0)
    ax.spines[spine].set_color('#444444')

ax.tick_params(axis='both', which='major',
               direction='out', length=4, width=1.0,
               labelsize=10, colors='#333333', pad=4)

# 网格线（仅在需要辅助读数时启用）
ax.grid(True, which='major', linestyle='--', linewidth=0.5,
        color='#E0E0E0', alpha=0.7, zorder=0)
ax.set_axisbelow(True)
```

### 9.5 vis_style.py 全局模板

```python
# LIM/code/utils/vis_style.py
import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_style():
    """在所有绘图脚本的最顶部调用一次。"""
    plt.rcParams.update({
        # 字体
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Georgia",
                               "DejaVu Serif", "serif"],
        "mathtext.fontset":   "stix",
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    11,
        # 坐标轴
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.0,
        "axes.edgecolor":     "#444444",
        # 刻度
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   4,
        "ytick.major.size":   4,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        # 网格
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "grid.color":         "#E0E0E0",
        "grid.alpha":         0.7,
        # 图像
        "figure.dpi":         100,    # 屏幕预览
        "savefig.dpi":        300,    # 正式输出
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })

# 颜色常量（见 9.2 节）
COLORS = { ... }  # 同 9.2 节完整定义
```

### 9.6 禁止使用的色图

禁止使用 `jet`、`rainbow`、`hsv`。
热力图使用 `Blues`，差值图使用 `RdBu_r`，密度图使用 `viridis`。

---

## 10. 输出文件规范

### 10.1 每个实验目录结构

```
LIM/experiments/E24_vitg_combined_gender/
├── best.pt              ← 验证集最优 checkpoint（含模型权重 + 超参数）
├── last.pt              ← 最后一个 epoch 的 checkpoint
├── history.json         ← 每个 epoch 的 train_loss / val_loss / val_mae
├── metrics.json         ← 最终测试集指标（见下方格式）
├── split.json           ← train/val/test 的 case_id 列表
└── predictions.csv      ← 测试集每条样本的预测结果（见下方格式）
```

**metrics.json 格式：**

```json
{
  "experiment_id": "E24_vitg_combined_gender",
  "backbone": "dinov2_vitg",
  "loss": "combined",
  "use_gender": true,
  "test_mae": 5.23,
  "test_mae_male": 5.10,
  "test_mae_female": 5.38,
  "test_mae_0_60": 6.12,
  "test_mae_60_120": 4.89,
  "test_mae_120_180": 5.01,
  "test_mae_180_228": 6.45,
  "test_gw_mae": 4.98,
  "test_rmse": 7.12,
  "test_r2": 0.982,
  "best_epoch": 67,
  "total_epochs": 80
}
```

**predictions.csv 列：**

| 列名 | 说明 |
|---|---|
| case_id | 样本 ID |
| gender | M / F |
| true_age | 真实骨龄（月） |
| pred_age | 预测骨龄（月） |
| abs_error | 绝对误差（月） |
| age_group | 所属年龄段（0_60 / 60_120 / 120_180 / 180_228） |

### 10.2 汇总结果文件

训练全部完成后，运行：

```powershell
conda run -n boneage python LIM/code/analyze_results.py --mode summary
```

生成 `LIM/experiments/summary.csv`，每行一个实验，列为所有指标。

---

## 11. 审查节点

### P1：Embedding 提取质量检查

**触发时机：** 每个 Backbone 的 RSNA embedding 提取完成后。

**检查内容：**
1. 确认 `embeddings/` 目录下文件数量 = 12,611（RSNA）或 6,204（RHPE）。
2. 随机抽取 10 个 `.npy` 文件，确认 shape 和 dtype 符合预期。
3. 检查 `metadata.csv` 的 `true_age` 列无缺失值。

```powershell
conda run -n boneage python -c "
import os, numpy as np, pandas as pd

emb_dir  = 'LIM/embeddings/rsna/dinov2_vitg/embeddings'
meta_csv = 'LIM/embeddings/rsna/dinov2_vitg/metadata/rsna_metadata.csv'

files = [f for f in os.listdir(emb_dir) if f.endswith('.npy')]
print(f'File count: {len(files)}')

import random
for f in random.sample(files, 10):
    arr = np.load(os.path.join(emb_dir, f))
    print(f'  {f}: shape={arr.shape}, dtype={arr.dtype}')

df = pd.read_csv(meta_csv)
print(f'Metadata rows: {len(df)}, null true_age: {df.true_age.isnull().sum()}')
"
```

### P2：训练收敛检查

**触发时机：** 每个实验训练完成后。

**检查内容：**
1. 打开 `history.json`，确认 val_mae 有下降趋势（非单调递增）。
2. 确认 `best_epoch` 不是第 1 或最后 1 个 epoch（若是，说明训练不稳定）。
3. 若 val_mae > 15 月，标记为异常实验，检查数据加载是否正确。

### P3：最终结果审查

**触发时机：** 所有 24 个实验完成后，生成 summary.csv 后。

**检查内容：**
1. 确认 DINOv2 系列 MAE 随模型规模单调下降（ViT/S > ViT/B > ViT/L > ViT/g）。
   若出现反转，检查对应 embedding 提取是否有误。
2. 确认 GW-MAE 训练相比 MAE 训练在 60–180 月年龄段的误差有所降低。
3. 确认跨数据集泛化实验（RHPE）的 MAE 不超过 RSNA 测试集 MAE 的 2 倍。
   若超过，在报告中明确说明数据集分布差异。

---

## 12. 禁止事项

1. **禁止在 骨龄/bone/ 目录下写入任何新文件**（该目录为只读原始数据区）。
2. **禁止使用 骨龄/bone/dinov2/bone_age_predictor_best.pth**（学长微调权重，
   来路不透明，不得用于本项目任何 embedding 提取）。
3. **禁止使用 骨龄/weixin_imgs/ 数据集**（数据来源不清晰）。
4. **禁止硬编码绝对路径**（所有路径通过第 3.3 节的常量体系推导）。
5. **禁止在 RHPE 数据上进行任何形式的训练**（RHPE 仅用于跨数据集泛化推理）。
6. **禁止使用 jet / rainbow / hsv 色图**（见第 9.6 节）。
7. **禁止跳过 P1 审查直接进入训练**（embedding 质量未验证的训练结果不可信）。
