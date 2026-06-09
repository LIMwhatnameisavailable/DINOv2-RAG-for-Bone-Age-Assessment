# LIM 骨龄评估系统性消融实验 — 执行报告

**报告日期**：2026-06-08  
**报告人**：Claude Code（自动执行）

---

## 一、项目概述

**研究目标**：骨龄自动评估系统性消融实验，对比 6 种 Backbone（ResNet-50、EfficientNet-B4、DINOv2 ViT-S/B/L/g）× 3 种损失函数（MAE、GW-MAE、Combined）× 2 种性别模式（有/无）= 24 个实验，并在 RHPE 数据集上进行跨数据集泛化测试。

**数据集**：
- RSNA Bone Age 2017：12,611 张训练图像 + 200 张测试图像（无标签），骨龄范围 12–228 月
- RHPE：6,204 张有标签图像（train+val 合并），仅用于泛化推理，禁止训练

**运行环境**：
- GPU：NVIDIA GeForce RTX 3090 (24 GB)
- 软件：Python 3.10 + PyTorch 2.1.2 + CUDA 11.8
- DINOv2 权重来源：本地预下载（`~/LIM_data/dinov2_weights/`），禁止在线加载
- CNN 权重来源：timm 本地 safetensors（ImageNet-1k 预训练）

---

## 二、执行过程与结果

### 阶段 1：Embedding 提取

**做了什么**：对 6 种 Backbone × 2 个数据集（RSNA、RHPE）逐一提取图像特征向量，保存为 .npy 文件。

**结果**：

| Backbone | 维度 | RSNA 提取数 | RHPE 提取数 |
|----------|:----:|:-----------:|:-----------:|
| ResNet-50 | 2048 | 12,611/12,611 | 6,204/6,204 |
| EfficientNet-B4 | 1792 | 12,611/12,611 | 6,204/6,204 |
| DINOv2 ViT-S | 384 | 12,611/12,611 | 6,204/6,204 |
| DINOv2 ViT-B | 768 | 12,611/12,611 | 6,204/6,204 |
| DINOv2 ViT-L | 1024 | 12,611/12,611 | 6,204/6,204 |
| DINOv2 ViT-g | 1536 | 12,611/12,611 | 6,204/6,204 |

---

### 阶段 2：24 个 MLP 实验训练

**做了什么**：在 12,611 张 RSNA 图像上使用 80/10/10 分层分割，训练 3 层 MLP（输入→512→128→1），AdamW 优化器（lr=1e-3，weight_decay=1e-4），CosineAnnealingLR 调度，early stopping patience=15，最多 80 epoch。

**结果**（按 Test MAE 升序排列）：

| 排名 | 实验ID | Backbone | 损失函数 | 性别 | Test MAE | Test RMSE | Test R² |
|:----:|:-------|:---------|:---------|:----:|:--------:|:---------:|:-------:|
| 1 | E16 | ViT-B | Combined | 有 | 8.24 | 10.71 | 0.935 |
| 2 | E14 | ViT-B | MAE | 有 | 8.24 | 10.70 | 0.935 |
| 3 | E15 | ViT-B | GW-MAE | 有 | 8.33 | 10.85 | 0.933 |
| 4 | E12 | ViT-S | Combined | 有 | 8.49 | 11.10 | 0.929 |
| 5 | E11 | ViT-S | GW-MAE | 有 | 8.55 | 11.00 | 0.931 |
| 6 | E10 | ViT-S | MAE | 有 | 8.64 | 11.19 | 0.928 |
| 7 | E22 | ViT-g | MAE | 有 | 8.64 | 11.37 | 0.925 |
| 8 | E24 | ViT-g | Combined | 有 | 8.69 | 11.39 | 0.925 |
| 9 | E23 | ViT-g | GW-MAE | 有 | 8.73 | 11.44 | 0.925 |
| 10 | E18 | ViT-L | MAE | 有 | 9.16 | 12.05 | 0.916 |
| 11 | E20 | ViT-L | Combined | 有 | 9.26 | 12.13 | 0.915 |
| 12 | E19 | ViT-L | GW-MAE | 有 | 9.51 | 12.62 | 0.908 |
| 13 | E13 | ViT-B | MAE | 无 | 10.38 | 13.25 | 0.899 |
| 14 | E21 | ViT-g | MAE | 无 | 10.66 | 13.50 | 0.896 |
| 15 | E09 | ViT-S | MAE | 无 | 10.89 | 14.01 | 0.887 |
| 16 | E08 | EffNet-B4 | Combined | 有 | 11.32 | 14.59 | 0.878 |
| 17 | E17 | ViT-L | MAE | 无 | 11.33 | 14.69 | 0.876 |
| 18 | E07 | EffNet-B4 | GW-MAE | 有 | 11.39 | 14.73 | 0.876 |
| 19 | E06 | EffNet-B4 | MAE | 有 | 11.60 | 14.98 | 0.872 |
| 20 | E03 | ResNet-50 | GW-MAE | 有 | 13.85 | 17.76 | 0.820 |
| 21 | E05 | EffNet-B4 | MAE | 无 | 13.96 | 17.73 | 0.821 |
| 22 | E04 | ResNet-50 | Combined | 有 | 14.04 | 18.07 | 0.815 |
| 23 | E02 | ResNet-50 | MAE | 有 | 14.10 | 18.16 | 0.812 |
| 24 | E01 | ResNet-50 | MAE | 无 | 16.34 | 20.65 | 0.757 |

**可观测到的数值范围**：
- DINOv2 系列（E09-E24）：Test MAE 范围为 8.24–11.33
- CNN 系列（E01-E08）：Test MAE 范围为 11.32–16.34
- 前三名均为 ViT-B 实验（E16/E14/E15）
- 末四位均为 ResNet-50 实验（E01-E04）

---

### 阶段 3：跨数据集泛化

**做了什么**：每个 Backbone 选取该系列 MAE+性别 实验的最佳 checkpoint（E02/E06/E10/E14/E18/E22），在 RHPE 全量 6,204 张图像上推理。

**结果**：

| Backbone | 来源实验 | RSNA Test MAE | RHPE Val MAE | RHPE RMSE |
|:---------|:--------:|:-------------:|:------------:|:---------:|
| ResNet-50 | E02 | 14.10 | 18.84 | 24.11 |
| EfficientNet-B4 | E06 | 11.60 | 16.66 | 21.46 |
| ViT-S | E10 | 8.64 | 16.82 | 21.28 |
| ViT-B | E14 | 8.24 | 15.03 | 18.86 |
| ViT-L | E18 | 9.16 | 15.23 | 18.99 |
| ViT-g | E22 | 8.64 | 28.21 | 33.03 |

---

### 阶段 4：分析与可视化

**做了什么**：汇总 summary.csv（24 行），生成 7 张图表保存至 `LIM/figures/`。

**结果**：

| 文件名 | 内容 |
|:-------|:-----|
| `backbone_comparison.png` | 6 Backbone × 有/无性别 MAE 对比 |
| `loss_ablation.png` | 6 Backbone × 3 种损失函数 MAE 对比 |
| `gender_analysis.png` | 分性别 MAE + 预测散点图 |
| `age_error_heatmap.png` | 年龄段误差热力图 |
| `gw_mae_weight_curve.png` | GW-MAE 权重函数曲线 |
| `generalization_comparison.png` | RSNA vs RHPE 泛化对比 + 年龄段误差 |
| `error_distribution_by_age_group.png` | 4 年龄段误差分布直方图 |

---

## 三、产出文件清单

| 内容 | 路径 |
|:-----|:-----|
| 实验汇总表 | `LIM/experiments/summary.csv`（24 行） |
| 24 个实验详情 | `LIM/experiments/E01/` ~ `E24/`（各含 best.pt + metrics.json + history.json） |
| RSNA Embedding | `LIM/embeddings/rsna/{backbone}/`（各 12,611 个 .npy） |
| RHPE Embedding | `LIM/embeddings/rhpe/{backbone}/`（各 6,204 个 .npy） |
| 跨数据集泛化结果 | `LIM/generalization/`（6 个 backbone 的 metrics.json） |
| 分析图表 | `LIM/figures/*.png`（7 张） |
| 完整运行日志 | `LIM/logs/overnight_run.log` |
| 异常记录 | `LIM/logs/errors.txt` |

---

## 四、ViT-g RHPE 泛化异常诊断

**现象**：ViT-g RSNA test MAE=8.64，但 RHPE 泛化 MAE=28.21（衰减 +226%），远超其他 backbone（+34%~+95%）。

### 诊断步骤与结果

**Step 1：Embedding 统计对比**

| 指标 | RSNA ViT-g | RHPE ViT-g | RHPE ViT-B（对照） |
|:-----|:----------:|:----------:|:------------------:|
| 维度 | 1536 | 1536 | 768 |
| 均值 | -0.0047 | -0.0039 | -0.0025 |
| 标准差 | 1.2917 | 1.2912 | 1.6904 |
| 均值差异 | — | **0.0007**（阈值0.1，正常） | — |
| 标准差比值 | — | **0.9997**（阈值0.5~2.0，正常） | — |
| NaN / 全零向量 | 0 / 0 | 0 / 0 | 0 / 0 |

→ **Embedding 数据正常，统计量几乎一致。**

**Step 2：提取日志确认**
- RSNA ViT-g 提取：预训练权重 ✅（03:23 完成）
- RHPE ViT-g 提取：预训练权重 ✅（06:43 完成）
- E22 训练时间：08:16（在 RSNA 预训练重跑之后）✅
- 无 error/warning/NaN

**Step 3：Metadata 对齐**
- Metadata 6,204 行 ↔ Labels 6,204 行，case_id 完全对齐 ✅
- true_age 范围一致：12~228 月 ✅

**Step 4：年龄分段泛化对比**

| 年龄段 | ViT-B (E14) RHPE | ViT-g (E22) RHPE | ViT-g RSNA test |
|:------|:----------------:|:----------------:|:---------------:|
| 0-60 月 | 15.00 | **6.93** | 9.60 |
| 60-120 月 | 10.12 | **16.88** | 9.08 |
| 120-180 月 | 14.98 | **33.32** ⚠️ | 8.01 |
| 180+ 月 | 31.20 | **52.80** ⚠️ | 10.37 |

### 诊断结论

**不是数据问题**，ViT-g 的 RSNA 和 RHPE embedding 统计量几乎一致（均值差 0.0007，标准差比 0.9997），提取日志确认均使用预训练权重，metadata 完全对齐。

问题出在模型本身的泛化能力。ViT-g 在低年龄段（0-60 月）RHPE 泛化反而比 ViT-B 更好（6.93 vs 15.00），但从 **120 月开始急剧恶化**（33.32、52.80）。在 RSNA 测试集上同年龄段 MAE 仅 8~10。

初步判断：ViT-g（1.1B 参数，1536 维特征）对 RSNA 特有的高龄段特征模式过拟合更强。RHPE 的高龄段 X 光影像风格与 RSNA 存在分布差异时，ViT-g 的大容量表示不如小模型鲁棒。该结果属于有效实验发现，无需重跑 embedding。

---

## 五、执行中遇到的异常

| 问题 | 影响范围 | 处理方式 |
|:-----|:---------|:---------|
| HuggingFace CDN 不可用，ViT-g 首次使用随机权重 | RSNA ViT-g embedding | 上传本地预训练权重后清空重跑 |
| `pip install` 时联网重试卡死 | 首次环境准备 | 增加 `HF_HUB_OFFLINE=1` 环境变量 |
| CNN 脚本 `HF_HUB_OFFLINE` 设置时机过晚 | RHPE CNN 提取 | 改为 `env HF_HUB_OFFLINE=1` 调用 |
| RHPE CSV 列名 `ID` 与脚本默认 `id` 不匹配 | RHPE CNN 提取 | 修改 `extract_cnn_embeddings.py` 列名匹配逻辑 |
| RHPE 图片文件名补零格式检测逻辑错误 | RHPE CNN 提取 | 修复 `__getitem__` 中条件判断 |
| `analyze_results.py` 实验目录名查找硬编码 | 图表生成 | 增加 `_find_exp_dir()` 函数改用前缀匹配 |
| `infer_rhpe.py` 无法导入 `gw_mae` 模块 | 泛化推理 | 将 `utils/` 加入 `sys.path` |

---

## 五、目录结构

```
LIM/CCBBD/
├── LIM/
│   ├── code/                     ← 提取/训练/分析脚本
│   ├── embeddings/
│   │   ├── rsna/                 ← 6 backbone × 12611 个 embedding ✅
│   │   └── rhpe/                 ← 6 backbone × 6204 个 embedding ✅
│   ├── experiments/              ← 24 个实验 (E01~E24) ✅
│   ├── generalization/           ← 6 backbone 泛化结果 ✅
│   ├── figures/                  ← 7 张分析图 ✅
│   └── logs/
│       ├── progress.md           ← 本文件
│       ├── errors.txt            ← 异常/修复记录
│       ├── pipeline.log          ← 旧自动流水线日志 (6月7日)
│       └── overnight_run.log     ← 通宵流水线日志
├── LIM_data/                     ← 数据集
│   ├── rsna/   (12,611 张 + labels.csv)
│   ├── rhpe/   (6,212 张 + labels.csv)
│   └── dinov2_weights/           ← DINOv2 预训练权重
└── data/ → LIM_data              ← 软链接
```

---

**报告结束**
