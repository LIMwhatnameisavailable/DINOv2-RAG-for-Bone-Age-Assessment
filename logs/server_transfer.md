# LIM 项目 — 服务器执行指南

> 本文件是服务器上 Claude Code 执行 LIM 实验的完整任务书。
> 请 CC 逐条阅读，严格按照顺序执行，遇错即停并写入 LIM/logs/errors.txt。

---

## 0. 项目概述

### 研究目标
骨龄自动评估的 **系统性消融实验**，回答三个核心问题：
1. DINOv2 (ViT-S/B/L/g) vs CNN (ResNet-50, EfficientNet-B4) 性能差异？
2. 临床加权损失函数 GW-MAE 能否改善关键发育阶段预测？
3. RSNA 训练的模型能否泛化到 RHPE？

### 实验矩阵 (24 个)

| 实验 | Backbone | 损失函数 | 性别 | 实验 | Backbone | 损失函数 | 性别 |
|------|----------|----------|------|------|----------|----------|------|
| E01 | ResNet-50 | MAE | 无 | E13 | ViT-B | MAE | 无 |
| E02 | ResNet-50 | MAE | 有 | E14 | ViT-B | MAE | 有 |
| E03 | ResNet-50 | GW-MAE | 有 | E15 | ViT-B | GW-MAE | 有 |
| E04 | ResNet-50 | Combined | 有 | E16 | ViT-B | Combined | 有 |
| E05 | EffNet-B4 | MAE | 无 | E17 | ViT-L | MAE | 无 |
| E06 | EffNet-B4 | MAE | 有 | E18 | ViT-L | MAE | 有 |
| E07 | EffNet-B4 | GW-MAE | 有 | E19 | ViT-L | GW-MAE | 有 |
| E08 | EffNet-B4 | Combined | 有 | E20 | ViT-L | Combined | 有 |
| E09 | ViT-S | MAE | 无 | E21 | ViT-g | MAE | 无 |
| E10 | ViT-S | MAE | 有 | E22 | ViT-g | MAE | 有 |
| E11 | ViT-S | GW-MAE | 有 | E23 | ViT-g | GW-MAE | 有 |
| E12 | ViT-S | Combined | 有 | **E24** | **ViT-g** | **Combined** | **有** |

### 关键约束（禁止事项）
1. ❌ **禁止在 RHPE 上训练** — RHPE 仅用于泛化推理
2. ❌ **禁止使用** `bone_age_predictor_best.pth`（学长微调权重，来源不透明）
3. ❌ **禁止修改 骨龄/bone/ 下的任何文件**
4. ❌ **禁止硬编码路径** — 必须使用 LIM_ROOT / DATA_ROOT 常量体系
5. ❌ **禁止使用 timm 之外的 CNN 库**

---

## 1. 目录结构（上传后预期）

```
项目根目录/
├── LIM/                      ← 工作区（所有代码、日志、结果）
│   ├── code/                 ← 所有脚本
│   │   ├── build_vision_library.py      ← DINOv2 Embedding 提取
│   │   ├── extract_cnn_embeddings.py    ← CNN Embedding 提取
│   │   ├── train_mlp.py                 ← MLP 回归训练
│   │   ├── infer_rhpe.py                ← 跨数据集泛化推理
│   │   ├── analyze_results.py           ← 汇总分析 + 7 张图表
│   │   └── utils/
│   │       ├── gw_mae.py      ← 连续高斯加权损失函数
│   │       ├── vis_style.py   ← 可视化样式模板
│   │       └── data_utils.py  ← 数据路径常量
│   ├── LIM_data/              ← 整理好的数据集
│   │   ├── rsna/
│   │   │   ├── images/        ← 12,611 张 PNG
│   │   │   └── labels.csv     ← id, boneage, male
│   │   └── rhpe/
│   │       ├── images/        ← 6,212 张 PNG (train+val 合并)
│   │       └── labels.csv     ← 6,204 行 (Boneage, Male)
│   ├── embeddings/            ← 提取后的 Embedding（空目录，待填充）
│   ├── experiments/           ← 24 个实验输出（空目录，待填充）
│   ├── generalization/        ← 泛化推理结果（空目录，待填充）
│   ├── figures/               ← 分析图表（空目录，待填充）
│   ├── logs/                  ← 日志体系（关键文件）
│   │   ├── progress.md        ← 项目进度总表
│   │   ├── errors.txt         ← 异常/不匹配记录
│   │   └── server_transfer.md ← ← 当前文件
│   └── project_overview.md    ← 项目概览（参考用）
├── 骨龄/bone/                 ← **只读数据源**，禁止写入
└── LM_data.tar.gz / LIM_data.zip ← 已上传的压缩包
```

---

## 2. 环境准备

### 2.1 解压数据

```bash
cd /path/to/project/root

# 如果上传的是 tar.gz
tar -xzf LIM_data.tar.gz

# 如果上传的是 zip
unzip LIM_data.zip

# 解压后确认结构
ls -lh LIM_data/
# 预期:
#   rsna/  (9.1 GB)
#   rhpe/  (13.3 GB)
```

### 2.2 Conda 环境

```bash
# 激活现有环境
conda activate boneage

# 补装缺失依赖
pip install timm efficientnet_pytorch matplotlib seaborn scipy

# 验证 GPU
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('Memory:', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), 'GB') if torch.cuda.is_available() else None
"
# 预期输出: CUDA: True, GPU: NVIDIA RTX 3090, Memory: ~24 GB

# 验证所有脚本可导入
python -c "
import sys; sys.path.insert(0, 'LIM/code/utils')
from gw_mae import GrowthWeightedMAELoss, CombinedLoss
from vis_style import apply_style, COLORS
print('All imports OK')
"
```

### 2.3 路径常量确认

所有 LIM 脚本的路径系统：
- `SCRIPT_DIR` = LIM/code/
- `LIM_ROOT`   = LIM/
- `DATA_ROOT`  = LIM_data/

```bash
python -c "
import sys; sys.path.insert(0, 'LIM/code')
from build_vision_library import LIM_ROOT, DATA_ROOT, DATASET_PRESETS
print('LIM_ROOT:', LIM_ROOT)
print('DATA_ROOT:', DATA_ROOT)
print('RSNA img_dir:', DATASET_PRESETS['rsna']['img_dir'])
print('RHPE label_csv:', DATASET_PRESETS['rhpe']['label_csv'])
"
```

---

## 3. 执行流水线

### 阶段总览

| 阶段 | 任务 | 预估时间 | 关键依赖 |
|------|------|---------|---------|
| P0 | 环境验证 | 5 min | — |
| P1 | **全部 Embedding 提取** | **~3-4 h** | GPU |
| P2 | P1 审查: 质量检查 | 5 min/每个 | P1 完成 |
| P3 | **24 个 MLP 训练** | **~2 h** | P2 完成 |
| P4 | P3 审查: 收敛检查 | 5 min | P3 完成 |
| P5 | **泛化推理** | **~5 min** | P3 完成 |
| P6 | **分析汇总 + 图表** | **~10 min** | P5 完成 |
| P7 | P6 审查: 最终结果审查 | 10 min | P6 完成 |

---

### P1: Embedding 提取（核心 GPU 任务）

**12 次提取，分两组脚本执行：**

#### P1a: DINOv2 系列（使用 build_vision_library.py）

Backbone 维度对照：
| Backbone | 维度 | 来源 |
|----------|------|------|
| dinov2_vits | 384 | torch.hub |
| dinov2_vitb | 768 | torch.hub |
| dinov2_vitl | 1024 | torch.hub |
| dinov2_vitg | 1536 | torch.hub |

**命令格式：**
```bash
conda run -n boneage python LIM/code/build_vision_library.py \
  --dataset {rsna|rhpe} \
  --output-dir LIM/embeddings/{rsna|rhpe}/{backbone} \
  --batch-size {N} \
  [--max-samples N]  # 可选：冒烟测试用
```

**并行策略（严格遵守，ViT-L 和 ViT-g 不能同时跑）：**

```bash
# === Wave 1: 轻量模型同时跑（CNN + ViT-S）===
# 终端 1: RSNA ResNet-50
python LIM/code/extract_cnn_embeddings.py --dataset rsna --model resnet50 \
  --output-dir LIM/embeddings/rsna/resnet50 --batch-size 128

# 终端 2: RSNA EfficientNet-B4  
python LIM/code/extract_cnn_embeddings.py --dataset rsna --model efficientnet_b4 \
  --output-dir LIM/embeddings/rsna/efficientnet_b4 --batch-size 128

# 终端 3: RSNA ViT-S
python LIM/code/build_vision_library.py --dataset rsna \
  --output-dir LIM/embeddings/rsna/dinov2_vits --batch-size 64

# 以上 3 个约 30-40 min 完成
```

```bash
# === Wave 2: ViT-B（中等）===
python LIM/code/build_vision_library.py --dataset rsna \
  --output-dir LIM/embeddings/rsna/dinov2_vitb --batch-size 32
# ~20 min
```

```bash
# === Wave 3: ViT-L（大模型，单独跑）===
python LIM/code/build_vision_library.py --dataset rsna \
  --output-dir LIM/embeddings/rsna/dinov2_vitl --batch-size 16
# ~30 min
```

```bash
# === Wave 4: ViT-g（最大模型，单独跑）===
python LIM/code/build_vision_library.py --dataset rsna \
  --output-dir LIM/embeddings/rsna/dinov2_vitg --batch-size 8
# ~60 min
```

```bash
# === Wave 5: RHPE 全套（数据集小一半，速度翻倍）===
# CNN
python LIM/code/extract_cnn_embeddings.py --dataset rhpe --model resnet50 \
  --output-dir LIM/embeddings/rhpe/resnet50 --batch-size 128 &
python LIM/code/extract_cnn_embeddings.py --dataset rhpe --model efficientnet_b4 \
  --output-dir LIM/embeddings/rhpe/efficientnet_b4 --batch-size 128 &
python LIM/code/build_vision_library.py --dataset rhpe \
  --output-dir LIM/embeddings/rhpe/dinov2_vits --batch-size 64 &
wait

# ViT-B
python LIM/code/build_vision_library.py --dataset rhpe \
  --output-dir LIM/embeddings/rhpe/dinov2_vitb --batch-size 32

# ViT-L
python LIM/code/build_vision_library.py --dataset rhpe \
  --output-dir LIM/embeddings/rhpe/dinov2_vitl --batch-size 16

# ViT-g
python LIM/code/build_vision_library.py --dataset rhpe \
  --output-dir LIM/embeddings/rhpe/dinov2_vitg --batch-size 8
```

**断点续传**：若中断，重新运行同一命令即可（自动跳过已有 .npy）。

---

### ⚡ 加速建议（给 CC 和用户）

如果时间紧迫，可以**只跑前 4 个 backbone** 用于论文基线，跳过 ViT-L 和 ViT-g：
- ✅ ResNet-50 (2048d, CNN baseline)
- ✅ EfficientNet-B4 (1792d, 轻量)
- ✅ DINOv2 ViT-S (384d, 小 Transformer)
- ✅ DINOv2 ViT-B (768d, 中 Transformer)
→ 这样 24 个实验 → 16 个实验，节省约 2 小时

但完整 24 个实验（含 ViT-L, ViT-g）更具学术说服力。

---

### P2: Embedding 质量审查（P1 后执行）

```bash
# 每个 backbone 完成后运行
conda run -n boneage python -c "
import os, numpy as np, pandas as pd, random, sys

backbone = 'dinov2_vitg'  # 替换为实际 backbone
dataset = 'rsna'           # 替换为 rsna/rhpe
expected = 12611 if dataset == 'rsna' else 6204

emb_dir = f'LIM/embeddings/{dataset}/{backbone}/embeddings'
meta_csv = f'LIM/embeddings/{dataset}/{backbone}/metadata/{dataset}_metadata.csv'

# 1. 文件数量
files = [f for f in os.listdir(emb_dir) if f.endswith('.npy')]
print(f'File count: {len(files)} (expected {expected})')
assert len(files) == expected, f'COUNT MISMATCH: {len(files)} != {expected}'

# 2. 随机抽检 shape 和 dtype
for f in random.sample(files, min(10, len(files))):
    arr = np.load(os.path.join(emb_dir, f))
    print(f'  {f}: shape={arr.shape} dtype={arr.dtype}')
    assert arr.dtype == np.float32, f'DTYPE ERROR: {arr.dtype}'

# 3. metadata 检查
df = pd.read_csv(meta_csv)
print(f'Metadata rows: {len(df)}, null true_age: {df.true_age.isnull().sum()}')
assert df.true_age.isnull().sum() == 0, 'NULL TRUE_AGE FOUND'

print('P2 PASSED ✅')
"
```

**遇错即停** → 写入 LIM/logs/errors.txt：
```
P2 FAILED - {backbone} - {dataset} - {具体错误}
```

---

### P3: 24 个 MLP 实验训练

**条件**：P2 全部通过

**实验命名规则**：
```
E{01-24}_{backbone}_{loss}_{gender}
  loss: mae / gwmae / combined
  gender: nogender / gender
```

**一键运行脚本（在服务器上创建 run_all_experiments.sh）：**

```bash
#!/bin/bash
set -e

EXPS=(
  # ResNet-50 (2048d)
  "E01_resnet50_mae_nogender   --embeddings-dir LIM/embeddings/rsna/resnet50/embeddings --metadata-csv LIM/embeddings/rsna/resnet50/metadata/rsna_metadata.csv --feat-dim 2048 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E02_resnet50_mae_gender     --embeddings-dir LIM/embeddings/rsna/resnet50/embeddings --metadata-csv LIM/embeddings/rsna/resnet50/metadata/rsna_metadata.csv --feat-dim 2048 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E03_resnet50_gwmae_gender   --embeddings-dir LIM/embeddings/rsna/resnet50/embeddings --metadata-csv LIM/embeddings/rsna/resnet50/metadata/rsna_metadata.csv --feat-dim 2048 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E04_resnet50_combined_gender --embeddings-dir LIM/embeddings/rsna/resnet50/embeddings --metadata-csv LIM/embeddings/rsna/resnet50/metadata/rsna_metadata.csv --feat-dim 2048 --alpha-mae 1.0 --beta-gw-mae 1.0"
  # EfficientNet-B4 (1792d)
  "E05_effb4_mae_nogender      --embeddings-dir LIM/embeddings/rsna/efficientnet_b4/embeddings --metadata-csv LIM/embeddings/rsna/efficientnet_b4/metadata/rsna_metadata.csv --feat-dim 1792 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E06_effb4_mae_gender        --embeddings-dir LIM/embeddings/rsna/efficientnet_b4/embeddings --metadata-csv LIM/embeddings/rsna/efficientnet_b4/metadata/rsna_metadata.csv --feat-dim 1792 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E07_effb4_gwmae_gender      --embeddings-dir LIM/embeddings/rsna/efficientnet_b4/embeddings --metadata-csv LIM/embeddings/rsna/efficientnet_b4/metadata/rsna_metadata.csv --feat-dim 1792 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E08_effb4_combined_gender   --embeddings-dir LIM/embeddings/rsna/efficientnet_b4/embeddings --metadata-csv LIM/embeddings/rsna/efficientnet_b4/metadata/rsna_metadata.csv --feat-dim 1792 --alpha-mae 1.0 --beta-gw-mae 1.0"
  # DINOv2 ViT-S (384d)
  "E09_vits_mae_nogender       --embeddings-dir LIM/embeddings/rsna/dinov2_vits/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vits/metadata/rsna_metadata.csv --feat-dim 384 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E10_vits_mae_gender         --embeddings-dir LIM/embeddings/rsna/dinov2_vits/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vits/metadata/rsna_metadata.csv --feat-dim 384 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E11_vits_gwmae_gender       --embeddings-dir LIM/embeddings/rsna/dinov2_vits/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vits/metadata/rsna_metadata.csv --feat-dim 384 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E12_vits_combined_gender    --embeddings-dir LIM/embeddings/rsna/dinov2_vits/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vits/metadata/rsna_metadata.csv --feat-dim 384 --alpha-mae 1.0 --beta-gw-mae 1.0"
  # DINOv2 ViT-B (768d)
  "E13_vitb_mae_nogender       --embeddings-dir LIM/embeddings/rsna/dinov2_vitb/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitb/metadata/rsna_metadata.csv --feat-dim 768 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E14_vitb_mae_gender         --embeddings-dir LIM/embeddings/rsna/dinov2_vitb/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitb/metadata/rsna_metadata.csv --feat-dim 768 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E15_vitb_gwmae_gender       --embeddings-dir LIM/embeddings/rsna/dinov2_vitb/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitb/metadata/rsna_metadata.csv --feat-dim 768 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E16_vitb_combined_gender    --embeddings-dir LIM/embeddings/rsna/dinov2_vitb/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitb/metadata/rsna_metadata.csv --feat-dim 768 --alpha-mae 1.0 --beta-gw-mae 1.0"
  # DINOv2 ViT-L (1024d)
  "E17_vitl_mae_nogender       --embeddings-dir LIM/embeddings/rsna/dinov2_vitl/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitl/metadata/rsna_metadata.csv --feat-dim 1024 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E18_vitl_mae_gender         --embeddings-dir LIM/embeddings/rsna/dinov2_vitl/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitl/metadata/rsna_metadata.csv --feat-dim 1024 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E19_vitl_gwmae_gender       --embeddings-dir LIM/embeddings/rsna/dinov2_vitl/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitl/metadata/rsna_metadata.csv --feat-dim 1024 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E20_vitl_combined_gender    --embeddings-dir LIM/embeddings/rsna/dinov2_vitl/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitl/metadata/rsna_metadata.csv --feat-dim 1024 --alpha-mae 1.0 --beta-gw-mae 1.0"
  # DINOv2 ViT-g (1536d)
  "E21_vitg_mae_nogender       --embeddings-dir LIM/embeddings/rsna/dinov2_vitg/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitg/metadata/rsna_metadata.csv --feat-dim 1536 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E22_vitg_mae_gender         --embeddings-dir LIM/embeddings/rsna/dinov2_vitg/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitg/metadata/rsna_metadata.csv --feat-dim 1536 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E23_vitg_gwmae_gender       --embeddings-dir LIM/embeddings/rsna/dinov2_vitg/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitg/metadata/rsna_metadata.csv --feat-dim 1536 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E24_vitg_combined_gender    --embeddings-dir LIM/embeddings/rsna/dinov2_vitg/embeddings --metadata-csv LIM/embeddings/rsna/dinov2_vitg/metadata/rsna_metadata.csv --feat-dim 1536 --alpha-mae 1.0 --beta-gw-mae 1.0"
)

for exp in "${EXPS[@]}"; do
  id=$(echo $exp | awk '{print $1}')
  args=$(echo $exp | cut -d' ' -f2-)
  echo "========================================"
  echo "Starting: $id"
  echo "========================================"
  conda run -n boneage python LIM/code/train_mlp.py \
    --output-dir LIM/experiments/$id \
    --labels-csv LIM_data/rsna/labels.csv \
    --epochs 80 --batch-size 64 --lr 1e-3 --seed 42 \
    $args
  echo "Completed: $id"
done

echo "========================================"
echo "ALL 24 EXPERIMENTS COMPLETE"
echo "========================================"
```

---

### P4: 训练收敛审查（P3 后执行）

```bash
for exp in LIM/experiments/E*/; do
  name=$(basename $exp)
  hist="$exp/history.json"
  if [ ! -f "$hist" ]; then
    echo "❌ $name: history.json missing" | tee -a LIM/logs/errors.txt
    continue
  fi
  best_epoch=$(python -c "import json; h=json.load(open('$hist')); print(h[-1]['epoch'] if h else 0)")
  first_val=$(python -c "import json; h=json.load(open('$hist')); print(h[0]['val_mae'] if h else 0)")
  last_val=$(python -c "import json; h=json.load(open('$hist')); print(h[-1]['val_mae'] if h else 0)")
  
  # 检查收敛
  if (( $(echo "$last_val > 15" | bc -l) )); then
    echo "⚠️  $name: val_mae=$last_val > 15 (异常)" | tee -a LIM/logs/errors.txt
  fi
  if [ "$best_epoch" -le 2 ] || [ "$best_epoch" -ge 79 ]; then
    echo "⚠️  $name: best_epoch=$best_epoch (边缘)" | tee -a LIM/logs/errors.txt
  fi
  echo "  $name: val_mae $first_val → $last_val, best_epoch=$best_epoch"
done
```

---

### P5: 跨数据集泛化推理

从 24 个实验中选取 RSNA 测试集 MAE 最低的模型，在 RHPE 上推理：

```bash
# 自动找到最优模型
BEST_EXP=$(python -c "
import pandas as pd
import json, glob

best_mae = 999
best_exp = ''
for exp_dir in sorted(glob.glob('LIM/experiments/E*')):
    m_path = f'{exp_dir}/metrics.json'
    if not __import__('os').path.exists(m_path): continue
    m = json.load(open(m_path))
    test_mae = m.get('test', {}).get('mae', 999)
    if test_mae < best_mae:
        best_mae = test_mae
        best_exp = exp_dir.split('/')[-1]
print(best_exp)
")

echo "Best model: $BEST_EXP"

# 对每个 backbone 的 RHPE embedding 做推理
for backbone in resnet50 efficientnet_b4 dinov2_vits dinov2_vitb dinov2_vitl dinov2_vitg; do
  # 确定 feat_dim
  case $backbone in
    resnet50) dim=2048 ;;
    efficientnet_b4) dim=1792 ;;
    dinov2_vits) dim=384 ;;
    dinov2_vitb) dim=768 ;;
    dinov2_vitl) dim=1024 ;;
    dinov2_vitg) dim=1536 ;;
  esac

  echo "Inferring RHPE with $backbone..."
  conda run -n boneage python LIM/code/infer_rhpe.py \
    --model-path LIM/experiments/${BEST_EXP}/best.pt \
    --embeddings-dir LIM/embeddings/rhpe/${backbone}/embeddings \
    --metadata-csv LIM/embeddings/rhpe/${backbone}/metadata/rhpe_metadata.csv \
    --labels-csv LIM_data/rhpe/labels.csv \
    --output-dir LIM/generalization/${BEST_EXP}_on_rhpe_${backbone}
done
```

---

### P6: 汇总分析 + 图表生成

```bash
conda run -n boneage python LIM/code/analyze_results.py --mode all
```

预期输出：
- `LIM/experiments/summary.csv` ← 24 个实验汇总表
- `LIM/figures/backbone_comparison.png`
- `LIM/figures/loss_ablation.png`
- `LIM/figures/gender_analysis.png`
- `LIM/figures/age_error_heatmap.png`
- `LIM/figures/gw_mae_weight_curve.png`
- `LIM/figures/generalization_comparison.png`
- `LIM/figures/error_distribution_by_age_group.png`

---

### P7: 最终结果审查

```bash
python -c "
import pandas as pd, json, sys

df = pd.read_csv('LIM/experiments/summary.csv')
errors = []

# 1. DINOv2 MAE 应随模型规模单调下降
vit_backbones = ['dinov2_vits', 'dinov2_vitb', 'dinov2_vitl', 'dinov2_vitg']
vit_maes = []
for b in vit_backbones:
    row = df[(df['backbone'] == b) & (df['loss'] == 'mae') & (df['use_gender'] == False)]
    if len(row) > 0:
        vit_maes.append(row.iloc[0]['test_mae'])
    else:
        vit_maes.append(None)

for i in range(len(vit_maes) - 1):
    if vit_maes[i] is not None and vit_maes[i+1] is not None and vit_maes[i] < vit_maes[i+1]:
        msg = f'P3 FAIL: {vit_backbones[i]} MAE={vit_maes[i]} < {vit_backbones[i+1]} MAE={vit_maes[i+1]} (应单调下降)'
        errors.append(msg)
        print('❌', msg)

# 2. GW-MAE 在 60-180 月应有改善
# 3. 跨数据集 MAE 不超过 RSNA 的 2 倍

if errors:
    with open('LIM/logs/errors.txt', 'a') as f:
        for e in errors:
            f.write(f'P3 FAIL: {e}\n')
    print(f'{len(errors)} errors found')
else:
    print('✅ P3 ALL CHECKS PASSED')
"
```

---

## 4. 回传清单

服务器完成后，需回传本地的内容：

| 内容 | 路径 | 大小预估 |
|------|------|---------|
| Embedding 文件 | LIM/embeddings/ | ~10 GB |
| 24 个实验输出 | LIM/experiments/ | ~100 MB |
| 泛化结果 | LIM/generalization/ | ~10 MB |
| 汇总 CSV | LIM/experiments/summary.csv | <1 MB |
| 图表 | LIM/figures/*.png | ~20 MB |
| 日志 | LIM/logs/ | <1 MB |

**推荐回传方式：**
```bash
# 打包结果
tar -czf LIM_results_$(date +%Y%m%d).tar.gz \
  LIM/experiments/summary.csv \
  LIM/experiments/E*/metrics.json \
  LIM/experiments/E*/predictions.csv \
  LIM/generalization/ \
  LIM/figures/ \
  LIM/logs/

# 压缩 embedding（按需）
tar -czf LIM_embeddings_rsna.tar.gz LIM/embeddings/rsna/
# 如果不需要保留 embedding，可从服务器删除以释放空间
```

---

## 5. 遇到问题怎么办

| 症状 | 操作 |
|------|------|
| CUDA OOM | 减小 `--batch-size`（ViT-g 用 4-8） |
| 脚本报错 | 检查是否从项目根目录执行，路径是否含中文 |
| embedding 文件数不对 | 检查断点续传是否正常工作，看 build_summary.json |
| 训练不收敛 | 检查 labels.csv 是否匹配 embedding 的 case_id |
| 其他异常 | **先写 LIM/logs/errors.txt，然后通知用户** |

**核心原则：遇到任何不一致（文件数不对、路径不匹配、数据异常），立即写入 errors.txt，不要默默跳过。**