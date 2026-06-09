#!/bin/bash
# LIM 骨龄项目 通宵全自动流水线（P1→P7）
# 启动: nohup bash LIM/code/overnight_pipeline.sh > LIM/logs/overnight_run.log 2>&1 &
# 监控: tail -f LIM/logs/overnight_run.log

set -e
cd /mnt/disk2/srtp2024/LIM/CCBBD
PY=/mnt/disk2/srtp2024/miniconda3/envs/LIM_boneage/bin/python
STEPS_DIR=/mnt/disk2/srtp2024/LIM/logs/.steps
mkdir -p "$STEPS_DIR"

step_done() { [ -f "$STEPS_DIR/$1.done" ]; }
mark_done() { touch "$STEPS_DIR/$1.done"; echo "[$(date)] ✅ $1 完成"; }
npy_count() { ls "$1"/embeddings/*.npy 2>/dev/null | wc -l; }

echo "=============================================="
echo "🚀 LIM 通宵全自动流水线"
echo "启动时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="

# ── 1. RSNA ViT-S ──
if ! step_done rsna_vits; then
  EXPECTED=12611
  CUR=$(npy_count LIM/embeddings/rsna/dinov2_vits)
  if [ "$CUR" -ge "$EXPECTED" ]; then
    mark_done rsna_vits
  else
    echo "[RSNA ViT-S] $CUR/$EXPECTED → 启动提取..."
    $PY LIM/code/build_vision_library.py --model dinov2_vits14 --dataset rsna \
      --output-dir LIM/embeddings/rsna/dinov2_vits --batch-size 64 \
      2>&1 | tee LIM/logs/vits_rsna_full.log
    CUR2=$(npy_count LIM/embeddings/rsna/dinov2_vits)
    if [ "$CUR2" -ge "$EXPECTED" ]; then mark_done rsna_vits; else echo "❌ ViT-S count mismatch: $CUR2"; fi
  fi
else echo "⏭️ [SKIP] RSNA ViT-S 已完成"; fi

# ── 2. RSNA ViT-B ──
if ! step_done rsna_vitb; then
  EXPECTED=12611
  CUR=$(npy_count LIM/embeddings/rsna/dinov2_vitb)
  if [ "$CUR" -ge "$EXPECTED" ]; then
    mark_done rsna_vitb
  else
    echo "[RSNA ViT-B] $CUR/$EXPECTED → 启动提取..."
    $PY LIM/code/build_vision_library.py --model dinov2_vitb14 --dataset rsna \
      --output-dir LIM/embeddings/rsna/dinov2_vitb --batch-size 32 \
      2>&1 | tee LIM/logs/vitb_rsna_full.log
    CUR2=$(npy_count LIM/embeddings/rsna/dinov2_vitb)
    if [ "$CUR2" -ge "$EXPECTED" ]; then mark_done rsna_vitb; else echo "❌ ViT-B count mismatch: $CUR2"; fi
  fi
else echo "⏭️ [SKIP] RSNA ViT-B 已完成"; fi

# ── 3. RSNA ViT-L ──
if ! step_done rsna_vitl; then
  EXPECTED=12611
  CUR=$(npy_count LIM/embeddings/rsna/dinov2_vitl)
  if [ "$CUR" -ge "$EXPECTED" ]; then
    mark_done rsna_vitl
  else
    echo "[RSNA ViT-L] $CUR/$EXPECTED → 启动提取..."
    $PY LIM/code/build_vision_library.py --model dinov2_vitl14 --dataset rsna \
      --output-dir LIM/embeddings/rsna/dinov2_vitl --batch-size 16 \
      2>&1 | tee LIM/logs/vitl_rsna_full.log
    CUR2=$(npy_count LIM/embeddings/rsna/dinov2_vitl)
    if [ "$CUR2" -ge "$EXPECTED" ]; then mark_done rsna_vitl; else echo "❌ ViT-L count mismatch: $CUR2"; fi
  fi
else echo "⏭️ [SKIP] RSNA ViT-L 已完成"; fi

# ── 4. RSNA ViT-g (用预训练权重重跑) ──
if ! step_done rsna_vitg_pretrained; then
  EXPECTED=12611
  CUR=$(npy_count LIM/embeddings/rsna/dinov2_vitg)
  # ViT-g already has 12611 files but from random weights — redo if no .done marker for pretrained
  echo "[RSNA ViT-g] 用预训练权重重跑..."
  # Clear old random-weight embeddings
  rm -f LIM/embeddings/rsna/dinov2_vitg/embeddings/*.npy
  rm -f LIM/embeddings/rsna/dinov2_vitg/metadata/*.csv
  $PY LIM/code/build_vision_library.py --model dinov2_vitg14 --dataset rsna \
    --output-dir LIM/embeddings/rsna/dinov2_vitg --batch-size 8 \
    2>&1 | tee LIM/logs/vitg_rsna_full.log
  CUR2=$(npy_count LIM/embeddings/rsna/dinov2_vitg)
  if [ "$CUR2" -ge "$EXPECTED" ]; then mark_done rsna_vitg_pretrained; else echo "❌ ViT-g count mismatch: $CUR2"; fi
else echo "⏭️ [SKIP] RSNA ViT-g 预训练已完成"; fi

# ── 5. RHPE: 全部 6 个 backbone ──
echo ""
echo "══════════════════════════════════════════════"
echo "开始 RHPE Embedding 提取（6 个 backbone）"
echo "══════════════════════════════════════════════"

RHPE_EXPECTED=6204

run_rhpe() {
  local name=$1 model=$2 batch=$3
  local flag="rhpe_${name}"
  if step_done "$flag"; then
    echo "⏭️ [SKIP] RHPE $name 已完成"
    return
  fi
  local cur=$(npy_count "LIM/embeddings/rhpe/${name}")
  if [ "$cur" -ge "$RHPE_EXPECTED" ]; then
    mark_done "$flag"
    return
  fi
  echo "[RHPE $name] 启动提取..."
  if [ "$name" = "resnet50" ] || [ "$name" = "efficientnet_b4" ]; then
    $PY LIM/code/extract_cnn_embeddings.py --dataset rhpe --model "${model}" \
      --output-dir "LIM/embeddings/rhpe/${name}" --batch-size "${batch}" \
      2>&1 | tee "LIM/logs/${name}_rhpe.log"
  else
    $PY LIM/code/build_vision_library.py --model "${model}" --dataset rhpe \
      --output-dir "LIM/embeddings/rhpe/${name}" --batch-size "${batch}" \
      2>&1 | tee "LIM/logs/${model}_rhpe.log"
  fi
  local cur2=$(npy_count "LIM/embeddings/rhpe/${name}")
  if [ "$cur2" -ge "$RHPE_EXPECTED" ]; then mark_done "$flag"; else echo "❌ RHPE $name count: $cur2"; fi
}

# RHPE Wave 1: 轻量模型并行
run_rhpe resnet50       resnet50        128 &
run_rhpe efficientnet_b4 efficientnet_b4 128 &
run_rhpe dinov2_vits    dinov2_vits14   64  &
wait

# RHPE Wave 2: 中等
run_rhpe dinov2_vitb dinov2_vitb14 32

# RHPE Wave 3: 大模型
run_rhpe dinov2_vitl dinov2_vitl14 16

# RHPE Wave 4: 超大模型
run_rhpe dinov2_vitg dinov2_vitg14 8

echo ""
echo "=============================================="
echo "🎉 P1 Embedding 提取全部完成！"
echo "时间: $(date)"
echo "=============================================="

# ── P2: Embedding 质量审查 ──
echo ""
echo "══════════════════════════════════════════════"
echo "P2: Embedding 质量审查"
echo "══════════════════════════════════════════════"

$PY -c "
import os, numpy as np, pandas as pd, random, sys, glob

checks = [
    ('rsna', 'resnet50',       12611),
    ('rsna', 'efficientnet_b4',12611),
    ('rsna', 'dinov2_vits',    12611),
    ('rsna', 'dinov2_vitb',    12611),
    ('rsna', 'dinov2_vitl',    12611),
    ('rsna', 'dinov2_vitg',    12611),
    ('rhpe', 'resnet50',        6204),
    ('rhpe', 'efficientnet_b4', 6204),
    ('rhpe', 'dinov2_vits',     6204),
    ('rhpe', 'dinov2_vitb',     6204),
    ('rhpe', 'dinov2_vitl',     6204),
    ('rhpe', 'dinov2_vitg',     6204),
]

errors = []
for dataset, backbone, expected in checks:
    emb_dir = f'LIM/embeddings/{dataset}/{backbone}/embeddings'
    meta_csv = f'LIM/embeddings/{dataset}/{backbone}/metadata/{dataset}_metadata.csv'
    tag = f'{dataset}/{backbone}'
    if not os.path.exists(emb_dir):
        errors.append(f'P2 FAIL - {tag} - embeddings dir missing')
        continue
    files = [f for f in os.listdir(emb_dir) if f.endswith('.npy')]
    if len(files) != expected:
        errors.append(f'P2 FAIL - {tag} - count {len(files)} != {expected}')
        continue
    for f in random.sample(files, min(5, len(files))):
        arr = np.load(os.path.join(emb_dir, f))
        if arr.dtype != np.float32:
            errors.append(f'P2 FAIL - {tag} - dtype {arr.dtype}')
    if os.path.exists(meta_csv):
        df = pd.read_csv(meta_csv)
        if df.true_age.isnull().sum() > 0:
            errors.append(f'P2 FAIL - {tag} - null true_age found')
    else:
        errors.append(f'P2 WARN - {tag} - metadata csv missing')
    print(f'  [{tag}] {len(files)} files OK')

if errors:
    with open('LIM/logs/errors.txt', 'a') as f:
        for e in errors: f.write(e + chr(10))
    print(f'\\n{len(errors)} errors written to LIM/logs/errors.txt')
else:
    print('\\n✅ P2 ALL PASS')
"

# ── P3: 24 个 MLP 实验训练 ──
echo ""
echo "══════════════════════════════════════════════"
echo "P3: 24 个 MLP 实验训练"
echo "══════════════════════════════════════════════"

EXPS=(
  "E01_resnet50_mae_nogender    --feat-dim 2048 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E02_resnet50_mae_gender      --feat-dim 2048 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E03_resnet50_gwmae_gender    --feat-dim 2048 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E04_resnet50_combined_gender --feat-dim 2048 --alpha-mae 1.0 --beta-gw-mae 1.0"
  "E05_effb4_mae_nogender       --feat-dim 1792 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E06_effb4_mae_gender         --feat-dim 1792 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E07_effb4_gwmae_gender       --feat-dim 1792 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E08_effb4_combined_gender    --feat-dim 1792 --alpha-mae 1.0 --beta-gw-mae 1.0"
  "E09_vits_mae_nogender        --feat-dim 384  --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E10_vits_mae_gender          --feat-dim 384  --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E11_vits_gwmae_gender        --feat-dim 384  --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E12_vits_combined_gender     --feat-dim 384  --alpha-mae 1.0 --beta-gw-mae 1.0"
  "E13_vitb_mae_nogender        --feat-dim 768  --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E14_vitb_mae_gender          --feat-dim 768  --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E15_vitb_gwmae_gender        --feat-dim 768  --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E16_vitb_combined_gender     --feat-dim 768  --alpha-mae 1.0 --beta-gw-mae 1.0"
  "E17_vitl_mae_nogender        --feat-dim 1024 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E18_vitl_mae_gender          --feat-dim 1024 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E19_vitl_gwmae_gender        --feat-dim 1024 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E20_vitl_combined_gender     --feat-dim 1024 --alpha-mae 1.0 --beta-gw-mae 1.0"
  "E21_vitg_mae_nogender        --feat-dim 1536 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender"
  "E22_vitg_mae_gender          --feat-dim 1536 --alpha-mae 1.0 --beta-gw-mae 0.0"
  "E23_vitg_gwmae_gender        --feat-dim 1536 --alpha-mae 0.0 --beta-gw-mae 1.0"
  "E24_vitg_combined_gender     --feat-dim 1536 --alpha-mae 1.0 --beta-gw-mae 1.0"
)

BACKBONE_DIRS=(
  "resnet50"
  "resnet50"
  "resnet50"
  "resnet50"
  "efficientnet_b4"
  "efficientnet_b4"
  "efficientnet_b4"
  "efficientnet_b4"
  "dinov2_vits"
  "dinov2_vits"
  "dinov2_vits"
  "dinov2_vits"
  "dinov2_vitb"
  "dinov2_vitb"
  "dinov2_vitb"
  "dinov2_vitb"
  "dinov2_vitl"
  "dinov2_vitl"
  "dinov2_vitl"
  "dinov2_vitl"
  "dinov2_vitg"
  "dinov2_vitg"
  "dinov2_vitg"
  "dinov2_vitg"
)

for i in "${!EXPS[@]}"; do
  exp="${EXPS[$i]}"
  bb="${BACKBONE_DIRS[$i]}"
  id=$(echo "$exp" | awk '{print $1}')
  args=$(echo "$exp" | cut -d' ' -f2-)
  out_dir="LIM/experiments/$id"
  emb_dir="LIM/embeddings/rsna/${bb}"

  if [ -f "$out_dir/best.pt" ]; then
    echo "[SKIP] $id — best.pt exists"
    continue
  fi

  echo "[START] $id (backbone=$bb)"

  $PY LIM/code/train_mlp.py \
    --output-dir "$out_dir" \
    --embeddings-dir "${emb_dir}/embeddings" \
    --metadata-csv "${emb_dir}/metadata/rsna_metadata.csv" \
    --labels-csv LIM_data/rsna/labels.csv \
    --epochs 80 --batch-size 64 --lr 1e-3 --seed 42 \
    $args \
    2>&1 | tee "LIM/logs/${id}.log" \
  && echo "[DONE] $id" \
  || echo "P3 FAIL - $id" >> LIM/logs/errors.txt
done

echo "✅ P3 全部完成"

# ── P4: 收敛审查 ──
echo ""
echo "══════════════════════════════════════════════"
echo "P4: 训练收敛审查"
echo "══════════════════════════════════════════════"

$PY -c "
import json, glob, os
errors = []
for exp_dir in sorted(glob.glob('LIM/experiments/E*')):
    name = os.path.basename(exp_dir)
    hist_path = f'{exp_dir}/history.json'
    if not os.path.exists(hist_path):
        errors.append(f'P4 - {name} - history.json missing')
        continue
    h = json.load(open(hist_path))
    if not h: continue
    first_val = h[0].get('val_mae', 999)
    last_val  = h[-1].get('val_mae', 999)
    best_epoch = min(range(len(h)), key=lambda i: h[i].get('val_mae', 999))
    if last_val > 15: errors.append(f'P4 WARN - {name} - val_mae={last_val:.2f}')
    print(f'  {name}: {first_val:.2f} → {last_val:.2f}, best={best_epoch}')
if errors:
    with open('LIM/logs/errors.txt','a') as f:
        for e in errors: f.write(e+chr(10))
    print(f'\\n{len(errors)} warnings')
else:
    print('\\n✅ P4 ALL PASS')
"

# ── P5: 跨数据集泛化 ──
echo ""
echo "══════════════════════════════════════════════"
echo "P5: 跨数据集泛化推理 (RHPE)"
echo "══════════════════════════════════════════════"

$PY -c "
import os, subprocess
PY = '/mnt/disk2/srtp2024/miniconda3/envs/LIM_boneage/bin/python'

BBS = [
    ('resnet50',       'E02_resnet50_mae_gender',      2048),
    ('efficientnet_b4','E06_effb4_mae_gender',         1792),
    ('dinov2_vits',    'E10_vits_mae_gender',           384),
    ('dinov2_vitb',    'E14_vitb_mae_gender',           768),
    ('dinov2_vitl',    'E18_vitl_mae_gender',          1024),
    ('dinov2_vitg',    'E22_vitg_mae_gender',          1536),
]
for bb, exp_id, dim in BBS:
    model_path = f'LIM/experiments/{exp_id}/best.pt'
    out_dir = f'LIM/generalization/{exp_id}_on_rhpe_{bb}'
    if not os.path.exists(model_path):
        print(f'[SKIP] {bb} no model at {model_path}'); continue
    if os.path.exists(f'{out_dir}/predictions.csv'):
        print(f'[SKIP] {bb} already done'); continue
    ret = subprocess.run([
        PY, 'LIM/code/infer_rhpe.py',
        '--model-path', model_path,
        '--embeddings-dir', f'LIM/embeddings/rhpe/{bb}/embeddings',
        '--metadata-csv', f'LIM/embeddings/rhpe/{bb}/metadata/rhpe_metadata.csv',
        '--labels-csv', 'LIM_data/rhpe/labels.csv',
        '--output-dir', out_dir,
    ])
    if ret.returncode == 0: print(f'[DONE] {bb} -> RHPE')
    else: print(f'[FAIL] {bb}')
" 2>&1

echo "✅ P5 完成"

# ── P6: 分析 + 图表 ──
echo ""
echo "══════════════════════════════════════════════"
echo "P6: 汇总分析 + 7 张图表"
echo "══════════════════════════════════════════════"

$PY LIM/code/analyze_results.py --mode all 2>&1 | tee LIM/logs/analysis.log

echo "✅ P6 完成"

# ── P7: 结果审查 ──
echo ""
echo "══════════════════════════════════════════════"
echo "P7: 最终结果审查"
echo "══════════════════════════════════════════════"

$PY -c "
import pandas as pd
df = pd.read_csv('LIM/experiments/summary.csv')
print('=== FINAL RESULTS (sorted by test_mae) ===')
print(df.sort_values('test_mae')[['dataset','backbone','loss','use_gender','test_mae','test_rmse']].to_string(index=False))
"

echo ""
echo "=============================================="
echo "🎉🎉🎉 全流程完成！"
echo "时间: $(date)"
echo "=============================================="