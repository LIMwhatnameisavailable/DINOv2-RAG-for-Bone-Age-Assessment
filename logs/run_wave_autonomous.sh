#!/bin/bash
# LIM 项目全自动流水线（断点续跑，顺序执行）
# 启动: nohup bash LIM/logs/run_wave_autonomous.sh > LIM/logs/pipeline.log 2>&1 &
# 监控: tail -f LIM/logs/pipeline.log

set -e

cd /mnt/disk2/srtp2024/LIM/CCBBD
PYTHON=/mnt/disk2/srtp2024/miniconda3/envs/LIM_boneage/bin/python
STEPS_DIR=/mnt/disk2/srtp2024/LIM/CCBBD/LIM/logs/.steps
mkdir -p "$STEPS_DIR" LIM/logs LIM/embeddings/rsna

step_done() { [ -f "$STEPS_DIR/$1.done" ]; }
mark_done() { touch "$STEPS_DIR/$1.done"; echo "[$(date)] ✅ $1 完成"; }

echo "=============================================="
echo "🚀 LIM 项目全自动流水线"
echo "启动时间: $(date)"
echo "环境: LIM_boneage (Python 3.10, PyTorch 2.1.2)"
echo "GPU: RTX 3090 (24 GB)"
echo "模式: 串行执行（避免 OOM）"
echo "=============================================="

# ─────────────── 1. RSNA ResNet-50 ───────────────
if ! step_done rsna_resnet50; then
  echo ""
  echo "══════════════════════════════════════════════"
  echo "[1/5] RSNA ResNet-50 (batch 128, feat_dim=2048)"
  echo "══════════════════════════════════════════════"
  $PYTHON LIM/code/extract_cnn_embeddings.py \
    --dataset rsna --model resnet50 \
    --output-dir LIM/embeddings/rsna/resnet50 --batch-size 128 \
    2>&1 | tee LIM/logs/resnet50_rsna_v2.log
  if [ $? -eq 0 ]; then
    mark_done rsna_resnet50
  else
    echo "❌ ResNet-50 失败，查看 LIM/logs/resnet50_rsna_v2.log"
    exit 1
  fi
else
  echo "⏭️ [1/5] RSNA ResNet-50 已完成，跳过"
fi

# ─────────────── 2. RSNA EfficientNet-B4 ──────────
if ! step_done rsna_effb4; then
  echo ""
  echo "══════════════════════════════════════════════"
  echo "[2/5] RSNA EfficientNet-B4 (batch 128, feat_dim=1792)"
  echo "══════════════════════════════════════════════"
  $PYTHON LIM/code/extract_cnn_embeddings.py \
    --dataset rsna --model efficientnet_b4 \
    --output-dir LIM/embeddings/rsna/efficientnet_b4 --batch-size 128 \
    2>&1 | tee LIM/logs/effb4_rsna_v2.log
  if [ $? -eq 0 ]; then
    mark_done rsna_effb4
  else
    echo "❌ EffNet-B4 失败，查看 LIM/logs/effb4_rsna_v2.log"
    exit 1
  fi
else
  echo "⏭️ [2/5] RSNA EffNet-B4 已完成，跳过"
fi

# ─────────────── 3. RSNA ViT-g (DINOv2) ──────────
# 注意: build_vision_library.py 硬编码 dinov2_vitg14
#       使用 pretrained=False（权重需手动下载后更新）
if ! step_done rsna_vitg; then
  echo ""
  echo "══════════════════════════════════════════════"
  echo "[3/5] RSNA ViT-g (DINOv2, batch 16, feat_dim=1536)"
  echo "注意: pretrained=False, 需要上传预训练权重"
  echo "══════════════════════════════════════════════"
  $PYTHON LIM/code/build_vision_library.py \
    --dataset rsna \
    --output-dir LIM/embeddings/rsna/dinov2_vitg --batch-size 16 \
    2>&1 | tee LIM/logs/vitg_rsna_v2.log
  if [ $? -eq 0 ]; then
    mark_done rsna_vitg
  else
    echo "❌ ViT-g 失败，查看 LIM/logs/vitg_rsna_v2.log"
    exit 1
  fi
else
  echo "⏭️ [3/5] RSNA ViT-g 已完成，跳过"
fi

echo ""
echo "=============================================="
echo "🎉 RSNA Embedding 提取全部完成！"
echo "时间: $(date)"
echo "=============================================="

# ─────────────── P1 审查 ──────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "P1: Embedding 质量审查"
echo "══════════════════════════════════════════════"
for model in resnet50 efficientnet_b4 dinov2_vitg; do
  emb_dir="LIM/embeddings/rsna/$model/embeddings"
  meta_dir="LIM/embeddings/rsna/$model/metadata"
  count=$(find "$emb_dir" -name '*.npy' 2>/dev/null | wc -l)
  meta_count=$(find "$meta_dir" -name '*.csv' 2>/dev/null | wc -l)
  echo "  $model: $count / 12611 .npy (metadata: $meta_count csv)"
done