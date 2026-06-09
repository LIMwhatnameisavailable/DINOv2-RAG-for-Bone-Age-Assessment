"""
LIM/code/extract_cnn_embeddings.py

使用 timm 库的 CNN 模型（ResNet-50 / EfficientNet-B4）提取骨龄 X 光片 Embedding。

设计原则：
  - 与 build_vision_library.py 接口一致（相同的 CLI 风格、输出目录结构、metadata 格式）
  - 所有模型使用 ImageNet-1k 原始预训练权重
  - 移除分类头，取 global avg pooling 后的特征向量
  - 支持断点续传

用法：
  conda run -n boneage python LIM/code/extract_cnn_embeddings.py ^
    --dataset rsna ^
    --model resnet50 ^
    --output-dir LIM/embeddings/rsna/resnet50 ^
    --batch-size 32 ^
    --cpu

支持的 Backbone：
  - resnet50       → feat_dim=2048
  - efficientnet_b4 → feat_dim=1792
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── 路径常量 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # LIM/code/
LIM_ROOT   = SCRIPT_DIR.parent                        # LIM/
DATA_ROOT  = (LIM_ROOT / ".." / "data").resolve()    # data/

DATASET_PRESETS: dict[str, dict[str, str | None]] = {
    "rsna": {
        "label_csv": str(DATA_ROOT / "rsna" / "labels.csv"),
        "img_dir": str(DATA_ROOT / "rsna" / "images"),
    },
    "rhpe": {
        "label_csv": str(DATA_ROOT / "rhpe" / "labels.csv"),
        "img_dir": str(DATA_ROOT / "rhpe" / "images"),
    },
}

# CNN 模型配置
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "resnet50": {
        "timm_name": "resnet50",
        "feat_dim": 2048,
        "pretrained": True,
        "default_batch_size": 64,
    },
    "efficientnet_b4": {
        "timm_name": "efficientnet_b4",
        "feat_dim": 1792,
        "pretrained": True,
        "default_batch_size": 64,
    },
}


def _resolve_device(use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_name: str, device: torch.device) -> nn.Module:
    """加载 timm 模型，移除分类头，返回特征提取器。"""
    import timm

    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(MODEL_CONFIGS.keys())}")

    # 设置离线模式（避免联网超时）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        full_model = timm.create_model(
            cfg["timm_name"],
            pretrained=cfg["pretrained"],
            num_classes=0,
        )
        print(f"Model: {model_name} loaded from timm hub")
    except Exception:
        print(f"Hub download failed, loading local weights for {model_name}...")
        full_model = timm.create_model(
            cfg["timm_name"],
            pretrained=False,
            num_classes=0,
        )
        local_weights = {
            "resnet50":      "/tmp/timm_weights/resnet50.safetensors",
            "efficientnet_b4": "/tmp/timm_weights/efficientnet_b4.safetensors",
        }.get(model_name)
        if local_weights:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(local_weights)
            full_model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded local weights: {local_weights}")
        else:
            print(f"  WARNING: no local weights for {model_name}, using random init")

    full_model = full_model.to(device).eval()
    print(f"Model: {model_name} (feat_dim={cfg['feat_dim']}) ready on {device}")
    return full_model


def build_transform(model_name: str) -> T.Compose:
    """构建与 CNN 模型匹配的预处理流程。"""
    import timm
    model_cfg_key = MODEL_CONFIGS[model_name]["timm_name"]
    # 创建临时模型（不加载预训练权重）以获取数据配置
    model = timm.create_model(model_cfg_key, pretrained=False, num_classes=0)
    data_cfg = timm.data.resolve_data_config(model=model)
    # 确保 crop_pct 存在
    data_cfg.setdefault("crop_pct", 0.875)
    return timm.data.create_transform(**data_cfg)


class XRayDataset(Dataset):
    """从 CSV 加载 X 光图像，返回预处理后的 tensor。"""
    def __init__(self, label_csv: str, img_dir: str, id_col: str = "id",
                 img_fmt: str = "{}.png", transform: T.Compose | None = None):
        self.df = pd.read_csv(label_csv)
        self.img_dir = Path(img_dir)
        self.id_col = id_col
        self.img_fmt = img_fmt
        self.transform = transform or T.Compose([T.ToTensor()])

        # 统一列名以兼容 RSNA (id) 和 RHPE (ID)
        id_col_lower = id_col.lower()
        actual_col = None
        for col in self.df.columns:
            if col.lower() == id_col_lower:
                actual_col = col
                break
        if actual_col is None:
            raise ValueError(f"CSV missing id column '{id_col}'. Columns: {list(self.df.columns)}")
        self.id_col_actual = actual_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        row = self.df.iloc[idx]
        case_id = str(row[self.id_col_actual])

        # RHPE 使用 5 位补零 ID
        if self.img_fmt == "{:05d}.png":
            try:
                img_id = int(case_id)
            except ValueError:
                img_id = int(case_id)
        else:
            img_id = case_id

        if self.img_fmt == "{}.png":
            img_path = self.img_dir / f"{case_id}.png"
        else:
            img_path = self.img_dir / self.img_fmt.format(img_id if isinstance(img_id, int) else case_id)

        if not img_path.exists():
            # 备选：尝试不带补零
            img_path = self.img_dir / f"{case_id}.png"

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return str(case_id), image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract CNN embeddings for bone age X-rays")
    p.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()), required=True)
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True,
                   help="CNN backbone: resnet50 or efficientnet_b4")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override default batch size for the model")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples (for smoke test)")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p


def main() -> None:
    args = build_parser().parse_args()
    model_cfg = MODEL_CONFIGS[args.model]
    dataset_cfg = DATASET_PRESETS[args.dataset]
    label_csv = dataset_cfg["label_csv"]
    img_dir = dataset_cfg["img_dir"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输入验证
    if not label_csv or not os.path.exists(label_csv):
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")
    if not img_dir or not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    device = _resolve_device(args.cpu)
    batch_size = args.batch_size or model_cfg.get("default_batch_size", 64)
    feat_dim = model_cfg["feat_dim"]

    # ── 数据集 ──
    id_col = "id" if args.dataset == "rsna" else "ID"
    img_fmt = "{}.png" if args.dataset == "rsna" else "{:05d}.png"
    transform = build_transform(args.model)
    dataset = XRayDataset(label_csv, img_dir, id_col=id_col, img_fmt=img_fmt, transform=transform)

    if args.max_samples and args.max_samples < len(dataset):
        indices = list(range(args.max_samples))
        dataset.df = dataset.df.iloc[indices].reset_index(drop=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # ── 模型 ──
    model = build_model(args.model, device)
    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    # ── 提取 ──
    start_time = time.time()
    new_count = 0
    skip_count = 0
    metadata_rows: list[dict[str, Any]] = []

    for case_ids, images in tqdm(loader, desc=f"Extracting {args.model}"):
        images = images.to(device)
        with torch.inference_mode():
            features = model(images).cpu().numpy()  # shape: (B, feat_dim)

        for i, cid in enumerate(case_ids):
            out_path = embed_dir / f"{cid}.npy"
            if out_path.exists():
                skip_count += 1
            else:
                np.save(str(out_path), features[i].astype(np.float32))
                new_count += 1

            metadata_rows.append({
                "case_id": cid,
                "embedding_file": str(out_path.resolve()),
            })

    elapsed = time.time() - start_time
    print(f"\nDone. New: {new_count}, Skipped: {skip_count}, "
          f"Time: {elapsed:.0f}s ({elapsed / max(new_count + skip_count, 1):.2f}s/img)")

    # ── 构建 metadata ──
    meta_df = pd.DataFrame(metadata_rows).drop_duplicates(subset="case_id")
    # 合并标签信息
    label_df = pd.read_csv(label_csv)
    id_col_actual = "id" if "id" in label_df.columns else ("ID" if "ID" in label_df.columns else label_df.columns[0])
    age_col = next((c for c in ("boneage", "Boneage", "BoneAge", "true_age") if c in label_df.columns), None)
    gender_col = next((c for c in ("male", "Male", "gender", "Gender") if c in label_df.columns), None)

    label_df["case_id"] = label_df[id_col_actual].astype(str)
    if age_col:
        label_df["true_age"] = pd.to_numeric(label_df[age_col], errors="coerce")
    if gender_col:
        label_df["gender"] = label_df[gender_col].apply(
            lambda x: "M" if str(x).lower() in ("m", "male", "1", "true") else "F")

    meta_df = meta_df.merge(
        label_df[["case_id", "true_age", "gender"]], on="case_id", how="left")
    # 填充缺失值
    meta_df["true_age"] = meta_df["true_age"].fillna(0).astype(int)
    meta_df["gender"] = meta_df["gender"].fillna("U")

    # 写入 metadata CSV
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_csv_name = f"{args.dataset}_metadata.csv"
    meta_path = meta_dir / meta_csv_name
    meta_df.to_csv(meta_path, index=False)
    print(f"Metadata saved: {meta_path} ({len(meta_df)} rows)")

    # ── 写入 build summary ──
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "feat_dim": feat_dim,
        "device": str(device),
        "num_samples": len(meta_df),
        "new_samples": new_count,
        "existing_samples": skip_count,
        "elapsed_seconds": round(elapsed, 1),
        "embeddings_dir": str(embed_dir),
        "metadata_path": str(meta_path),
        "img_dir": str(img_dir),
        "label_csv": str(label_csv),
    }
    (output_dir / "build_summary.json").write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved: {output_dir / 'build_summary.json'}")
    print(f"Feature dimension: {feat_dim}")


if __name__ == "__main__":
    main()