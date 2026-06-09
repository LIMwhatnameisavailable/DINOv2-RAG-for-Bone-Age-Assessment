"""
LIM/code/infer_rhpe.py

跨数据集泛化推理：将 RSNA 上训练好的 MLP 模型，直接在 RHPE 数据集上推理。

流程：
  1. 加载 RHPE embedding（已由 build_vision_library.py 或 extract_cnn_embeddings.py 提取）
  2. 加载训练好的 MLP checkpoint
  3. 逐样本推理
  4. 与 RHPE 标签对比，输出指标
  5. 保存 predictions.csv + metrics.json

用法：
  conda run -n boneage python LIM/code/infer_rhpe.py ^
    --model-path LIM/experiments/E24_vitg_combined_gender/best.pt ^
    --embeddings-dir LIM/embeddings/rhpe/dinov2_vitg/embeddings ^
    --metadata-csv   LIM/embeddings/rhpe/dinov2_vitg/metadata/rhpe_metadata.csv ^
    --labels-csv     data/rhpe/labels.csv ^
    --output-dir     LIM/generalization/best_model_on_rhpe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── LIM 路径 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
LIM_ROOT   = SCRIPT_DIR.parent
DATA_ROOT  = (LIM_ROOT / ".." / "data").resolve()

# 将 utils 目录加入路径
sys.path.insert(0, str(SCRIPT_DIR / "utils"))
import gw_mae

# ── 模型定义（与 train_mlp.py 一致） ─────────────────────────────────────────

def _build_mlp(feat_dim: int, hidden_dims: tuple[int, ...],
               use_gender: bool) -> torch.nn.Module:
    import torch.nn as nn
    input_dim = feat_dim + (1 if use_gender else 0)
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers += [nn.Linear(in_dim, h_dim), nn.GELU(), nn.Dropout(0.2)]
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, 1))
    head = nn.Sequential(*layers)
    output_activation = nn.Softplus()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head
            self.output_activation = output_activation
            self.use_gender = use_gender
            self.feat_dim = feat_dim
            self.hidden_dims = hidden_dims

        def forward(self, embedding, gender=None):
            if self.use_gender:
                if gender is None:
                    raise ValueError("gender required")
                if gender.ndim == 1:
                    gender = gender.unsqueeze(1)
                x = torch.cat([embedding, gender], dim=1)
            else:
                x = embedding
            return self.output_activation(self.head(x)).squeeze(1)

    return MLP()


class InferenceDataset(Dataset):
    """加载 RHPE embedding 用于推理。"""
    def __init__(self, metadata_csv: Path, embeddings_dir: Path):
        self.df = pd.read_csv(metadata_csv)
        self.embeddings_dir = Path(embeddings_dir)
        required = {"case_id", "gender", "embedding_file"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"metadata_csv missing {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        emb_path = Path(str(row["embedding_file"]))
        if not emb_path.is_absolute():
            emb_path = (self.embeddings_dir / emb_path.name).resolve()
        if not emb_path.exists():
            raise FileNotFoundError(f"{emb_path}")
        emb = np.load(emb_path).astype(np.float32)

        # 归一化（如果 checkpoint 有保存 normalization 参数）
        return {
            "case_id": str(row["case_id"]),
            "embedding": torch.tensor(emb, dtype=torch.float32),
            "gender": torch.tensor(
                1.0 if str(row["gender"]).upper() == "M" else 0.0, dtype=torch.float32),
        }


def compute_metrics(preds: np.ndarray, targets: np.ndarray,
                    genders: np.ndarray | None = None) -> dict:
    """计算评估指标。"""
    preds = np.asarray(preds, dtype=np.float32).ravel()
    targets = np.asarray(targets, dtype=np.float32).ravel()
    abs_err = np.abs(preds - targets)
    signed_err = preds - targets

    gw_fn = gw_mae.GrowthWeightedMAELoss()
    weights = np.asarray(
        [gw_fn.weight(torch.tensor(float(a))).item() for a in targets], dtype=np.float32)

    metrics = {
        "num_samples": int(len(targets)),
        "mae": float(np.mean(abs_err)),
        "gw_mae": float(np.sum(abs_err * weights) / max(np.sum(weights), 1e-6)),
        "rmse": float(np.sqrt(np.mean(signed_err ** 2))),
        "median_ae": float(np.median(abs_err)),
        "mae_by_gender": {},
        "mae_by_age_group": {},
    }

    if genders is not None:
        genders_arr = np.asarray(genders)
        for g in sorted(set(str(x) for x in genders_arr)):
            mask = np.array([str(x) == g for x in genders_arr])
            if mask.any():
                metrics["mae_by_gender"][g] = float(np.mean(abs_err[mask]))

    for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 300)]:
        mask = (targets >= lo) & (targets < hi)
        if mask.any():
            metrics["mae_by_age_group"][f"{lo}_{hi}"] = float(np.mean(abs_err[mask]))

    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-dataset inference on RHPE")
    p.add_argument("--model-path", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--labels-csv", required=True, help="RHPE labels for evaluation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # ── 加载模型 ──
    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    extra = ckpt.get("extra", {})

    feat_dim = int(config.get("feat_dim", 1536))
    hidden_dims = tuple(int(v) for v in config.get("hidden_dims", [512, 128]))
    use_gender = bool(config.get("use_gender", True))
    alpha_mae = float(config.get("alpha_mae", 1.0))
    beta_gw_mae = float(config.get("beta_gw_mae", 0.0))

    print(f"  feat_dim={feat_dim} hidden={hidden_dims} gender={use_gender} "
          f"alpha_mae={alpha_mae} beta_gw_mae={beta_gw_mae}")

    model = _build_mlp(feat_dim, hidden_dims, use_gender).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # ── 加载 normalization 参数（如果存在） ──
    input_norm = extra.get("input_normalization", {})
    embedding_mean = np.array(input_norm.get("mean"), dtype=np.float32) if input_norm.get("mean") else None
    embedding_std = np.array(input_norm.get("std"), dtype=np.float32) if input_norm.get("std") else None

    # ── 数据集 ──
    dataset = InferenceDataset(Path(args.metadata_csv), Path(args.embeddings_dir))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    # ── 推理 ──
    all_ids: list[str] = []
    all_preds: list[float] = []
    all_genders: list[str] = []

    for batch in tqdm(loader, desc="Inferring on RHPE"):
        emb = batch["embedding"].to(device)
        gen = batch["gender"].to(device)

        # 应用训练集的 normalization
        if embedding_mean is not None and embedding_std is not None:
            mean_t = torch.tensor(embedding_mean, device=device)
            std_t = torch.tensor(embedding_std, device=device).clamp_min(1e-6)
            emb = (emb - mean_t) / std_t

        with torch.inference_mode():
            pred = model(emb, gen).cpu().numpy().ravel()

        all_ids.extend(batch["case_id"])
        all_preds.extend(pred.tolist())
        for v in batch["gender"].cpu().numpy().ravel():
            all_genders.append("M" if abs(float(v) - 1.0) < 1e-6 else "F")

    all_preds = np.array(all_preds, dtype=np.float32)

    # ── 加载 RHPE 真实标签 ──
    label_df = pd.read_csv(args.labels_csv)
    id_col = next((c for c in ("case_id", "ID", "id") if c in label_df.columns), None)
    age_col = next((c for c in ("Boneage", "boneage", "BoneAge", "true_age") if c in label_df.columns), None)
    if id_col is None or age_col is None:
        raise ValueError(f"labels_csv missing id/age column: {args.labels_csv}")

    label_df["case_id"] = label_df[id_col].astype(str)
    label_df["true_age"] = pd.to_numeric(label_df[age_col], errors="coerce")

    # ── 合并 ──
    result_df = pd.DataFrame({
        "case_id": all_ids,
        "gender": all_genders,
        "pred_age": all_preds,
    })
    result_df = result_df.merge(label_df[["case_id", "true_age"]], on="case_id", how="left")
    result_df = result_df.dropna(subset=["true_age"])

    result_df["abs_error"] = np.abs(result_df["pred_age"] - result_df["true_age"]).values

    print(f"\nEvaluated on {len(result_df)} RHPE samples with labels")

    # ── 指标 ──
    metrics = compute_metrics(
        result_df["pred_age"].values,
        result_df["true_age"].values,
        result_df["gender"].values,
    )

    print(f"RHPE Generalization Results:")
    print(f"  MAE:  {metrics['mae']:.4f} months")
    print(f"  GW-MAE: {metrics['gw_mae']:.4f} months")
    print(f"  RMSE: {metrics['rmse']:.4f} months")

    # ── 保存 ──
    result_df.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "model_path": str(args.model_path),
        "num_samples": len(result_df),
        **metrics,
        "config": config,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()