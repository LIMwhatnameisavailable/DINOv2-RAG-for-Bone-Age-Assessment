"""
LIM/code/train_mlp.py

在 DINOv2/CNN embedding 上训练 MLP 回归头进行骨龄预测。

迁移自 bone/code/train/train_mlp_from_embeddings.py，核心改动：
  1. 路径常量替换为 LIM_ROOT + DATA_ROOT 体系
  2. 损失函数使用 LIM/code/utils/gw_mae.py 的 CombinedLoss（连续高斯加权）
     （而非 bone/ 中 eval_utils.py 的分段常数 GrowthWindowWeights）
  3. 模型/数据集/工具函数内联（不依赖 bone/agents/mlp_age_service.py）
  4. 输出目录: LIM/experiments/{exp_id}/

用法:
  conda run -n boneage python LIM/code/train_mlp.py ^
    --embeddings-dir LIM/embeddings/rsna/dinov2_vitg/embeddings ^
    --metadata-csv   LIM/embeddings/rsna/dinov2_vitg/metadata/rsna_metadata.csv ^
    --labels-csv     data/rsna/labels.csv ^
    --output-dir     LIM/experiments/E21_vitg_mae_nogender ^
    --feat-dim 1536 --alpha-mae 1.0 --beta-gw-mae 0.0 --no-gender ^
    --epochs 80 --seed 42 --cpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# ── LIM 路径常量 ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent                         # LIM/code/
LIM_ROOT   = SCRIPT_DIR.parent                                       # LIM/
DATA_ROOT  = (LIM_ROOT / ".." / "data").resolve()                   # data/

sys.path.insert(0, str(SCRIPT_DIR / "utils"))
from gw_mae import CombinedLoss, GrowthWeightedMAELoss  # noqa: E402


# ── 默认路径 ────────────────────────────────────────────────────────────────
DEFAULT_FEAT_DIM = 1536
DEFAULT_HIDDEN_DIMS = (512, 128)
DEFAULT_METADATA_CSV = LIM_ROOT / "embeddings" / "rsna" / "dinov2_vitg" / "metadata" / "rsna_metadata.csv"
DEFAULT_LABELS_CSV = DATA_ROOT / "rsna" / "labels.csv"
DEFAULT_EMBEDDINGS_DIR = LIM_ROOT / "embeddings" / "rsna" / "dinov2_vitg" / "embeddings"
DEFAULT_OUTPUT_DIR = LIM_ROOT / "experiments"


# ============================================================================
# 模型定义
# ============================================================================

class BoneAgeEmbeddingMLP(nn.Module):
    """MLP 回归头，输入 embedding（可选+性别），输出骨龄（月）。

    结构: LayerNorm → Linear(feat_dim+gender, 512) → GELU → Dropout
                → Linear(512, 128) → GELU → Dropout
                → Linear(128, 1) → Softplus
    """
    def __init__(self, feat_dim: int, hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
                 use_gender: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dims = hidden_dims
        self.use_gender = use_gender
        input_dim = feat_dim + (1 if use_gender else 0)
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.GELU(), nn.Dropout(0.2)]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)
        self.output_activation = nn.Softplus()

    def forward(self, embedding: torch.Tensor,
                gender: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_gender:
            if gender is None:
                raise ValueError("gender feature required but model has use_gender=True")
            if gender.ndim == 1:
                gender = gender.unsqueeze(1)
            x = torch.cat([embedding, gender], dim=1)
        else:
            x = embedding
        return self.output_activation(self.head(x)).squeeze(1)


# ============================================================================
# 数据集
# ============================================================================

def _normalize_gender(value: Any) -> str:
    s = str(value).strip().lower()
    if s in ("m", "male", "1", "true"):
        return "M"
    if s in ("f", "female", "0", "false"):
        return "F"
    return "unknown"


class BoneAgeEmbeddingDataset(Dataset):
    """从 metadata CSV 加载 embedding 和标签。

    metadata_csv 必须含列: case_id, gender, true_age, embedding_file
    labels_csv 可选，用于覆盖/补充 true_age（例如从 labels.csv 取 Boneage 列）
    """
    def __init__(
        self,
        metadata_csv: Path,
        embeddings_dir: Path,
        *,
        label_csv: Path | None = None,
        use_gender: bool = True,
        embedding_mean: np.ndarray | None = None,
        embedding_std: np.ndarray | None = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.use_gender = use_gender
        self.embedding_mean = embedding_mean
        self.embedding_std = embedding_std

        self.frame = pd.read_csv(metadata_csv)
        required = {"case_id", "gender", "true_age", "embedding_file"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"metadata_csv missing columns: {sorted(missing)}")
        self.frame = self.frame.copy()
        self.frame["case_id"] = self.frame["case_id"].astype(str)
        self.frame["true_age"] = pd.to_numeric(self.frame["true_age"], errors="coerce")

        # 可选：用 labels.csv 中的 Boneage 覆盖 true_age
        if label_csv is not None:
            lf = pd.read_csv(label_csv)
            id_col = next((c for c in ("case_id", "ID", "id") if c in lf.columns), None)
            age_col = next((c for c in ("Boneage", "boneage", "BoneAge", "true_age", "Chronological")
                           if c in lf.columns), None)
            if id_col is None or age_col is None:
                raise ValueError(f"labels_csv missing id/age column: {label_csv}")
            lf = lf.copy()
            lf["case_id"] = lf[id_col].astype(str)
            lf["label_age"] = pd.to_numeric(lf[age_col], errors="coerce")
            self.frame = self.frame.merge(lf[["case_id", "label_age"]], on="case_id", how="left")
            self.frame["true_age"] = self.frame["label_age"].combine_first(self.frame["true_age"])
            self.frame.drop(columns=["label_age"], inplace=True, errors="ignore")

        before = len(self.frame)
        self.frame = self.frame[np.isfinite(self.frame["true_age"].to_numpy(dtype=float))].reset_index(drop=True)
        dropped = before - len(self.frame)
        if dropped > 0:
            print(f"[Dataset] Dropped {dropped} rows with non-finite true_age")
        if len(self.frame) == 0:
            raise ValueError("No valid samples after filtering")

    def __len__(self) -> int:
        return len(self.frame)

    def _gender_to_feat(self, val: Any) -> float:
        norm = _normalize_gender(val)
        return 1.0 if norm == "M" else (0.0 if norm == "F" else 0.5)

    def _norm_embedding(self, emb: np.ndarray) -> np.ndarray:
        if self.embedding_mean is None or self.embedding_std is None:
            return emb
        return (emb - self.embedding_mean) / np.clip(self.embedding_std, 1e-6, None)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        emb_path = Path(str(row["embedding_file"]))
        if not emb_path.is_absolute():
            emb_path = (self.embeddings_dir / emb_path.name).resolve()
        if not emb_path.exists():
            raise FileNotFoundError(f"embedding not found: {emb_path}")
        emb = np.load(emb_path).astype(np.float32)
        if not np.isfinite(emb).all():
            raise ValueError(f"non-finite embedding: {emb_path}")
        emb = self._norm_embedding(emb)
        return {
            "case_id": str(row["case_id"]),
            "embedding": torch.tensor(emb, dtype=torch.float32),
            "true_age": torch.tensor(float(row["true_age"]), dtype=torch.float32),
            "gender": torch.tensor(self._gender_to_feat(row["gender"]), dtype=torch.float32),
        }


# ============================================================================
# 指标计算
# ============================================================================

def compute_regression_metrics(
    preds: np.ndarray, targets: np.ndarray, genders: np.ndarray | None = None
) -> dict[str, Any]:
    """计算回归指标（MAE / GW-MAE / RMSE / 分性别MAE / 分年龄段MAE）。

    GW-MAE 使用 LIM/utils/gw_mae.py 的 GrowthWeightedMAELoss 权重函数。
    """
    preds = np.asarray(preds, dtype=np.float32).ravel()
    targets = np.asarray(targets, dtype=np.float32).ravel()
    abs_err = np.abs(preds - targets)
    signed_err = preds - targets

    # GW-MAE 权重
    gw_fn = GrowthWeightedMAELoss()
    weights = np.asarray([gw_fn.weight(torch.tensor(float(a))).item() for a in targets], dtype=np.float32)

    metrics: dict[str, Any] = {
        "num_samples": int(len(targets)),
        "mae": float(np.mean(abs_err)),
        "gw_mae": float(np.sum(abs_err * weights) / max(np.sum(weights), 1e-6)),
        "rmse": float(np.sqrt(np.mean(signed_err ** 2))),
        "median_ae": float(np.median(abs_err)),
        "r2": float(1.0 - np.sum(signed_err ** 2) / max(np.sum((targets - np.mean(targets)) ** 2), 1e-6)),
        "mae_by_gender": {},
        "mae_by_age_group": {},
    }

    # 分性别
    if genders is not None:
        genders_arr = np.asarray(genders)
        for g in sorted(set(str(x) for x in genders_arr)):
            mask = np.array([str(x) == g for x in genders_arr])
            if mask.any():
                metrics["mae_by_gender"][g] = float(np.mean(abs_err[mask]))

    # 分年龄段
    age_bins = [(0, 60), (60, 120), (120, 180), (180, 300)]
    for lo, hi in age_bins:
        mask = (targets >= lo) & (targets < hi)
        if mask.any():
            key = f"{int(lo)}_{int(hi)}"
            metrics["mae_by_age_group"][key] = float(np.mean(abs_err[mask]))

    return metrics


# ============================================================================
# 工具函数
# ============================================================================

def _parse_hidden_dims(raw: str) -> tuple[int, ...]:
    parts = [int(p.strip()) for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("hidden-dims cannot be empty")
    return tuple(parts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices(
    dataset: BoneAgeEmbeddingDataset,
    train_ratio: float, val_ratio: float, test_ratio: float, seed: int,
) -> tuple[list[int], list[int], list[int]]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    frame = dataset.frame.copy()
    age_bins = pd.cut(frame["true_age"], bins=[0, 24, 60, 120, 180, 240, np.inf],
                      labels=False, include_lowest=True)
    strat_key = frame["gender"].astype(str) + "_" + age_bins.fillna(-1).astype(int).astype(str)
    indices = np.arange(len(frame))

    try:
        tv_idx, te_idx = train_test_split(indices, test_size=test_ratio,
                                          random_state=seed, shuffle=True, stratify=strat_key)
    except ValueError:
        tv_idx, te_idx = train_test_split(indices, test_size=test_ratio,
                                          random_state=seed, shuffle=True, stratify=None)

    remaining = train_ratio + val_ratio
    val_share = val_ratio / remaining if remaining > 0 else 0
    try:
        tr_idx, va_idx = train_test_split(tv_idx, test_size=val_share,
                                          random_state=seed, shuffle=True,
                                          stratify=strat_key.iloc[tv_idx])
    except ValueError:
        tr_idx, va_idx = train_test_split(tv_idx, test_size=val_share,
                                          random_state=seed, shuffle=True, stratify=None)

    return tr_idx.tolist(), va_idx.tolist(), te_idx.tolist()


def _make_loader(dataset: BoneAgeEmbeddingDataset, indices: list[int],
                 batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def _compute_embedding_norm(
    dataset: BoneAgeEmbeddingDataset, indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not indices:
        raise ValueError("no indices for normalization")
    s = ss = None
    cnt = 0
    for idx in indices:
        emb = dataset[idx]["embedding"].numpy().astype(np.float64)
        if s is None:
            s = np.zeros_like(emb, dtype=np.float64)
            ss = np.zeros_like(emb, dtype=np.float64)
        s += emb
        ss += emb * emb
        cnt += 1
    assert s is not None and ss is not None
    mean = s / cnt
    var = np.maximum(ss / cnt - mean * mean, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> dict[str, Any]:
    model.eval()
    preds, targets, genders, ids = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            emb = batch["embedding"].to(device)
            gen = batch["gender"].to(device)
            tgt = batch["true_age"].to(device)
            out = model(emb, gen).cpu().numpy().ravel()
            preds.extend(out.tolist())
            targets.extend(tgt.cpu().numpy().ravel().tolist())
            for v in batch["gender"].cpu().numpy().ravel():
                genders.append("M" if abs(float(v) - 1.0) < 1e-6 else "F")
            ids.extend(batch["case_id"])

    metrics = compute_regression_metrics(np.array(preds), np.array(targets), np.array(genders))
    rows = [{"case_id": ids[i], "gender": genders[i],
             "true_age": float(targets[i]), "pred_age": float(preds[i]),
             "abs_error": abs(float(preds[i] - targets[i]))}
            for i in range(len(preds))]
    return {"metrics": metrics, "rows": rows}


# ============================================================================
# 主函数
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MLP on embeddings for bone age regression")
    p.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    p.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV),
                   help="Optional label CSV to fill true_age")
    p.add_argument("--embeddings-dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    p.add_argument("--output-dir", required=True,
                   help="Output directory (e.g. LIM/experiments/E01_...)")
    p.add_argument("--feat-dim", type=int, default=DEFAULT_FEAT_DIM)
    p.add_argument("--hidden-dims", default=",".join(str(v) for v in DEFAULT_HIDDEN_DIMS))
    p.add_argument("--no-gender", action="store_true", help="Disable gender input")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 = disable")
    p.add_argument("--alpha-mae", type=float, default=1.0, help="MAE coefficient")
    p.add_argument("--beta-gw-mae", type=float, default=1.0, help="GW-MAE coefficient")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    use_gender = not args.no_gender
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device} | Output: {output_dir}")
    print(f"Config: feat_dim={args.feat_dim} hidden={hidden_dims} "
          f"gender={use_gender} alpha_mae={args.alpha_mae} beta_gw_mae={args.beta_gw_mae}")

    # ── 数据集 ──
    label_csv = Path(args.labels_csv) if args.labels_csv and Path(args.labels_csv).exists() else None
    dataset = BoneAgeEmbeddingDataset(
        metadata_csv=Path(args.metadata_csv),
        embeddings_dir=Path(args.embeddings_dir),
        label_csv=label_csv,
        use_gender=use_gender,
    )
    tr_idx, va_idx, te_idx = _split_indices(
        dataset, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    print(f"Samples: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    # ── Embedding 归一化（基于训练集） ──
    emb_mean, emb_std = _compute_embedding_norm(dataset, tr_idx)
    dataset.embedding_mean = emb_mean
    dataset.embedding_std = emb_std

    train_loader = _make_loader(dataset, tr_idx, args.batch_size, True, args.num_workers)
    val_loader = _make_loader(dataset, va_idx, args.batch_size, False, args.num_workers)
    test_loader = _make_loader(dataset, te_idx, args.batch_size, False, args.num_workers)

    # ── 模型 + 损失 ──
    model = BoneAgeEmbeddingMLP(
        feat_dim=args.feat_dim, hidden_dims=hidden_dims, use_gender=use_gender
    ).to(device)

    # 使用 LIM 的 CombinedLoss（连续高斯加权 GW-MAE）
    criterion = CombinedLoss(alpha_mae=args.alpha_mae, beta_gw_mae=args.beta_gw_mae)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    grad_clip = max(float(args.grad_clip), 0.0) if float(args.grad_clip) > 0 else None

    # ── 训练循环 ──
    best_val_mae = float("inf")
    best_epoch = -1
    wait = 0
    best_state: dict | None = None
    history: list[dict] = []
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = r_mae = r_gw = 0.0
        n = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            emb = batch["embedding"].to(device)
            gen = batch["gender"].to(device)
            tgt = batch["true_age"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(emb, gen)
            loss = criterion(pred, tgt)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = int(tgt.shape[0])
            running_loss += float(loss.item()) * bs
            r_mae += float(nn.L1Loss()(pred, tgt).item()) * bs
            r_gw += float(GrowthWeightedMAELoss()(pred, tgt).item()) * bs
            n += bs

        train_loss = running_loss / n
        train_mae = r_mae / n
        train_gw = r_gw / n

        val_res = _evaluate(model, val_loader, device)
        val_m = val_res["metrics"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_gw_mae": train_gw,
            "val_mae": val_m["mae"],
            "val_gw_mae": val_m["gw_mae"],
            "val_rmse": val_m["rmse"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} train_mae={train_mae:.4f} "
              f"val_mae={val_m['mae']:.4f} val_gw_mae={val_m['gw_mae']:.4f}")

        if val_m["mae"] < best_val_mae:
            best_val_mae = float(val_m["mae"])
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "state_dict": best_state,
                "config": {
                    "feat_dim": args.feat_dim,
                    "hidden_dims": list(hidden_dims),
                    "use_gender": use_gender,
                    "alpha_mae": args.alpha_mae,
                    "beta_gw_mae": args.beta_gw_mae,
                    "seed": args.seed,
                },
                "extra": {
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                    "input_normalization": {"mean": emb_mean.tolist(), "std": emb_std.tolist()},
                },
            }, best_path)
        else:
            wait += 1

        torch.save(model.state_dict(), last_path)
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── 保存最终结果 ──
    if not best_path.exists() and best_state is not None:
        torch.save({"state_dict": best_state, "config": {}, "extra": {}}, best_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = _evaluate(model, train_loader, device)
    val_eval = _evaluate(model, val_loader, device)
    test_eval = _evaluate(model, test_loader, device)

    metrics = {"train": train_eval["metrics"], "val": val_eval["metrics"],
               "test": test_eval["metrics"]}

    # 写文件
    def _js(x): return json.loads(json.dumps(x, default=str))
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(_js(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "split.json").write_text(
        json.dumps(_js({"train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx}),
                   indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(train_eval["rows"]).to_csv(output_dir / "train_predictions.csv", index=False)
    pd.DataFrame(val_eval["rows"]).to_csv(output_dir / "val_predictions.csv", index=False)
    pd.DataFrame(test_eval["rows"]).to_csv(output_dir / "test_predictions.csv", index=False)

    # 实验配置摘要
    summary = {
        "experiment_id": output_dir.name,
        "backbone": "unknown",  # 需从路径推断或由调用者指定
        "loss": ("mae" if args.alpha_mae > 0 and args.beta_gw_mae == 0 else
                 "gwmae" if args.alpha_mae == 0 and args.beta_gw_mae > 0 else "combined"),
        "use_gender": use_gender,
        **{f"test_{k}": v for k, v in test_eval["metrics"].items() if isinstance(v, (int, float))},
        "best_epoch": best_epoch,
        "total_epochs": len(history),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps({**summary, "args": vars(args)}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults saved to: {output_dir}")
    print(f"Test MAE: {test_eval['metrics']['mae']:.4f}  "
          f"GW-MAE: {test_eval['metrics']['gw_mae']:.4f}  "
          f"RMSE: {test_eval['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    import sys
    main()