"""LIM/code/build_vision_library.py

迁移自 bone/code/vision/build_vision_library.py。

修改说明：
  1. 路径常量替换为 LIM_ROOT + DATA_ROOT 体系
  2. 输出目录默认指向 LIM/embeddings/{dataset}/{backbone}/
  3. --weights 默认值改为 None（不使用微调权重）
  4. 新增 --max-samples 参数用于冒烟测试
  5. metadata 列：case_id / gender / true_age / image_path / embedding_file
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
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

PATCH_H = 75
PATCH_W = 50
DEFAULT_FEAT_DIM = 1536
DEFAULT_BATCH_SIZE = 16
DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZE_STD = (0.229, 0.224, 0.225)

# ── 路径常量 ────────────────────────────────────────────────────────────────
LIM_ROOT = Path(__file__).resolve().parent.parent            # LIM/
DATA_ROOT = (LIM_ROOT.parent / "LIM_data").resolve()         # LIM_data/

# ── 模型映射表 ────────────────────────────────────────────────────────────────
DINOV2_MODELS = {
    "dinov2_vits14": {"feat_dim": 384,  "default_batch": 64,  "filename": "dinov2_vits14_pretrain.pth"},
    "dinov2_vitb14": {"feat_dim": 768,  "default_batch": 32,  "filename": "dinov2_vitb14_pretrain.pth"},
    "dinov2_vitl14": {"feat_dim": 1024, "default_batch": 16,  "filename": "dinov2_vitl14_pretrain.pth"},
    "dinov2_vitg14": {"feat_dim": 1536, "default_batch": 8,   "filename": "dinov2_vitg14_pretrain.pth"},
}
DINOV2_WEIGHT_DIR = (LIM_ROOT.parent / "LIM_data" / "dinov2_weights").resolve()

DATASET_PRESETS = {
    "rhpe": {
        "csv": "rhpe/labels.csv",
        "img_dir": "rhpe/images",
        "default_output": "embeddings/rhpe/dinov2_vitg",
    },
    "rsna": {
        "csv": "rsna/labels.csv",
        "img_dir": "rsna/images",
        "default_output": "embeddings/rsna/dinov2_vitg",
    },
    "weixin": {
        "csv": None,
        "img_dir": "weixin_imgs",
        "default_output": "embeddings/weixin/dinov2_vitg",
    },
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _resolve_device(use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    normalized = amp_dtype.lower()
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 AMP is not supported on this CUDA device. Use --amp-dtype fp16.")
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}. Available: fp16, bf16")


def _resolve_path(path_value: str | Path | None) -> Path | None:
    """Resolve a path relative to DATA_ROOT (for data / weights)."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (DATA_ROOT / path).resolve()


def _parse_float_tuple(raw_value: str, expected_length: int | None = None) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw_value.split(",") if part.strip())
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"Expected {expected_length} values, got {len(values)}: {raw_value}")
    return values


def build_transform(args: argparse.Namespace) -> T.Compose:
    steps: list[Any] = []

    if not args.no_blur:
        steps.append(T.GaussianBlur(args.blur_kernel_size, sigma=(args.blur_sigma_min, args.blur_sigma_max)))

    steps.extend(
        [
            T.Resize((args.resize_h, args.resize_w)),
            T.CenterCrop((args.crop_h, args.crop_w)),
            T.ToTensor(),
            T.Normalize(mean=args.normalize_mean, std=args.normalize_std),
        ]
    )

    return T.Compose(steps)


def _extract_backbone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    backbone_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("dinov2."):
            backbone_state_dict[key.removeprefix("dinov2.")] = value
    return backbone_state_dict


def _is_viable_dinov2_repo(repo_dir: Path) -> bool:
    hubconf = repo_dir / "hubconf.py"
    backbones = repo_dir / "dinov2" / "hub" / "backbones.py"
    if not hubconf.exists() or not backbones.exists():
        return False
    try:
        content = backbones.read_text(encoding="utf-8")
    except Exception:
        return False
    return "dinov2_vitg14" in content


def _load_backbone(device: torch.device, model_name: str = "dinov2_vitg14", weight_path: Path | None = None) -> nn.Module:
    candidate_repos: list[Path] = [
        DATA_ROOT / "dinov2",
        Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main",
    ]

    backbone: nn.Module | None = None
    errors: list[str] = []

    for repo in candidate_repos:
        if not _is_viable_dinov2_repo(repo):
            continue
        try:
            backbone = torch.hub.load(str(repo), model_name, source="local", pretrained=False)
            print(f"Loaded DINOv2 backbone '{model_name}' from local repo: {repo}")
            break
        except Exception as exc:
            errors.append(f"local load failed from {repo}: {exc}")

    if backbone is None:
        # Bypass proxy/GitHub rate-limit issues that can block hub download
        _saved_env: dict[str, str | None] = {}
        _orig_validate = None
        try:
            for _key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
                _saved_env[_key] = os.environ.pop(_key, None)
            # Skip the forked-repo check (can fail under rate limiting)
            import torch.hub as _hub
            _orig_validate = getattr(_hub, "_validate_not_a_forked_repo", None)
            _hub._validate_not_a_forked_repo = lambda *a, **kw: None
            backbone = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False, trust_repo=True)
            print(f"Loaded DINOv2 backbone '{model_name}' from remote torch.hub repository")
        except Exception as exc:
            errors.append(f"remote load failed: {exc}")
            detail = "\n".join(errors) if errors else "no extra detail"
            raise RuntimeError(f"Unable to load DINOv2 backbone '{model_name}'. Details:\n{detail}") from exc
        finally:
            if _orig_validate is not None:
                _hub._validate_not_a_forked_repo = _orig_validate
            for _key, _val in _saved_env.items():
                if _val is not None:
                    os.environ[_key] = _val

    # 加载预训练权重（优先本地文件）
    if weight_path is not None and weight_path.exists():
        checkpoint = torch.load(weight_path, map_location="cpu")
        state_dict = _extract_backbone_state_dict(checkpoint)
        if not state_dict:
            # 可能是完整 checkpoint 而非纯 state_dict
            state_dict = {k.removeprefix("dinov2."): v for k, v in checkpoint.items() if k.startswith("dinov2.")}
        if state_dict:
            missing, _ = backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from: {weight_path}")
            if missing:
                print(f"[WARN] Missing keys: {len(missing)}")
        else:
            print(f"[WARN] No dinov2.* keys found in {weight_path}, using random init")
    elif weight_path is not None:
        print(f"[WARN] Weight file not found: {weight_path}, using random init")
    else:
        # 尝试从默认位置加载
        weight_filename = DINOV2_MODELS.get(model_name, {}).get("filename", f"{model_name}_pretrain.pth")
        default_weight_path = DINOV2_WEIGHT_DIR / weight_filename
        if default_weight_path.exists():
            checkpoint = torch.load(default_weight_path, map_location="cpu")
            state_dict = {k.removeprefix("dinov2."): v for k, v in checkpoint.items() if k.startswith("dinov2.")}
            if not state_dict:
                state_dict = checkpoint
            missing, _ = backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from default path: {default_weight_path}")
            if missing:
                print(f"[WARN] Missing keys: {len(missing)}")
        else:
            print(f"[WARN] No pretrained weights found for {model_name}, using random init")

    backbone = backbone.to(device)
    backbone.eval()
    return backbone


class VisionLibraryDataset(Dataset):
    def __init__(self, csv_path: Path | None, img_dir: Path, transform=None, dataset_name: str | None = None):
        self.img_dir = img_dir
        self.transform = transform
        self.dataset_name = (dataset_name or "").strip().lower()
        self.rows: list[dict[str, Any]] = []

        if csv_path is not None:
            df = pd.read_csv(csv_path, dtype={"id": str})
            self.rows = self._rows_from_dataframe(df)
        else:
            self.rows = self._rows_from_directory(img_dir)

    @staticmethod
    def _normalize_gender(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "unknown"
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "m", "male"}:
            return "M"
        if text in {"0", "false", "f", "female"}:
            return "F"
        return str(value)

    def _rows_from_dataframe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        aliases = {
            "case_id": ["case_id", "id", "ID", "Case ID"],
            "gender": ["male", "Male", "sex", "Sex", "gender", "Gender"],
            "true_age": ["boneage", "Boneage", "BoneAge", "Chronological", "chronological", "age", "Age"],
            "image_path": ["image_path", "path", "img_path"],
        }

        found: dict[str, str | None] = {}
        for target, candidates in aliases.items():
            found[target] = next((c for c in candidates if c in df.columns), None)

        rows: list[dict[str, Any]] = []
        skipped = 0
        for _, row in df.iterrows():
            case_id = str(row[found["case_id"]]) if found["case_id"] else str(row.get("id", ""))
            gender = self._normalize_gender(row[found["gender"]]) if found["gender"] else "unknown"
            true_age = row[found["true_age"]] if found["true_age"] else None
            image_path_value = row[found["image_path"]] if found["image_path"] else None

            # RHPE data commonly uses zero-padded filenames like 00001.png.
            default_image_path = self.img_dir / f"{case_id}.png"
            if self.dataset_name == "rhpe" and case_id.isdigit():
                rhpe_candidates = [
                    self.img_dir / f"{int(case_id):05d}.png",
                    self.img_dir / f"{case_id}.png",
                ]
                default_image_path = next((candidate for candidate in rhpe_candidates if candidate.exists()), rhpe_candidates[0])

            final_image_path = str(image_path_value) if image_path_value is not None and not pd.isna(image_path_value) else str(default_image_path)

            # Skip rows whose image file does not exist or is empty
            fp = Path(final_image_path)
            if not fp.exists() or fp.stat().st_size == 0:
                skipped += 1
                continue

            rows.append(
                {
                    "case_id": case_id,
                    "gender": gender,
                    "true_age": None if pd.isna(true_age) else true_age,
                    "image_path": final_image_path,
                }
            )
        if skipped > 0:
            print(f"[WARN] Skipped {skipped} rows with empty/missing image files in {self.dataset_name or 'unknown'} dataset")
        return rows

    def _rows_from_directory(self, img_dir: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        skipped = 0
        for image_path in sorted(img_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                if image_path.stat().st_size == 0:
                    skipped += 1
                    continue
                rows.append(
                    {
                        "case_id": image_path.stem,
                        "gender": "unknown",
                        "true_age": None,
                        "image_path": str(image_path),
                    }
                )
        if skipped > 0:
            print(f"[WARN] Skipped {skipped} empty image files in directory scan")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (DATA_ROOT / image_path).resolve()

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "case_id": row["case_id"],
            "gender": row["gender"],
            "true_age": row["true_age"],
            "image_path": str(image_path),
            "image": image,
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    return {
        "case_id": [item["case_id"] for item in batch],
        "gender": [item["gender"] for item in batch],
        "true_age": [item["true_age"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "image": images,
    }


def _resolve_dataset_paths(dataset: str, csv_path: str | None, img_dir: str | None, output_dir: str | None) -> tuple[Path | None, Path, Path]:
    dataset_key = dataset.lower()
    preset = DATASET_PRESETS.get(dataset_key)
    if preset is None:
        raise ValueError(f"Unsupported dataset: {dataset}. Available: {list(DATASET_PRESETS)}")

    resolved_csv = _resolve_path(csv_path if csv_path is not None else preset["csv"])
    resolved_img_dir = _resolve_path(img_dir if img_dir is not None else preset["img_dir"])

    if output_dir is not None:
        resolved_output_dir = Path(output_dir).resolve()  # relative to CWD
    elif preset["default_output"] is not None:
        resolved_output_dir = (LIM_ROOT / preset["default_output"]).resolve()
    else:
        raise ValueError("output_dir is required")

    if resolved_img_dir is None:
        raise ValueError("img_dir cannot be None")

    return resolved_csv, resolved_img_dir, resolved_output_dir


def build_vision_library(args: argparse.Namespace) -> None:
    csv_path, img_dir, output_dir = _resolve_dataset_paths(args.dataset, args.csv, args.img_dir, args.output_dir)
    weight_path = _resolve_path(args.weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    metadata_dir = output_dir / "metadata"
    indexes_dir = output_dir / "indexes"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(use_cpu=args.cpu)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    transform = build_transform(args)
    backbone = _load_backbone(device=device, model_name=args.model, weight_path=weight_path)
    if use_cuda and args.channels_last:
        backbone = backbone.to(memory_format=torch.channels_last)

    dataset = VisionLibraryDataset(csv_path=csv_path, img_dir=img_dir, transform=transform, dataset_name=args.dataset)
    metadata_path = metadata_dir / f"{args.dataset.lower()}_metadata.csv"

    existing_metadata_rows: list[dict[str, Any]] = []
    existing_case_ids: set[str] = set()
    missing_embedding_from_metadata_count = 0

    if args.overwrite:
        print("Overwrite mode enabled: will rebuild all per-case embeddings and metadata from scratch.")
        for embedding_file in embeddings_dir.glob("*.npy"):
            embedding_file.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
    else:
        if metadata_path.exists():
            existing_df = pd.read_csv(metadata_path, dtype={"case_id": str})
            raw_existing_rows = existing_df.to_dict(orient="records")

            # Keep only rows whose embedding file still exists; missing files will be re-generated.
            for row in raw_existing_rows:
                case_id = str(row.get("case_id", ""))
                embedding_file = row.get("embedding_file")
                if embedding_file is not None and not pd.isna(embedding_file):
                    embedding_path = Path(str(embedding_file))
                    if not embedding_path.is_absolute():
                        embedding_path = (DATA_ROOT / embedding_path).resolve()
                else:
                    embedding_path = embeddings_dir / f"{case_id}.npy"

                if embedding_path.exists():
                    existing_metadata_rows.append(row)
                    existing_case_ids.add(case_id)
                else:
                    missing_embedding_from_metadata_count += 1

            if missing_embedding_from_metadata_count > 0:
                print(
                    f"[WARN] Found {missing_embedding_from_metadata_count} metadata rows with missing embedding files. "
                    "These samples will be reprocessed."
                )

        # Resume can still work if metadata is missing but per-case embeddings already exist.
        existing_case_ids.update(path.stem for path in embeddings_dir.glob("*.npy"))

        original_total = len(dataset.rows)
        dataset.rows = [row for row in dataset.rows if str(row["case_id"]) not in existing_case_ids]
        print(
            f"Resume mode: found {len(existing_case_ids)} existing cases, "
            f"skip {original_total - len(dataset.rows)} cases, remaining {len(dataset.rows)} cases."
        )

    # ── 冒烟测试截断 ────────────────────────────────────────────────────────
    if args.max_samples is not None and args.max_samples > 0:
        original_count = len(dataset.rows)
        dataset.rows = dataset.rows[: args.max_samples]
        print(f"Max samples mode: truncated from {original_count} to {len(dataset.rows)} samples.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        collate_fn=_collate_batch,
    )

    metadata_rows: list[dict[str, Any]] = existing_metadata_rows.copy()
    total_new_samples = len(dataset)

    amp_enabled = bool(use_cuda and not args.no_amp)
    amp_dtype = _resolve_amp_dtype(args.amp_dtype) if amp_enabled else None

    with torch.inference_mode():
        pbar = tqdm(
            total=total_new_samples,
            desc=f"Building {args.dataset.upper()} vision library",
            unit="img",
            dynamic_ncols=True,
            mininterval=0.2,
        )
        processed = 0
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            batch_start = time.perf_counter()
            images = batch["image"]
            if use_cuda and args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            images = images.to(device, non_blocking=use_cuda)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled and amp_dtype is not None else nullcontext()
            with autocast_ctx:
                features = backbone.forward_features(images)["x_norm_clstoken"]
            batch_embeddings = features.detach().cpu().numpy().astype(np.float32)

            for i in range(len(batch["case_id"])):
                case_id = str(batch["case_id"][i])
                embedding_path = embeddings_dir / f"{case_id}.npy"
                np.save(embedding_path, batch_embeddings[i])
                metadata_rows.append(
                    {
                        "case_id": case_id,
                        "gender": batch["gender"][i],
                        "true_age": batch["true_age"][i],
                        "image_path": batch["image_path"][i],
                        "embedding_file": str(embedding_path),
                    }
                )
            processed += len(batch["case_id"])
            pbar.update(len(batch["case_id"]))
            pbar.set_postfix_str(f"batch {batch_idx}/{total_batches}")
            pbar.refresh()

            if args.log_every_n_batches > 0 and (batch_idx % args.log_every_n_batches == 0 or processed == total_new_samples):
                batch_seconds = time.perf_counter() - batch_start
                percent = 100.0 if total_new_samples == 0 else (processed / total_new_samples * 100.0)
                tqdm.write(
                    f"[progress] {processed}/{total_new_samples} imgs ({percent:.1f}%), "
                    f"batch {batch_idx}/{total_batches}, batch_time={batch_seconds:.2f}s"
                )
        pbar.close()

    if not metadata_rows:
        raise ValueError(f"No images found for dataset '{args.dataset}' in {img_dir}")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_path, index=False, quoting=csv.QUOTE_MINIMAL)
    num_embedding_files = len(list(embeddings_dir.glob("*.npy")))

    summary = {
        "dataset": args.dataset,
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
        "channels_last": bool(args.channels_last),
        "num_samples": int(len(metadata_df)),
        "existing_samples": int(len(existing_case_ids)) if not args.overwrite else 0,
        "missing_embeddings_reprocessed": int(missing_embedding_from_metadata_count),
        "new_samples": int(total_new_samples),
        "resume_enabled": bool(not args.overwrite),
        "embedding_mode": "per_case_npy",
        "num_embedding_files": int(num_embedding_files),
        "embeddings_dir": str(embeddings_dir),
        "metadata_path": str(metadata_path),
        "img_dir": str(img_dir),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "preprocess": {
            "no_blur": bool(args.no_blur),
            "blur_kernel_size": int(args.blur_kernel_size),
            "blur_sigma": [float(args.blur_sigma_min), float(args.blur_sigma_max)],
            "resize": [int(args.resize_h), int(args.resize_w)],
            "crop": [int(args.crop_h), int(args.crop_w)],
            "normalize_mean": list(args.normalize_mean),
            "normalize_std": list(args.normalize_std),
        },
    }
    with open(output_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DINOv2 visual library for RHPE / RSNA / Weixin")
    parser.add_argument("--model", choices=sorted(DINOV2_MODELS.keys()), default="dinov2_vitg14", help="DINOv2 model variant")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), required=True, help="Dataset name")
    parser.add_argument("--csv", default=None, help="Optional metadata CSV path")
    parser.add_argument("--img-dir", default=None, help="Image directory")
    parser.add_argument("--output-dir", default=None, help="Visual library output directory")
    parser.add_argument("--weights", default=None, help="Custom checkpoint path containing dinov2.* weights")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing outputs and rebuild from scratch")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit to N samples for smoke testing")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Use CPU only")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP when using CUDA")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16", help="AMP dtype for CUDA")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory format for CUDA")
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=1,
        help="Print textual progress every N batches (0 disables batch logs)",
    )
    parser.add_argument("--no-blur", action="store_true", help="Disable Gaussian blur during preprocessing")
    parser.add_argument("--blur-kernel-size", type=int, default=9, help="Gaussian blur kernel size")
    parser.add_argument("--blur-sigma-min", type=float, default=0.1, help="Gaussian blur minimum sigma")
    parser.add_argument("--blur-sigma-max", type=float, default=2.0, help="Gaussian blur maximum sigma")
    parser.add_argument("--resize-h", type=int, default=PATCH_H * 14, help="Resize height")
    parser.add_argument("--resize-w", type=int, default=PATCH_W * 14, help="Resize width")
    parser.add_argument("--crop-h", type=int, default=PATCH_H * 14, help="Center crop height")
    parser.add_argument("--crop-w", type=int, default=PATCH_W * 14, help="Center crop width")
    parser.add_argument(
        "--normalize-mean",
        type=lambda raw: _parse_float_tuple(raw, expected_length=3),
        default=DEFAULT_NORMALIZE_MEAN,
        help="Normalization mean, comma-separated three values",
    )
    parser.add_argument(
        "--normalize-std",
        type=lambda raw: _parse_float_tuple(raw, expected_length=3),
        default=DEFAULT_NORMALIZE_STD,
        help="Normalization std, comma-separated three values",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_vision_library(args)


if __name__ == "__main__":
    main()