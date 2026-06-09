"""
骨龄项目 — 数据路径常量与完整性检查工具
======================================
所有重要路径和检查函数集中在此，供其他脚本 import 使用。

Usage:
    python LIM/code/utils/data_utils.py          # 执行全部检查
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# 本文件位置: LIM/code/utils/data_utils.py → LIM 根目录
SCRIPT_DIR = Path(__file__).resolve().parent  # LIM/code/utils
LIM_ROOT = SCRIPT_DIR.parent.parent           # LIM/

# LIM/ 与 骨龄/、bone/ 平级，数据位于 骨龄/bone/ 下
CODE_ROOT = LIM_ROOT.parent / "骨龄"            # 骨龄/（代码+数据仓库）
REPO_ROOT = CODE_ROOT

# ── RSNA ──────────────────────────────────────────────────────────────────
RSNA_TRAIN_CSV = REPO_ROOT / "bone" / "boneage-training-dataset.csv"
# RSNA 图片实际在嵌套子目录 boneage-training-dataset 内
RSNA_IMG_DIR    = REPO_ROOT / "bone" / "boneage-training-dataset" / "boneage-training-dataset"

# ── RHPE ──────────────────────────────────────────────────────────────────
RHPE_TRAIN_CSV  = REPO_ROOT / "bone" / "RHPE_Annotations" / "RHPE_Annotations" / "BONEAGE" / "boneage_train.csv"
RHPE_VAL_CSV    = REPO_ROOT / "bone" / "RHPE_Annotations" / "RHPE_Annotations" / "BONEAGE" / "boneage_val.csv"
# 注意: boneage_test.csv 不存在，测试标签在 gender_test.csv（无 Boneage 列）
RHPE_TEST_CSV   = REPO_ROOT / "bone" / "RHPE_Annotations" / "RHPE_Annotations" / "BONEAGE" / "boneage_test.csv"
RHPE_IMG_TRAIN  = REPO_ROOT / "bone" / "RHPE_train" / "images"
RHPE_IMG_VAL    = REPO_ROOT / "bone" / "RHPE_val" / "images"
RHPE_IMG_TEST   = REPO_ROOT / "bone" / "RHPE_test" / "images"

# ── Embedding / model / output 目录 ────────────────────────────────────────
EMB_ROOT   = REPO_ROOT / "bone" / "library"
EXP_ROOT   = REPO_ROOT / "bone" / "code" / "train" / "MLP_weight"
GEN_ROOT   = REPO_ROOT / "bone" / "code" / "result"
FIG_ROOT   = LIM_ROOT / "results" / "figures"

# ── Metadata 路径 ──────────────────────────────────────────────────────────
RHPE_METADATA_CSV = (
    REPO_ROOT
    / "bone"
    / "library"
    / "RHPE"
    / "vision_library"
    / "metadata"
    / "rhpe_metadata.csv"
)

# ---------------------------------------------------------------------------
# Integrity check helpers
# ---------------------------------------------------------------------------


def check_dataset_integrity(
    csv_path: Path,
    img_dir: Path,
    id_col: str = "id",
    id_fmt: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    检查一个数据集分片（train／val／test）的完整性：

    * CSV 文件是否存在、是否为空
    * CSV 中所有 ID 是否都能在 img_dir 中找到对应图片
    * 如果 id_fmt 指定（如 ``"{}.png"``），自动拼接后再检查

    返回 ``{"csv_ok": bool, "img_ok": bool, "missing": list}``。
    """
    import pandas as pd  # lazy import 避免启动时过慢

    result: dict = {"csv_ok": False, "img_ok": False, "missing": [], "n_total": 0}

    # ── 1. 检查 CSV ──────────────────────────────────────────────────────
    if not csv_path.exists():
        if verbose:
            print(f"  [FAIL] CSV 不存在: {csv_path}")
        return result
    if csv_path.stat().st_size == 0:
        if verbose:
            print(f"  [FAIL] CSV 为空: {csv_path}")
        return result

    df = pd.read_csv(csv_path)
    result["n_total"] = len(df)
    result["csv_ok"] = True
    if verbose:
        print(f"  [OK]   CSV 存在且非空 ({len(df)} 行)")

    # ── 2. 检查 ID 列 ────────────────────────────────────────────────────
    if id_col not in df.columns:
        if verbose:
            print(f"  [FAIL] CSV 缺少 ID 列 '{id_col}' (现有: {list(df.columns)})")
        return result

    ids = df[id_col].astype(str).str.strip()

    # ── 3. 检查图片文件存在性 ─────────────────────────────────────────────
    missing = []
    for idx in ids:
        if id_fmt:
            try:
                fname = id_fmt.format(int(idx))
            except ValueError:
                fname = id_fmt.format(idx)
        else:
            fname = idx
        candidate = img_dir / fname
        if not candidate.exists():
            missing.append(fname)

    if missing:
        n = len(missing)
        result["missing"] = missing[:20]  # 最多报告前 20 个
        if verbose:
            print(f"  [FAIL] 缺失 {n}/{len(ids)} 张图片 (显示前 {min(n, 20)}):")
            for m in missing[:20]:
                print(f"         {m}")
    else:
        result["img_ok"] = True
        if verbose:
            print(f"  [OK]   全部 {len(ids)} 张图片均存在")

    return result


# ---------------------------------------------------------------------------
# Main: 依次检查所有数据集分片
# ---------------------------------------------------------------------------


def run_all_checks(verbose: bool = True) -> list[dict]:
    """
    对所有已知数据集分片依次执行完整性检查。

    Parameters
    ----------
    verbose : bool
        是否打印逐项检查结果。

    Returns
    -------
    list[dict]
        每个分片的检查结果字典。
    """
    checks = [
        ("RSNA train", RSNA_TRAIN_CSV, RSNA_IMG_DIR, "id", "{}.png"),
        ("RHPE train", RHPE_TRAIN_CSV, RHPE_IMG_TRAIN, "ID", "{:05d}.png"),
        ("RHPE val", RHPE_VAL_CSV, RHPE_IMG_VAL, "ID", "{:05d}.png"),
        ("RHPE test (labels missing)", RHPE_TEST_CSV, RHPE_IMG_TEST, "ID", "{:05d}.png"),
    ]

    all_results = []
    for name, csv, img, idc, fmt in checks:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  检查: {name}")
            print(f"  CSV : {csv}")
            print(f"  Img : {img}")
            print(f"{'='*60}")
        res = check_dataset_integrity(
            csv_path=csv,
            img_dir=img,
            id_col=idc,
            id_fmt=fmt,
            verbose=verbose,
        )
        res["name"] = name
        all_results.append(res)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"  汇总")
        print(f"{'='*60}")
        n_ok = sum(1 for r in all_results if r["csv_ok"] and r["img_ok"])
        print(f"  {n_ok}/{len(all_results)} 分片通过完整性检查\n")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_checks(verbose=True)