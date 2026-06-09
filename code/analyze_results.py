"""
LIM/code/analyze_results.py

汇总 24 个 MLP 实验结果并生成可视化图表。

用法:
  # 汇总所有实验到 summary.csv
  conda run -n boneage python LIM/code/analyze_results.py --mode summary

  # 生成全部 7 张图（需 summary.csv 已存在）
  conda run -n boneage python LIM/code/analyze_results.py --mode figures

  # 一步完成
  conda run -n boneage python LIM/code/analyze_results.py --mode all

图表清单（见 myproject_instruction.md 第 8 节）:
  Fig 1: backbone_comparison.png   — 6 Backbone × 性别消融
  Fig 2: loss_ablation.png         — 6 Backbone × 3 损失函数
  Fig 3: gender_analysis.png       — 分性别误差分析
  Fig 4: age_error_heatmap.png     — 年龄段误差热力图
  Fig 5: gw_mae_weight_curve.png   — GW-MAE 权重函数曲线
  Fig 6: generalization_comparison.png — RSNA vs RHPE 泛化对比
  Fig 7: error_distribution_by_age_group.png — 分年龄段误差分布
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
LIM_ROOT   = SCRIPT_DIR.parent
DATA_ROOT  = (LIM_ROOT / ".." / "data").resolve()

sys.path.insert(0, str(SCRIPT_DIR / "utils"))
from vis_style import apply_style, COLORS, FIGSIZE, style_axes


EXPERIMENTS_DIR = LIM_ROOT / "experiments"
FIGURES_DIR     = LIM_ROOT / "figures"
SUMMARY_CSV     = EXPERIMENTS_DIR / "summary.csv"

# ── 实验矩阵定义 ──
EXPERIMENTS: list[dict] = [
    # (exp_id, backbone, loss, gender)
    dict(id="E01", backbone="resnet50",        loss="mae",      gender=False),
    dict(id="E02", backbone="resnet50",        loss="mae",      gender=True),
    dict(id="E03", backbone="resnet50",        loss="gwmae",    gender=True),
    dict(id="E04", backbone="resnet50",        loss="combined", gender=True),
    dict(id="E05", backbone="efficientnet_b4", loss="mae",      gender=False),
    dict(id="E06", backbone="efficientnet_b4", loss="mae",      gender=True),
    dict(id="E07", backbone="efficientnet_b4", loss="gwmae",    gender=True),
    dict(id="E08", backbone="efficientnet_b4", loss="combined", gender=True),
    dict(id="E09", backbone="dinov2_vits",     loss="mae",      gender=False),
    dict(id="E10", backbone="dinov2_vits",     loss="mae",      gender=True),
    dict(id="E11", backbone="dinov2_vits",     loss="gwmae",    gender=True),
    dict(id="E12", backbone="dinov2_vits",     loss="combined", gender=True),
    dict(id="E13", backbone="dinov2_vitb",     loss="mae",      gender=False),
    dict(id="E14", backbone="dinov2_vitb",     loss="mae",      gender=True),
    dict(id="E15", backbone="dinov2_vitb",     loss="gwmae",    gender=True),
    dict(id="E16", backbone="dinov2_vitb",     loss="combined", gender=True),
    dict(id="E17", backbone="dinov2_vitl",     loss="mae",      gender=False),
    dict(id="E18", backbone="dinov2_vitl",     loss="mae",      gender=True),
    dict(id="E19", backbone="dinov2_vitl",     loss="gwmae",    gender=True),
    dict(id="E20", backbone="dinov2_vitl",     loss="combined", gender=True),
    dict(id="E21", backbone="dinov2_vitg",     loss="mae",      gender=False),
    dict(id="E22", backbone="dinov2_vitg",     loss="mae",      gender=True),
    dict(id="E23", backbone="dinov2_vitg",     loss="gwmae",    gender=True),
    dict(id="E24", backbone="dinov2_vitg",     loss="combined", gender=True),
]

BACKBONE_ORDER = ["resnet50", "efficientnet_b4", "dinov2_vits",
                  "dinov2_vitb", "dinov2_vitl", "dinov2_vitg"]
BACKBONE_LABELS = {
    "resnet50": "ResNet-50", "efficientnet_b4": "EffNet-B4",
    "dinov2_vits": "ViT-S", "dinov2_vitb": "ViT-B",
    "dinov2_vitl": "ViT-L", "dinov2_vitg": "ViT-g",
}
FEAT_DIMS = {
    "resnet50": 2048, "efficientnet_b4": 1792,
    "dinov2_vits": 384, "dinov2_vitb": 768,
    "dinov2_vitl": 1024, "dinov2_vitg": 1536,
}
LOSS_LABELS = {"mae": "MAE", "gwmae": "GW-MAE", "combined": "MAE+GW-MAE"}


# ============================================================================
# 汇总
# ============================================================================

def _find_exp_dir(exp_id: str) -> Path:
    """Find experiment directory by ID prefix (handles E01 vs E01_resnet50_mae_nogender)."""
    # Try exact match first
    exact = EXPERIMENTS_DIR / exp_id
    if exact.is_dir():
        return exact
    # Try prefix match
    for p in sorted(EXPERIMENTS_DIR.iterdir()):
        if p.is_dir() and p.name.startswith(exp_id + "_"):
            return p
    return exact


def build_summary() -> pd.DataFrame:
    """扫描 experiments/ 目录，汇总 metrics.json 到 summary.csv"""
    rows = []
    for exp in EXPERIMENTS:
        exp_dir = _find_exp_dir(exp["id"])
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"  ⚠️  Missing: {metrics_path}")
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = m.get("test", {})
        row = {
            "experiment_id": exp["id"],
            "backbone": exp["backbone"],
            "loss": exp["loss"],
            "use_gender": exp["gender"],
            "test_mae": test.get("mae", None),
            "test_gw_mae": test.get("gw_mae", None),
            "test_rmse": test.get("rmse", None),
            "test_median_ae": test.get("median_ae", None),
            "test_num_samples": test.get("num_samples", None),
            "test_r2": test.get("r2", None),
        }
        # 分性别
        m_by_g = test.get("mae_by_gender", {})
        for g in ("M", "F"):
            row[f"test_mae_{g}"] = m_by_g.get(g, None)
        # 分年龄段
        m_by_a = test.get("mae_by_age_group", {})
        for key in ("0_60", "60_120", "120_180", "180_228"):
            row[f"test_mae_{key}"] = m_by_a.get(key, None)
        rows.append(row)
        print(f"  {exp['id']}: MAE={row['test_mae']}")

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary saved to: {SUMMARY_CSV} ({len(df)} experiments)")
    return df


# ============================================================================
# 图表
# ============================================================================

def fig1_backbone_comparison(df: pd.DataFrame) -> None:
    """Fig 1: Backbone 性能对比 — 分组条形图 (有性别 vs 无性别, MAE loss)"""
    import matplotlib.pyplot as plt

    # 筛选 MAE loss 的实验
    sub = df[(df["loss"] == "mae")].copy()
    sub["label"] = sub["backbone"].map(BACKBONE_LABELS)

    fig, ax = plt.subplots(figsize=FIGSIZE["wide_1x2"])
    x = np.arange(len(BACKBONE_ORDER))
    w = 0.35

    nogender_vals = [sub[(sub["backbone"] == b) & (sub["use_gender"] == False)]["test_mae"].values[0]
                     if len(sub[(sub["backbone"] == b) & (sub["use_gender"] == False)]) > 0 else 0
                     for b in BACKBONE_ORDER]
    gender_vals = [sub[(sub["backbone"] == b) & (sub["use_gender"] == True)]["test_mae"].values[0]
                   if len(sub[(sub["backbone"] == b) & (sub["use_gender"] == True)]) > 0 else 0
                   for b in BACKBONE_ORDER]

    ax.bar(x - w/2, nogender_vals, w, label="No Gender", color=COLORS["neutral"])
    ax.bar(x + w/2, gender_vals, w, label="With Gender", color=COLORS["mae"])

    # 标注特征维度
    for i, b in enumerate(BACKBONE_ORDER):
        ax.annotate(f"d={FEAT_DIMS[b]}", (i, max(nogender_vals[i], gender_vals[i]) + 0.3),
                    ha="center", fontsize=9, color=COLORS["text_secondary"])

    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_LABELS[b] for b in BACKBONE_ORDER], fontsize=11)
    ax.set_ylabel(r"$\it{MAE}$ (months)", fontsize=12)
    ax.set_title("Backbone Performance Comparison (MAE Loss)")
    ax.legend(fontsize=11, frameon=False)
    style_axes(ax)

    path = FIGURES_DIR / "backbone_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_loss_ablation(df: pd.DataFrame) -> None:
    """Fig 2: 损失函数消融 — 分组条形图 (MAE / GW-MAE / Combined)"""
    import matplotlib.pyplot as plt

    sub = df[df["use_gender"] == True].copy()
    fig, ax = plt.subplots(figsize=FIGSIZE["wide_1x2"])
    x = np.arange(len(BACKBONE_ORDER))
    w = 0.25

    for i, loss in enumerate(["mae", "gwmae", "combined"]):
        vals = []
        for b in BACKBONE_ORDER:
            m = sub[(sub["backbone"] == b) & (sub["loss"] == loss)]["test_mae"]
            vals.append(m.values[0] if len(m) > 0 else 0)
        ax.bar(x + (i - 1) * w, vals, w, label=LOSS_LABELS[loss], color=COLORS[loss])

    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_LABELS[b] for b in BACKBONE_ORDER], fontsize=11)
    ax.set_ylabel(r"$\it{MAE}$ (months)", fontsize=12)
    ax.set_title("Loss Function Ablation (with Gender)")
    ax.legend(fontsize=11, frameon=False)
    style_axes(ax)

    path = FIGURES_DIR / "loss_ablation.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_gender_analysis(df: pd.DataFrame) -> None:
    """Fig 3: 性别分层误差 — 1×2 子图（柱状图 + 散点图）"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["wide_1x2"])

    # 左图：MAE(M) vs MAE(F)
    ax = axes[0]
    sub = df[df["use_gender"] == True].copy()
    x = np.arange(len(BACKBONE_ORDER))
    w = 0.35
    m_vals = []
    f_vals = []
    for b in BACKBONE_ORDER:
        row = sub[sub["backbone"] == b].iloc[0] if len(sub[sub["backbone"] == b]) > 0 else None
        m_vals.append(row["test_mae_M"] if row is not None and pd.notna(row.get("test_mae_M")) else 0)
        f_vals.append(row["test_mae_F"] if row is not None and pd.notna(row.get("test_mae_F")) else 0)

    ax.bar(x - w/2, m_vals, w, label="Male", color=COLORS["male"])
    ax.bar(x + w/2, f_vals, w, label="Female", color=COLORS["female"])
    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_LABELS[b] for b in BACKBONE_ORDER], fontsize=9)
    ax.set_ylabel(r"$\it{MAE}$ (months)")
    ax.set_title("(A) Gender-stratified MAE")
    ax.legend(fontsize=10, frameon=False)
    style_axes(ax)

    # 右图：散点图 — 取最优模型（Vit-g, combined）
    ax = axes[1]
    exp_row = sub[(sub["backbone"] == "dinov2_vitg") & (sub["loss"] == "combined")]
    if len(exp_row) > 0:
        exp_dir = _find_exp_dir(exp_row.iloc[0]["experiment_id"])
        pred_path = exp_dir / "test_predictions.csv"
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            for g, color in [("M", COLORS["male"]), ("F", COLORS["female"])]:
                gdf = pred_df[pred_df["gender"] == g]
                ax.scatter(gdf["true_age"], gdf["pred_age"], c=color, alpha=0.5, s=10, label=g)
            lims = [0, 240]
            ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
    ax.set_xlabel(r"$\it{True\ Age}$ (months)")
    ax.set_ylabel(r"$\it{Predicted\ Age}$ (months)")
    ax.set_title("(B) Predicted vs True Age (ViT-g, Combined)")
    ax.legend(fontsize=10, frameon=False)
    style_axes(ax)

    path = FIGURES_DIR / "gender_analysis.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_age_error_heatmap(df: pd.DataFrame) -> None:
    """Fig 4: 年龄段误差热力图"""
    import matplotlib.pyplot as plt

    sub = df[df["use_gender"] == True].copy()
    # 对每个 backbone 取 MAE 最低的损失函数配置
    data = []
    age_keys = ["test_mae_0_60", "test_mae_60_120", "test_mae_120_180", "test_mae_180_228"]
    age_labels = ["0-60", "60-120", "120-180", "180-228"]
    for b in BACKBONE_ORDER:
        bdf = sub[sub["backbone"] == b]
        best = bdf.loc[bdf["test_mae"].idxmin()] if len(bdf) > 0 else None
        vals = [best[k] if best is not None and pd.notna(best.get(k)) else 0 for k in age_keys]
        data.append(vals)

    fig, ax = plt.subplots(figsize=FIGSIZE["heatmap"])
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0)

    # 标注数值
    for i in range(len(BACKBONE_ORDER)):
        for j in range(len(age_keys)):
            ax.text(j, i, f"{data[i][j]:.1f}", ha="center", va="center",
                    fontsize=10, color="white" if data[i][j] > np.mean(data) else "black")

    ax.set_xticks(range(len(age_keys)))
    ax.set_xticklabels(age_labels, fontsize=11)
    ax.set_yticks(range(len(BACKBONE_ORDER)))
    ax.set_yticklabels([BACKBONE_LABELS[b] for b in BACKBONE_ORDER], fontsize=11)
    ax.set_xlabel("Age Group (months)")
    ax.set_ylabel("Backbone")
    ax.set_title("Age Group Error Heatmap (Best Config per Backbone)")
    fig.colorbar(im, ax=ax, label="MAE (months)")

    path = FIGURES_DIR / "age_error_heatmap.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_gw_mae_weight_curve() -> None:
    """Fig 5: GW-MAE 权重函数曲线"""
    import matplotlib.pyplot as plt
    import torch

    from gw_mae import GrowthWeightedMAELoss

    gw = GrowthWeightedMAELoss(mu=120.0, sigma=36.0, w_min=0.5, w_max=2.0)
    ages = torch.linspace(0, 240, 241)
    weights = gw.weight(ages)

    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    ax.plot(ages.numpy(), weights.numpy(), color=COLORS["gwmae"], linewidth=2)

    # 标注峰值
    ax.axvline(120, color=COLORS["gwmae"], linestyle="--", alpha=0.5)
    ax.annotate(r"$\mu=120$ months (peak)", xy=(120, 2.0), xytext=(140, 2.1),
                fontsize=10, color=COLORS["text_primary"])

    # 年龄段背景
    for lo, hi, color_key in [(0, 60, "age_0_60"), (60, 120, "age_60_120"),
                               (120, 180, "age_120_180"), (180, 240, "age_180_228")]:
        ax.axvspan(lo, hi, alpha=0.3, color=COLORS[color_key])
        ax.text((lo + hi) / 2, 0.55, f"{lo}-{hi}", ha="center", fontsize=9,
                color=COLORS["text_secondary"])

    ax.set_xlabel(r"$\it{Age}$ (months)")
    ax.set_ylabel(r"$\it{Weight\ w(age)}$")
    ax.set_title("GW-MAE Weight Function")
    ax.set_xlim(0, 240)
    ax.set_ylim(0.4, 2.2)
    style_axes(ax)

    path = FIGURES_DIR / "gw_mae_weight_curve.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_generalization_comparison(df: pd.DataFrame | None = None) -> None:
    """Fig 6: 跨数据集泛化对比 — 1×2 子图

    左图：RSNA 测试最优模型在 RSNA test vs RHPE val 上的 MAE 对比
    右图：该模型在 RHPE 上的年龄段误差分布
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["wide_1x2"])

    # ── 找最优 RSNA 实验（按 test_mae）──
    summary_path = SUMMARY_CSV
    best_row = None
    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        if len(sdf) > 0:
            best_row = sdf.loc[sdf["test_mae"].idxmin()]

    # ── 从 generalization 目录收集 RHPE 结果 ──
    gen_metrics: dict[str, dict] = {}
    gen_root = LIM_ROOT / "generalization"
    if gen_root.is_dir():
        for d in sorted(gen_root.iterdir()):
            if not d.is_dir():
                continue
            m_path = d / "metrics.json"
            if not m_path.exists():
                continue
            parts = d.name.split("_on_rhpe_")
            if len(parts) == 2:
                bb = parts[1]
                gm = json.loads(m_path.read_text(encoding="utf-8"))
                gen_metrics[bb] = gm

    # ── 左图：RSNA test vs RHPE val MAE ──
    ax = axes[0]
    rsna_mae = best_row["test_mae"] if best_row is not None else None
    best_bb = best_row["backbone"] if best_row is not None else None
    rhpe_mae = None
    if best_bb and best_bb in gen_metrics:
        rhpe_mae = gen_metrics[best_bb].get("mae", None)

    labels = []
    vals = []
    bar_colors = []
    if rsna_mae is not None:
        labels.append("RSNA (test)")
        vals.append(rsna_mae)
        bar_colors.append(COLORS["rsna"])
    if rhpe_mae is not None:
        labels.append("RHPE (val)")
        vals.append(rhpe_mae)
        bar_colors.append(COLORS["rhpe"])

    if vals:
        bars = ax.bar(labels, vals, color=bar_colors, width=0.4, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.2f}", ha="center", fontsize=10, color=COLORS["text_primary"])
        ax.set_ylabel(r"$\it{MAE}$ (months)", fontsize=12)
    bb_label = BACKBONE_LABELS.get(best_bb, best_bb) if best_bb else ""
    ax.set_title(f"(A) Cross-dataset Generalization ({bb_label})")
    style_axes(ax)

    # ── 右图：RHPE 年龄段误差 ──
    ax = axes[1]
    if best_bb and best_bb in gen_metrics:
        age_mae = gen_metrics[best_bb].get("mae_by_age_group", {})
        keys = ["0_60", "60_120", "120_180", "180_228"]
        labels2 = ["0-60", "60-120", "120-180", "180-228"]
        vals2 = [age_mae.get(k, 0) for k in keys]
        if any(v > 0 for v in vals2):
            bars = ax.bar(labels2, vals2, color=COLORS["rhpe"], width=0.5, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals2):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.2f}", ha="center", fontsize=10, color=COLORS["text_primary"])
            ax.set_ylabel(r"$\it{MAE}$ (months)", fontsize=12)
            ax.set_title("(B) RHPE Error by Age Group")
        else:
            ax.text(0.5, 0.5, "No age group data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color=COLORS["text_secondary"])
    else:
        ax.text(0.5, 0.5, "No generalization data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=COLORS["text_secondary"])
    style_axes(ax)

    path = FIGURES_DIR / "generalization_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig7_error_distribution(df: pd.DataFrame) -> None:
    """Fig 7: 误差分布直方图 — 2×2 子图"""
    import matplotlib.pyplot as plt

    # 找 MAE 和 GW-MAE 训练的最佳模型
    mae_row = df[(df["loss"] == "mae") & (df["backbone"] == "dinov2_vitg")]
    gw_row = df[(df["loss"] == "gwmae") & (df["backbone"] == "dinov2_vitg")]

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["grid_2x2"])
    age_groups = [(0, 60, "0-60 months"), (60, 120, "60-120 months"),
                  (120, 180, "120-180 months"), (180, 228, "180-228 months")]

    for idx, (lo, hi, title) in enumerate(age_groups):
        ax = axes[idx // 2][idx % 2]
        for label, row, color in [
            ("MAE-trained", mae_row, COLORS["mae"]),
            ("GW-MAE-trained", gw_row, COLORS["gwmae"]),
        ]:
            if len(row) == 0:
                continue
            exp_dir = _find_exp_dir(row.iloc[0]["experiment_id"])
            pred_path = exp_dir / "test_predictions.csv"
            if not pred_path.exists():
                continue
            pdf = pd.read_csv(pred_path)
            subset = pdf[(pdf["true_age"] >= lo) & (pdf["true_age"] < hi)]
            if len(subset) == 0:
                continue
            errors = subset["abs_error"].values
            ax.hist(errors, bins=20, alpha=0.5, color=color, label=label, density=True)

        ax.set_xlabel(r"$\it{|Error|}$ (months)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8, frameon=False)
        style_axes(ax)

    fig.suptitle("Error Distribution by Age Group (ViT-g)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = FIGURES_DIR / "error_distribution_by_age_group.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 主函数
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze LIM experiment results")
    p.add_argument("--mode", choices=["summary", "figures", "all"],
                   default="all", help="What to run")
    return p


def main() -> None:
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    args = build_parser().parse_args()

    df = None
    if args.mode in ("summary", "all"):
        print("Building summary.csv...")
        df = build_summary()

    if args.mode in ("figures", "all"):
        if df is None and SUMMARY_CSV.exists():
            df = pd.read_csv(SUMMARY_CSV)
        elif df is None:
            print("No summary found. Run --mode summary first.")
            return

        print("Generating figures...")
        fig1_backbone_comparison(df)
        fig2_loss_ablation(df)
        fig3_gender_analysis(df)
        fig4_age_error_heatmap(df)
        fig5_gw_mae_weight_curve()
        fig6_generalization_comparison()
        fig7_error_distribution(df)
        print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()