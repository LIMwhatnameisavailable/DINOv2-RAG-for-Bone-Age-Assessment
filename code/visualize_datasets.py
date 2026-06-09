"""LIM/code/visualize_datasets.py
2×2 综合数据集分布图：RSNA vs RHPE
输出：LIM/figures/dataset_overview.png (200 DPI)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── 先导入并应用全局样式 ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from vis_style import apply_style, COLORS, style_axes, FIGSIZE

apply_style()

# ── 路径 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # LIM/code/
LIM_ROOT   = SCRIPT_DIR.parent                        # LIM/
DATA_ROOT  = (LIM_ROOT / ".." / "骨龄" / "bone").resolve()  # 骨龄/bone/

RSNA_CSV      = DATA_ROOT / "boneage-training-dataset.csv"
RHPE_TRAIN_CSV = (
    DATA_ROOT / "RHPE_Annotations" / "RHPE_Annotations"
    / "BONEAGE" / "boneage_train.csv"
)
RHPE_VAL_CSV   = (
    DATA_ROOT / "RHPE_Annotations" / "RHPE_Annotations"
    / "BONEAGE" / "boneage_val.csv"
)

# ── 读取数据 ──────────────────────────────────────────────────────────────
# RSNA
try:
    rsna_df = pd.read_csv(RSNA_CSV)
except Exception as e:
    print(f"[ERROR] RSNA CSV read failed: {e}")
    print(f"  path: {RSNA_CSV}")
    sys.exit(1)

# 打印实际列名以便调试
print(f"RSNA columns: {list(rsna_df.columns)}")

# RHPE（训练集+验证集合并）
try:
    rhpe_train = pd.read_csv(RHPE_TRAIN_CSV)
    rhpe_val   = pd.read_csv(RHPE_VAL_CSV)
    print(f"RHPE train columns: {list(rhpe_train.columns)}")
    print(f"RHPE val columns:   {list(rhpe_val.columns)}")
    rhpe_df = pd.concat([rhpe_train, rhpe_val], ignore_index=True)
except Exception as e:
    print(f"[ERROR] RHPE CSV read failed: {e}")
    sys.exit(1)

# ── 列名映射 ──────────────────────────────────────────────────────────────
# RSNA: id, boneage, male
# RHPE: ID, Male, Boneage, Chronological

def _to_bool(val):
    """Convert various male formats to boolean (True=男)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "t", "1", "male")

# ── 提取字段 ──────────────────────────────────────────────────────────────
# RSNA
rsna_ages = rsna_df["boneage"].dropna().astype(float).values
rsna_male = rsna_df["male"].dropna().apply(_to_bool)
rsna_n    = len(rsna_ages)
rsna_mean = float(np.mean(rsna_ages))
rsna_std  = float(np.std(rsna_ages))
rsna_male_count   = int(rsna_male.sum())
rsna_female_count = int((~rsna_male).sum())

# RHPE
rhpe_ages = rhpe_df["Boneage"].dropna().astype(float).values
rhpe_male = rhpe_df["Male"].dropna().apply(_to_bool)
rhpe_n    = len(rhpe_ages)
rhpe_mean = float(np.mean(rhpe_ages))
rhpe_std  = float(np.std(rhpe_ages))
rhpe_male_count   = int(rhpe_male.sum())
rhpe_female_count = int((~rhpe_male).sum())

# ── 颜色 ──────────────────────────────────────────────────────────────────
RSNA_COLOR   = COLORS["rsna"]    # "#457B9D" 钢蓝
RHPE_COLOR   = "#E67E22"         # 橙色（RHPE 用橙色系）
MALE_COLOR   = COLORS["male"]    # "#457B9D"
FEMALE_COLOR = COLORS["female"]  # "#E64B35"

# ── 绘图 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["grid_2x2"])  # (10, 8)
fig.suptitle("Dataset Overview: RSNA vs RHPE",
             fontsize=16, fontweight="bold", y=0.98)

# ── [0,0] RSNA 骨龄分布 ─────────────────────────────────────────────────
ax = axes[0, 0]
ax.hist(rsna_ages, bins=40, density=True, alpha=0.35,
        color=RSNA_COLOR, label=f"RSNA (n={rsna_n})")
# KDE
kde_x = np.linspace(rsna_ages.min(), rsna_ages.max(), 500)
kde   = stats.gaussian_kde(rsna_ages)
ax.plot(kde_x, kde(kde_x), color=RSNA_COLOR, linewidth=2, label="KDE")
ax.axvline(rsna_mean, color=RSNA_COLOR, linestyle="--", linewidth=1,
           label=f"Mean={rsna_mean:.1f}")
ax.set_xlabel("Age (months)")
ax.set_ylabel("Density")
ax.set_title("RSNA Bone Age Distribution")
style_axes(ax)
ax.legend(fontsize=8, frameon=False)

# ── [0,1] RHPE 骨龄分布 ────────────────────────────────────────────────
ax = axes[0, 1]
ax.hist(rhpe_ages, bins=40, density=True, alpha=0.35,
        color=RHPE_COLOR, label=f"RHPE (n={rhpe_n})")
kde_x = np.linspace(rhpe_ages.min(), rhpe_ages.max(), 500)
kde   = stats.gaussian_kde(rhpe_ages)
ax.plot(kde_x, kde(kde_x), color=RHPE_COLOR, linewidth=2, label="KDE")
ax.axvline(rhpe_mean, color=RHPE_COLOR, linestyle="--", linewidth=1,
           label=f"Mean={rhpe_mean:.1f}")
ax.set_xlabel("Age (months)")
ax.set_ylabel("Density")
ax.set_title("RHPE Bone Age Distribution")
style_axes(ax)
ax.legend(fontsize=8, frameon=False)

# ── [1,0] RSNA 性别比例饼图 ────────────────────────────────────────────
def make_autopct(count):
    def _autopct(pct):
        n = int(round(pct / 100.0 * count))
        return f"{pct:.1f}%\n(n={n})"
    return _autopct

ax = axes[1, 0]
ax.pie(
    [rsna_female_count, rsna_male_count],
    labels=["Female", "Male"],
    autopct=make_autopct(rsna_n),
    colors=[FEMALE_COLOR, MALE_COLOR],
    startangle=90,
    textprops={"fontsize": 10},
)
ax.set_title("RSNA Gender Ratio")

# ── [1,1] RHPE 性别比例饼图 ────────────────────────────────────────────
ax = axes[1, 1]
ax.pie(
    [rhpe_female_count, rhpe_male_count],
    labels=["Female", "Male"],
    autopct=make_autopct(rhpe_n),
    colors=[FEMALE_COLOR, MALE_COLOR],
    startangle=90,
    textprops={"fontsize": 10},
)
ax.set_title("RHPE Gender Ratio")

# ── 保存 ────────────────────────────────────────────────────────────────
out_path = LIM_ROOT / "figures" / "dataset_overview.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留空间
plt.savefig(out_path, dpi=200, facecolor="white")
print(f"\nFigure saved to: {out_path}")

# ── 报告统计数据 ────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Dataset Statistics Report")
print(f"{'='*55}")
print(f"  RSNA:")
print(f"    Samples : {rsna_n}")
print(f"    Age     : {rsna_mean:.2f} ± {rsna_std:.2f} months")
print(f"    Range   : {rsna_ages.min():.0f} – {rsna_ages.max():.0f} months")
print(f"    Male    : {rsna_male_count} ({rsna_male_count/rsna_n*100:.1f}%)")
print(f"    Female  : {rsna_female_count} ({rsna_female_count/rsna_n*100:.1f}%)")
print(f"  RHPE (train + val):")
print(f"    Samples : {rhpe_n}")
print(f"    Age     : {rhpe_mean:.2f} ± {rhpe_std:.2f} months")
print(f"    Range   : {rhpe_ages.min():.0f} – {rhpe_ages.max():.0f} months")
print(f"    Male    : {rhpe_male_count} ({rhpe_male_count/rhpe_n*100:.1f}%)")
print(f"    Female  : {rhpe_female_count} ({rhpe_female_count/rhpe_n*100:.1f}%)")
print(f"{'='*55}")
