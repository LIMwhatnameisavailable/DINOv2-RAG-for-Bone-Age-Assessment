"""
LIM/code/build_generalization_summary.py

生成跨数据集泛化结果汇总表，保存至 LIM/generalization/generalization_summary.csv
"""
import json
import os
import pandas as pd
from pathlib import Path

LIM_ROOT = Path(__file__).resolve().parent.parent  # LIM/

# RSNA 内部测试集结果（来自 summary.csv，只取 MAE 损失 + 有性别 的行）
summary_csv = LIM_ROOT / "experiments" / "summary.csv"
df_rsna = pd.read_csv(summary_csv)

# 每个 backbone 取 MAE+性别 的实验（E02/E06/E10/E14/E18/E22）
rsna_rows = df_rsna[
    (df_rsna["loss"] == "mae") & (df_rsna["use_gender"] == True)
][["backbone", "experiment_id", "test_mae", "test_rmse", "test_r2"]].copy()
rsna_rows.columns = ["backbone", "rsna_exp_id", "rsna_mae", "rsna_rmse", "rsna_r2"]

# RHPE 泛化结果（从 generalization/ 目录读取）
GEN_ROOT = LIM_ROOT / "generalization"
rhpe_rows = []

for gen_dir in sorted(GEN_ROOT.iterdir()):
    if not gen_dir.is_dir():
        continue
    metrics_path = gen_dir / "metrics.json"
    if not metrics_path.exists():
        continue

    m = json.loads(metrics_path.read_text(encoding="utf-8"))

    # 从目录名解析 backbone（格式：E22_vitg_mae_gender_on_rhpe_dinov2_vitg）
    dir_name = gen_dir.name
    backbone_raw = dir_name.split("_on_rhpe_")[-1] if "_on_rhpe_" in dir_name else dir_name

    rhpe_rows.append({
        "backbone_raw": backbone_raw,
        "gen_dir": dir_name,
        "rhpe_mae": m.get("mae", m.get("test_mae", None)),
        "rhpe_rmse": m.get("rmse", m.get("test_rmse", None)),
        "rhpe_r2": m.get("r2", m.get("test_r2", None)),
        "rhpe_num_samples": m.get("num_samples", m.get("test_num_samples", None)),
    })

df_rhpe = pd.DataFrame(rhpe_rows)
print("=== RHPE 目录扫描结果 ===")
print(df_rhpe.to_string())

# backbone 名称标准化映射
backbone_map = {
    "resnet50":        "resnet50",
    "efficientnet_b4": "efficientnet_b4",
    "dinov2_vits":     "dinov2_vits",
    "dinov2_vitb":     "dinov2_vitb",
    "dinov2_vitl":     "dinov2_vitl",
    "dinov2_vitg":     "dinov2_vitg",
}
df_rhpe["backbone"] = df_rhpe["backbone_raw"].map(backbone_map)

# 合并
df_merged = rsna_rows.merge(df_rhpe[["backbone", "rhpe_mae", "rhpe_rmse", "rhpe_r2", "rhpe_num_samples"]],
                             on="backbone", how="left")

# 计算泛化衰减百分比
df_merged["generalization_gap"] = (
    (df_merged["rhpe_mae"] - df_merged["rsna_mae"]) / df_merged["rsna_mae"] * 100
).round(1)

# backbone 显示名
label_map = {
    "resnet50": "ResNet-50", "efficientnet_b4": "EfficientNet-B4",
    "dinov2_vits": "ViT-S", "dinov2_vitb": "ViT-B",
    "dinov2_vitl": "ViT-L", "dinov2_vitg": "ViT-g",
}
backbone_order = ["resnet50", "efficientnet_b4", "dinov2_vits", "dinov2_vitb", "dinov2_vitl", "dinov2_vitg"]
df_merged["backbone_label"] = df_merged["backbone"].map(label_map)
df_merged["backbone"] = pd.Categorical(df_merged["backbone"], categories=backbone_order, ordered=True)
df_merged = df_merged.sort_values("backbone").reset_index(drop=True)

# 保存
out_path = LIM_ROOT / "generalization" / "generalization_summary.csv"
df_merged.to_csv(out_path, index=False, float_format="%.4f")

print("\n=== 最终泛化汇总表 ===")
print(df_merged[[
    "backbone_label", "rsna_exp_id",
    "rsna_mae", "rsna_rmse", "rsna_r2",
    "rhpe_mae", "rhpe_rmse", "rhpe_r2",
    "generalization_gap"
]].to_string(index=False))
print(f"\n已保存至: {out_path}")