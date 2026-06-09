"""
LIM/code/utils/vis_style.py
全局可视化样式配置
所有绘图脚本在生成任何图像前必须调用 apply_style()
"""
import matplotlib.pyplot as plt


def apply_style() -> None:
    """在所有绘图脚本的最顶部调用一次。"""
    plt.rcParams.update({
        # 字体
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Georgia",
                              "DejaVu Serif", "serif"],
        "mathtext.fontset":  "stix",
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   11,
        # 坐标轴
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    "#444444",
        # 刻度
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        # 网格
        "axes.grid":         True,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "grid.color":        "#E0E0E0",
        "grid.alpha":        0.7,
        # 图像输出
        "figure.dpi":        100,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


# ── 语义配色（全项目固定，禁止随意更换）────────────────────────────────────────
COLORS = {
    # Backbone 系列（按规模递增）
    "resnet50":        "#457B9D",
    "efficientnet_b4": "#2A9D8F",
    "dinov2_vits":     "#E9C46A",
    "dinov2_vitb":     "#E64B35",
    "dinov2_vitl":     "#6A4C93",
    "dinov2_vitg":     "#333333",
    # 损失函数
    "mae":             "#457B9D",
    "gwmae":           "#E64B35",
    "combined":        "#2A9D8F",
    # 性别
    "male":            "#457B9D",
    "female":          "#E64B35",
    # 数据集
    "rsna":            "#457B9D",
    "rhpe":            "#E64B35",
    # 年龄段背景
    "age_0_60":        "#FFF5F5",
    "age_60_120":      "#F0F7FF",
    "age_120_180":     "#F0FFF4",
    "age_180_228":     "#FFFFF0",
    # 通用
    "neutral":         "#D3D3D3",
    "text_primary":    "#333333",
    "text_secondary":  "#666666",
    "axis_line":       "#444444",
    "grid_line":       "#E0E0E0",
}

# 标准图幅（英寸）
FIGSIZE = {
    "single":      (6,  5),
    "wide_1x2":    (12, 5),
    "grid_2x2":    (10, 8),
    "heatmap":     (10, 6),
}


def style_axes(ax) -> None:
    """对单个 Axes 应用标准坐标轴样式，在每个子图创建后调用。"""
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color("#444444")
    ax.tick_params(axis="both", which="major",
                   direction="out", length=4, width=1.0,
                   labelsize=10, colors="#333333", pad=4)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5,
            color="#E0E0E0", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    # 快速验证：生成一张测试图，确认样式正常应用
    import numpy as np

    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["wide_1x2"])

    # 左图：用 Backbone 配色画折线
    x = np.linspace(0, 228, 100)
    for key in ["resnet50", "efficientnet_b4", "dinov2_vits",
                "dinov2_vitb", "dinov2_vitl", "dinov2_vitg"]:
        axes[0].plot(x, np.random.rand(100) * 10 + 5,
                     color=COLORS[key], label=key, linewidth=1.5)
    style_axes(axes[0])
    axes[0].set_xlabel(r"$\it{Age}$ (months)")
    axes[0].set_ylabel(r"$\it{MAE}$ (months)")
    axes[0].set_title("Backbone Color Test")
    axes[0].legend(fontsize=9, frameon=False)

    # 右图：用损失函数配色画条形
    bars = ["mae", "gwmae", "combined"]
    axes[1].bar(bars, [7.2, 6.8, 6.5],
                color=[COLORS[b] for b in bars], width=0.5)
    style_axes(axes[1])
    axes[1].set_ylabel(r"$\it{MAE}$ (months)")
    axes[1].set_title("Loss Color Test")

    plt.tight_layout()
    out_path = "LIM/figures/vis_style_test.png"
    plt.savefig(out_path)
    print(f"测试图已保存至：{out_path}")
    print("vis_style.py 验证通过。")