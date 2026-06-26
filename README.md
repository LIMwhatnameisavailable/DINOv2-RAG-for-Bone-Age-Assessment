# 🦷 基于 DINOv2 的骨龄评估迁移实验

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![DINOv2](https://img.shields.io/badge/DINOv2-ViT--S/B/L/g-green)
![License](https://img.shields.io/badge/用途-科研教育-yellow)

本项目在 **RSNA 2017** 与 **RHPE** 两个公开手部 X 光数据集上，系统评估不同视觉骨干网络、损失函数设计与性别特征融合对骨龄预测精度的影响，并开展零样本跨数据集泛化测试。所有骨干网络权重完全冻结，仅训练轻量 MLP 回归头，以纯迁移学习视角比较自监督 ViT 与 CNN 基线的特征表征能力。

---

## 📌 研究问题

1. DINOv2 四种规模变体（ViT-S/B/L/g）与 CNN 基线（ResNet-50、EfficientNet-B4）在骨龄评估任务上的特征表征能力是否存在系统性差异？
2. 临床阶段加权损失函数 **GW-MAE** 能否在青春期前关键发育窗口（60–180 月）取得优于标准 MAE 的预测精度？
3. 在 RSNA 上训练的模型能否在 RHPE 上保持可接受的**零样本跨数据集泛化**性能？

---

## 🗂️ 数据集

| 数据集 | 规模 | 骨龄范围 | 用途 |
|--------|------|----------|------|
| RSNA Bone Age 2017 | 12,611 张 | 12–228 月 | 训练 / 内部测试 |
| RHPE | 6,204 张 | 12–228 月 | 跨数据集泛化测试（仅推理） |

两个数据集均为公开数据集，**不随代码一并发布**，需自行下载后放置于 `LIM_data/rsna/` 与 `LIM_data/rhpe/`。

- **RSNA Bone Age 2017**：[官方页面](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017) · [Kaggle 镜像](https://www.kaggle.com/datasets/kmader/rsna-bone-age)
- **RHPE**：[官方页面](https://bcv-uniandes.github.io/baar-wp/)

---

## 🔬 方法框架

```
手部 X 光图像
     ↓
[阶段一] 特征提取（冻结骨干网络）
     · DINOv2 ViT-S/B/L/g    →  build_vision_library.py
     · ResNet-50 / EfficientNet-B4  →  extract_cnn_embeddings.py
     ↓
[阶段二] MLP 回归头训练（train_mlp.py）
     · 损失函数消融：MAE / GW-MAE / MAE+GW-MAE
     · 性别特征消融：有 / 无
     ↓
[阶段三] 跨数据集零样本推理（infer_rhpe.py）
     ↓
[阶段四] 分析与可视化（analyze_results.py）
```

---

## 🧠 骨干网络与特征维度

| Backbone | 类型 | 特征维度 | 预训练权重 |
|----------|------|----------|------------|
| ResNet-50 | CNN | 2048 | ImageNet-1k |
| EfficientNet-B4 | CNN | 1792 | ImageNet-1k |
| DINOv2 ViT-S/14 | 自监督 ViT | 384 | DINOv2 |
| DINOv2 ViT-B/14 | 自监督 ViT | 768 | DINOv2 |
| DINOv2 ViT-L/14 | 自监督 ViT | 1024 | DINOv2 |
| DINOv2 ViT-g/14 | 自监督 ViT | 1536 | DINOv2 |

> 所有骨干网络权重在训练阶段**完全冻结**，仅训练 MLP 回归头。

---

## 🧪 实验矩阵

**6 种骨干网络 × 3 种损失函数 × 2 种性别输入 = 共 24 组消融实验（E01–E24）**

| 维度 | 选项 |
|------|------|
| 骨干网络 | ResNet-50 · EfficientNet-B4 · ViT-S · ViT-B · ViT-L · ViT-g |
| 损失函数 | MAE · GW-MAE · MAE + GW-MAE（Combined） |
| 性别输入 | 有 / 无 |

---

## 📊 主要结果

- **DINOv2 vs CNN**：DINOv2 系列 RSNA 测试集 MAE 均不超过 9.16 月，CNN 系列为 11.60–14.10 月，差距达 **2.4–5.9 月**
- **规模效应非单调**：ViT-B（MAE = **8.24 月**）优于参数量最大的 ViT-g（MAE = 8.64 月）
- **性别信息**：引入后对所有 backbone 均有正向效果，MAE 降幅 **1.5–3.4 月**
- **GW-MAE**：在 CNN 系列上有小幅改善；在大规模 ViT 上改善不稳定
- **跨数据集泛化**：
  - ViT-B 最稳健（RHPE MAE = 15.03 月，衰减 +82%）
  - ViT-g 出现异常衰减（RHPE MAE = 28.21 月，衰减 **+227%**），误差随年龄段升高急剧放大

---

## 📂 目录结构

```
BoneAge-DINOv2/
├── code/
│   ├── build_vision_library.py     # DINOv2 特征提取
│   ├── extract_cnn_embeddings.py   # CNN 特征提取
│   ├── train_mlp.py                # MLP 回归头训练（24 组实验）
│   ├── infer_rhpe.py               # 跨数据集零样本推理
│   ├── analyze_results.py          # 结果分析与可视化
│   └── utils/
│       ├── gw_mae.py               # GW-MAE 损失函数实现
│       ├── vis_style.py            # 统一可视化风格
│       └── data_utils.py           # 数据加载工具函数
├── embeddings/                     # 预提取特征向量（.npy）
├── experiments/                    # 24 组实验结果（E01–E24）
├── generalization/                 # 跨数据集泛化结果
├── figures/                        # 输出图表（300 DPI PNG）
├── logs/                           # 训练与推理日志
├── myproject_instruction.md        # 逐步操作说明
├── project_overview.md             # 项目总览
└── README.md
```

> **数据目录**（不含于仓库）：`LIM_data/rsna/` 与 `LIM_data/rhpe/`

---

## ▶️ 使用方法

激活 `boneage` conda 环境后，按顺序运行：

```bash
conda activate boneage

# 阶段一：特征提取
python code/build_vision_library.py      # DINOv2 系列
python code/extract_cnn_embeddings.py    # CNN 基线

# 阶段二：MLP 训练（24 组消融）
python code/train_mlp.py

# 阶段三：跨数据集推理
python code/infer_rhpe.py

# 阶段四：分析与可视化
python code/analyze_results.py
```

---

## 🛠️ 环境依赖

| 包 | 用途 |
|----|------|
| `torch` / `torchvision` | 模型推理与训练 |
| `timm` | EfficientNet-B4 / ResNet-50 权重加载 |
| `numpy` / `pandas` | 数据处理 |
| `scikit-learn` | 评估指标 |
| `matplotlib` / `seaborn` | 可视化（300 DPI） |

```bash
conda create -n boneage python=3.10
conda activate boneage
pip install torch torchvision timm numpy pandas scikit-learn matplotlib seaborn
```

---

## 👤 作者

**LIMwhatnameisavailable**
东南大学生物科学与医学工程学院 · 数字医学工程全国重点实验室

联系方式：213230182@seu.removethis.edu.cn

---

## 📄 许可

本项目仅供**科研与教育用途**。
RSNA 与 RHPE 数据集须遵守各自原始许可协议。
