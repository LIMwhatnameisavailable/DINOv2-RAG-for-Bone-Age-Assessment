# 基于 DINOv2 的骨龄评估迁移实验

本项目在 RSNA 2017 与 RHPE 两个公开手部 X 光数据集上，系统评估
不同视觉骨干网络、损失函数设计与性别特征融合对骨龄预测精度的影响，
并开展零样本跨数据集泛化测试。

## 研究问题

1. DINOv2 四种规模变体（ViT-S/B/L/g）与 CNN 基线（ResNet-50、
   EfficientNet-B4）在骨龄评估任务上的特征表征能力是否存在系统性差异？
2. 临床阶段加权损失函数 GW-MAE 能否在青春期前关键发育窗口
   （60–180 月）取得优于标准 MAE 的预测精度？
3. 在 RSNA 上训练的模型能否在 RHPE 上保持可接受的零样本泛化性能？


## 数据集

| 数据集 | 规模 | 骨龄范围 | 用途 |
|---|---|---|---|
| RSNA Bone Age 2017 | 12,611 张 | 12–228 月 | 训练 / 内部测试 |
| RHPE | 6,204 张 | 12–228 月 | 跨数据集泛化测试（仅推理） |

两个数据集均为公开数据集，不随代码一并发布。
- RSNA Bone Age 2017：https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017
  （Kaggle 镜像：https://www.kaggle.com/datasets/kmader/rsna-bone-age）
- RHPE：https://bcv-uniandes.github.io/baar-wp/



## 方法框架

```
  手部 X 光图像
       ↓
  [阶段一] 特征提取（冻结骨干网络）
       · DINOv2 ViT-S/B/L/g  →  build_vision_library.py
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

## 骨干网络与特征维度

| Backbone | 类型 | 特征维度 | 预训练权重 |
|---|---|---|---|
| ResNet-50 | CNN | 2048 | ImageNet-1k |
| EfficientNet-B4 | CNN | 1792 | ImageNet-1k |
| DINOv2 ViT-S/14 | 自监督 ViT | 384 | DINOv2 |
| DINOv2 ViT-B/14 | 自监督 ViT | 768 | DINOv2 |
| DINOv2 ViT-L/14 | 自监督 ViT | 1024 | DINOv2 |
| DINOv2 ViT-g/14 | 自监督 ViT | 1536 | DINOv2 |

所有骨干网络权重在训练阶段完全冻结，仅训练 MLP 回归头。


## 实验矩阵（24 组消融实验）

6 种骨干网络 × 3 种损失函数（MAE / GW-MAE / Combined）× 性别输入（有/无）
→ 共 24 组实验（E01–E24）


## 主要结果

- DINOv2 系列 RSNA 测试集 MAE 均不超过 9.16 月，CNN 系列为
  11.60–14.10 月，两类方法差距达 2.4–5.9 月
- DINOv2 内部规模效应非单调：ViT-B（MAE = 8.24 月）优于参数量
  最大的 ViT-g（MAE = 8.64 月）
- 性别信息引入对所有 backbone 均有正向效果，MAE 降幅 1.5–3.4 月
- GW-MAE 在 CNN 系列上有小幅改善，在大规模 ViT 上改善不稳定
- 跨数据集泛化：ViT-B 表现最稳健（RHPE MAE = 15.03 月，衰减 +82%）；
  ViT-g 出现异常衰减（RHPE MAE = 28.21 月，衰减 +227%），
  误差随年龄段升高急剧放大

---

## 目录结构

```
  ├── code/
  │   ├── build_vision_library.py     # DINOv2 特征提取
  │   ├── extract_cnn_embeddings.py   # CNN 特征提取
  │   ├── train_mlp.py                # MLP 训练
  │   ├── infer_rhpe.py               # 跨数据集推理
  │   ├── analyze_results.py          # 分析与可视化
  │   └── utils/
  │       ├── gw_mae.py               # GW-MAE 损失函数
  │       ├── vis_style.py            # 可视化风格
  │       └── data_utils.py           # 数据工具函数
  ├── embeddings/                     # 预提取特征向量（.npy）
  ├── experiments/                    # 24 组实验结果
  ├── generalization/                 # 跨数据集泛化结果
  ├── figures/                        # 输出图表（300 DPI）
  ├── logs/
  ├── myproject_instruction.md
  └── project_overview.md
```

数据目录（不含于仓库）：LIM_data/rsna/ 与 LIM_data/rhpe/

## 环境依赖

  Python 3.x，conda 环境 boneage
  主要依赖：torch · timm · numpy · pandas · scikit-learn · matplotlib · seaborn
