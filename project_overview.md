# 项目概览报告

**生成时间**: 2026-05-18 16:50 (CST)

---

## 1. 目录结构

### 项目根目录

```
├── .claude/                          [CONFIG] AI 助手配置 (settings.local.json)
├── .vscode/                          [CONFIG] 编辑器配置 (settings.json — conda 环境)
├── bone/                             [CODE]   主项目代码 (~4700 个文件, ~1.5 GB)
│   ├── .git/                         [OTHER]  bone 项目的 Git 仓库
│   ├── code/                         [CODE]   Python 源码与实验结果
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── *.py                      ~30 个 Python 脚本
│   │   ├── ablation/                 消融实验脚本
│   │   ├── agents/                   Gemini + RAG agent 脚本
│   │   ├── demos/                    演示脚本
│   │   ├── legacy/                   遗留旧版脚本 (gpt.py, trainer.py 等)
│   │   ├── library_complement/       知识库补充脚本
│   │   ├── output/                   输出文件
│   │   ├── rag/                      RAG 检索系统 (hybrid graph retriever)
│   │   ├── result/                   实验结果 (多个带时间戳的运行记录)
│   │   ├── test/                     测试脚本及 order.txt
│   │   ├── train/                    MLP 训练脚本与权重
│   │   │   ├── README.md
│   │   │   ├── train_mlp_from_embeddings.py
│   │   │   ├── train_mlp_ensemble.py
│   │   │   ├── ensemble_predict.py
│   │   │   └── MLP_weight/           训练好的 MLP 权重与预测结果
│   │   └── vision/                   DINOv2 推理脚本
│   ├── library/                      RAG 系统知识库
│   │   ├── RHPE/                     RHPE 数据集知识库 (~3762 个文件)
│   │   │   ├── data/                 (空 — 数据在 骨龄/bone/ 下)
│   │   │   ├── logic_library/        逻辑特征库 (indexes, JSON features)
│   │   │   ├── logic_library_test/   测试用逻辑库
│   │   │   ├── logic_library_test_1/ 额外测试用逻辑库
│   │   │   ├── memory_library/       记忆库文件
│   │   │   ├── vision_library/       视觉 embedding 与元数据
│   │   │   └── vision_library_test/  测试用视觉库
│   │   ├── RSNA/                     RSNA 知识库 (空文件夹 — 数据集尚未构建)
│   │   └── weixin/                   微信知识库 (空文件夹)
│   └── prompt_test/                  Prompt 测试文件
├── bone.zip                          ~258 MB 压缩包 (项目代码备份?)
├── 骨龄/                              [DATA]   主数据集 (~19402 个文件, ~19 GB)
│   ├── bone/                         RSNA + RHPE 数据集，含 DINOv2 源码
│   │   ├── .git/
│   │   ├── boneage-test-dataset.csv   测试集标签 CSV (Case ID, Sex)
│   │   ├── boneage-training-dataset/  RSNA 训练图像 (数据集双层目录, ~12611 个 PNG 文件)
│   │   ├── boneage-training-dataset.csv RSNA 训练标签 (id, boneage, male)
│   │   ├── boneage-test-dataset/     RSNA 测试图像 (~200 个 PNG 文件)
│   │   ├── code/                     额外代码
│   │   ├── data/                     (空)
│   │   ├── dinov2/                   Facebook 官方 DINOv2 仓库
│   │   ├── RHPE_Annotations/         RHPE 标注 (解剖学 ROI, boneage CSV)
│   │   ├── RHPE_test/                RHPE 测试图像 (~80 个 PNG 文件)
│   │   ├── RHPE_train/               RHPE 训练图像 (~5496 个 PNG 文件)
│   │   └── RHPE_val/                 RHPE 验证图像 (~716 个 PNG 文件)
│   └── weixin_imgs/                  微信图像 (~44 个文件, 男/子 目录)
└── 骨龄.zip                           ~18.8 GB 压缩包 (骨龄/ 文件夹)
```

### 文件类型统计 (排除 `.git` 与 `.pyc`)

| 扩展名 | 数量 | 说明 |
|-------|------|------|
| `.png` | 19,106 | X 光图像 (RHPE/RSNA 数据集) |
| `.json` | 2,667 | 逻辑特征、倒排索引、实验结果、配置文件 |
| `.npy` | 1,821 | DINOv2 embeddings (1536 维向量) |
| `.py` | 181 | Python 源代码 |
| `.csv` | 109 | 标签、元数据、实验结果 |
| `.jpg` | 44 | 微信图像 |
| `.sample` | 42 | DINOv2 配置样本文件 |
| `.pt` | 22 | PyTorch checkpoint 文件 |
| `.yaml` | 15 | 配置文件 (DINOv2 配置) |
| `.md` | 8 | 文档 |
| `.txt` | 7 | 文本文件 |
| `.pth` | 3 | PyTorch 模型权重 |
| `.ipynb` | 2 | Jupyter notebook |
| `.pptx` | 2 | PowerPoint 文件 |
| `.pdf` | 2 | PDF 文件 |

**总磁盘占用**: ~38.40 GB

---

## 2. 关键文件与分类

### 文档

| 路径 | 类型 | 说明 |
|------|------|------|
| `bone/code/README.md` | [DOC] | 代码目录结构概览 |
| `bone/code/rag/README.md` | [DOC] | 混合图检索系统文档 |
| `bone/code/rag/QUICKSTART.md` | [DOC] | RAG 系统快速入门指南 |
| `bone/code/train/README.md` | [DOC] | MLP 训练指南与集成实验结果 |
| `骨龄/bone/dinov2/README.md` | [DOC] | Meta DINOv2 官方 README |
| `骨龄/bone/dinov2/MODEL_CARD.md` | [DOC] | DINOv2 模型卡 |
| `骨龄/骨龄.pdf` | [DOC] | 骨龄评估相关参考文献 |
| `骨龄/文献.pptx` | [DOC] | 骨龄项目参考文献幻灯片 |
| `骨龄/演示文稿1.pptx` | [DOC] | 项目相关演示文稿 |
| `骨龄/KBS__DCSTN____compile_local___Copy_.pdf` | [DOC] | KBS/DCSTN 相关参考文献 |

### 训练入口

| 路径 | 类型 | 说明 |
|------|------|------|
| `bone/code/train/train_mlp_from_embeddings.py` | [CODE] | 在 DINOv2 embedding 上训练 MLP 进行骨龄回归 |
| `bone/code/train/train_mlp_ensemble.py` | [CODE] | 5 折集成 MLP 训练 |
| `bone/code/train/ensemble_predict.py` | [CODE] | 集成模型预测脚本 |
| `bone/code/agents/run_gemini2_rag_eval.py` | [CODE] | 主评估流程: Gemini2+RAG 工作流 |
| `骨龄/bone/dinov2/dinov2/run/train/train.py` | [CODE] | DINOv2 官方训练脚本 (参考用) |
| `bone/datain.py` | [CODE] | 数据加载与预处理工具 |

### 推理与评估

| 路径 | 类型 | 说明 |
|------|------|------|
| `bone/code/bone_age_inference_runner.py` | [CODE] | DINOv2 批量/单张推理入口 |
| `bone/code/evaluate_inference.py` | [CODE] | 阶段一评估入口 |
| `bone/code/agents/gemini_switch.py` | [CODE] | Gemini 1/2 切换与 RAG 流程入口 |
| `bone/code/agents/rag_eval_service.py` | [CODE] | RAG 评估服务 |
| `bone/code/agents/mlp_age_service.py` | [CODE] | MLP 骨龄预测服务 |
| `bone/code/agents/eval_utils.py` | [CODE] | 评估工具函数 |
| `bone/code/vision/bone_age_inference_runner.py` | [CODE] | 视觉推理 runner |

### RAG 系统

| 路径 | 类型 | 说明 |
|------|------|------|
| `bone/code/rag/__init__.py` | [CODE] | RAG 包初始化 |
| `bone/code/rag/hybrid_retriever.py` | [CODE] | 核心混合检索器 (图检索 + 视觉检索) |
| `bone/code/rag/graph_retrieval.py` | [CODE] | 逻辑检索 (倒排索引加权投票) |
| `bone/code/rag/visual_retrieval.py` | [CODE] | 视觉检索 (余弦相似度密集向量搜索) |
| `bone/code/rag/rag_inference_engine.py` | [CODE] | 端到端 RAG 推理引擎 |
| `bone/code/rag/feature_weights.py` | [CODE] | 年龄感知特征权重管理器 |
| `bone/code/rag/utils.py` | [CODE] | 工具函数 (余弦相似度, top-k 等) |
| `bone/code/rag/demo.py` | [CODE] | RAG 系统演示脚本 |

### 逻辑/视觉知识库构建

| 路径 | 类型 | 说明 |
|------|------|------|
| `bone/code/build_logic_library.py` | [CODE] | 构建逻辑知识库 (indexes, json features) |
| `bone/code/build_vision_library.py` | [CODE] | 构建视觉知识库 (embeddings, metadata) |
| `bone/code/select_rhpe_balanced_supplement.py` | [CODE] | 选择均衡补充病例 |
| `bone/code/repair_rhpe_alignment.py` | [CODE] | 修复 RHPE 对齐问题 |

### 数据集标签

| 路径 | 类型 | 说明 |
|------|------|------|
| `骨龄/bone/boneage-training-dataset.csv` | [LABEL] | RSNA 训练标签: id, boneage, male (12611 例) |
| `骨龄/bone/boneage-test-dataset.csv` | [LABEL] | RSNA 测试标签: Case ID, Sex (200 例) |
| `骨龄/bone/RHPE_Annotations/RHPE_Annotations/BONEAGE/boneage_train.csv` | [LABEL] | RHPE 训练: ID, Male, Boneage, Chronological (~5496) |
| `骨龄/bone/RHPE_Annotations/RHPE_Annotations/BONEAGE/boneage_val.csv` | [LABEL] | RHPE 验证 (~716) |
| `骨龄/bone/RHPE_Annotations/RHPE_Annotations/BONEAGE/gender_test.csv` | [LABEL] | 性别测试 |
| `骨龄/bone/RHPE_Annotations/RHPE_Annotations/ANATOMICAL_ROIS/` | [LABEL] | 解剖学 ROI 标注 (train/val/test JSON) |
| `骨龄/特征.xlsx` | [LABEL] | 骨龄特征汇总表 |

### 配置文件

| 路径 | 类型 | 说明 |
|------|------|------|
| `骨龄/bone/dinov2/dinov2/configs/eval/*.yaml` | [CONFIG] | DINOv2 评估配置 (vitb14, vitg14, vitl14, vits14) |
| `骨龄/bone/dinov2/dinov2/configs/train/*.yaml` | [CONFIG] | DINOv2 训练配置 (vitg14, vitl14, vitl16_short) |
| `骨龄/bone/dinov2/conda.yaml` | [CONFIG] | DINOv2 conda 环境定义 |
| `骨龄/bone/dinov2/setup.py` | [CONFIG] | DINOv2 包安装配置 |
| `骨龄/bone/dinov2/requirements.txt` | [CONFIG] | DINOv2 依赖列表 |
| `.claude/settings.local.json` | [CONFIG] | Claude Code 权限设置 |
| `.vscode/settings.json` | [CONFIG] | VS Code: conda 环境管理器 |

### 模型权重

| 路径 | 类型 | 说明 |
|------|------|------|
| `骨龄/bone/dinov2/bone_age_predictor_best.pth` | [MODEL] | ~4.55 GB — DINOv2 微调骨龄预测器 |
| `bone/code/boneage_model.pth` | [MODEL] | ~71 MB — 骨龄 MLP 回归头权重 |
| `骨龄/bone/code/boneage_model.pth` | [MODEL] | ~71 MB — 骨龄 MLP 回归头权重 (副本) |
| `bone/code/train/MLP_weight/RHPE/YYMMDD_HHMMSS/best.pt` | [MODEL] | 每次训练的最佳 checkpoint |
| `bone/code/train/MLP_weight/RHPE/ensemble/seed_*/best.pt` | [MODEL] | 5-seed 集成 checkpoint |

---

## 3. 项目理解

### 总体目标

本项目是一个**骨龄评估系统 (bone age estimation system)**，采用多阶段 AI 流水线，从手部 X 光片中预测骨骼年龄（骨龄，以月为单位）。

### 数据集

本项目使用**三个数据集**:

| 数据集 | 类型 | 规模 | 位置 |
|--------|------|------|------|
| **RSNA Bone Age Dataset** | 公开 (RSNA 2017) | ~12,611 训练 + ~200 测试 | `骨龄/bone/boneage-training-dataset/` |
| **RHPE Dataset** (主要) | 公开 | ~5,496 训练 + ~716 验证 + ~80 测试 | `骨龄/bone/RHPE_train/`, `RHPE_val/`, `RHPE_test/` |
| **WeChat (微信) Dataset** | 公开(guideline) | ~44 张图像 | `骨龄/weixin_imgs/` |

### 架构与流水线

本系统采用 **多阶段 RAG 增强流水线 (multi-stage RAG-enhanced pipeline)**:

#### Stage 1 — DINOv2 视觉特征提取 (Feature Extraction)
- 使用 Meta 的 DINOv2 (ViT-g/14, 1.1B 参数)
- 微调 checkpoint: `bone_age_predictor_best.pth` (~4.55 GB)
- 从手部 X 光片中提取 **1536 维 embedding**

#### Stage 2 — RAG 检索增强生成 (Retrieval-Augmented Generation)
采用混合图检索 (Hybrid Graph Retriever):

- **逻辑检索 (Logical Path)**: 倒排索引加权投票
  - 年龄感知权重 (Age-aware feature weights)
  - 特征类型: 腕骨 (carpal)、指骨 (phalanges)、桡骨 (radius)、尺骨茎突 (ulna styloid)、籽骨 (sesamoid)、融合状态 (fusion)
- **视觉检索 (Visual Path)**: 余弦相似度密集向量搜索
- **混合融合 (Hybrid Fusion)**: 软交集 + 重排序 → Top-3
- 已为 RHPE 构建知识库: ~1012 逻辑案例, ~256 视觉案例

#### Stage 3 — MLP 回归头 (Regression)
- 在 DINOv2 embedding 上训练: 1536 → 1024 → 512 → 256 → 1
- 可选的性别特征输入
- 损失函数: **MAE + GW-MAE** (Gender-Weighted MAE)
- 5-seed 集成结果 (RHPE 测试集):

| 指标 | 最佳单模型 | 5 折集成 |
|:---|---:|---:|
| **Test MAE** | 5.997 | **5.643** |
| **Test GW-MAE** | 5.877 | **5.589** |
| **Test RMSE** | 7.523 | **7.317** |
| **Median AE** | 5.122 | **4.477** |
| **Within 6m** | 54.9% | **63.7%** |
| **Within 12m** | 87.4% | **87.9%** |

#### Stage 4 — Gemini 大模型集成 (LLM Integration)
- 使用 Google Gemini 2.0 / 3.1 Pro 进行诊断
- 流程: RAG 结果 + MLP 先验 → 构建 prompt → Gemini 诊断
- 多种预测模式: `mlp-only`, `rag-only`, `mlp+rag`, 全流水线

### 深度学习框架

**PyTorch** — 通过 `.pt` / `.pth` 权重文件、`torch` 导入、DINOv2 基于 PyTorch 等确认。

### 任务类型

**回归 (Regression)** — 预测连续型骨龄（月龄）。

主要指标: MAE、Median AE、RMSE、MAE(F)、MAE(M)

### 实验追踪

实验结果按时间戳目录组织:
```
bone/code/result/YYYYMMDD_HHMMSS/
├── summary.csv, B_summary.csv     — 聚合指标
├── A_cases/result_case_*.json     — 逐 case 详情
└── cases/                         — 额外输出
```

### 主要依赖

**DINOv2 依赖**:
- PyTorch 2.0 (CUDA 11.7), xformers 0.0.18
- omegaconf, fvcore, iopath, submitit
- mmcv-full 1.5.0, mmsegmentation 0.27.0

**项目特定依赖**:
- numpy, pandas, scikit-learn
- tqdm
- Google Generative AI (gemini)

### GitHub 链接

本项目两个 git 仓库均未配置远程地址，目前没有关联的 GitHub 链接。
> 项目中包含的 DINOv2 官方仓库: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

---

## 4. 建议的后续步骤

### 1. 数据完整性检查
- 确认 `boneage_train.csv` / `boneage_val.csv` 中所有图像文件均存在
- 验证 RHPE_Annotations 与实际图像文件匹配
- 交叉检查知识库 embedding 与图像 ID 的对应关系

### 2. 环境配置
- 使用提供的 `conda.yaml` 创建 conda 环境
- 确保安装 CUDA 兼容的 PyTorch (CUDA 11.7+)
- 配置 Gemini API key 以支持 LLM 阶段

### 3. 复现基线 — 训练 MLP baseline
```bash
python code/train/train_mlp_from_embeddings.py \
  --metadata-csv library/RHPE/vision_library/metadata/rhpe_metadata.csv \
  --embeddings-dir library/RHPE/vision_library/embeddings \
  --output-dir code/train/MLP_weight \
  --dataset-name rhpe
```

### 4. 运行推理 — 在小样本上测试 RAG 流水线
```bash
python code/agents/run_gemini2_rag_eval.py \
  --dataset rhpe \
  --csv 骨龄/bone/RHPE_Annotations/RHPE_Annotations/BONEAGE/boneage_val.csv \
  --img-dir 骨龄/bone/RHPE_val/images \
  --vision-library-dir library/RHPE/vision_library \
  --logic-library-dir library/RHPE/logic_library \
  --output-root code/result/test_run
```

### 5. 阅读代码 — 按顺序阅读关键脚本
1. `code/train/train_mlp_from_embeddings.py` — MLP 训练流水线
2. `code/agents/run_gemini2_rag_eval.py` — 主评估流水线
3. `code/rag/hybrid_retriever.py` — RAG 核心逻辑
4. `code/agents/eval_utils.py` — 共享工具函数

### 6. 数据扩充
- 将 RSNA 数据集通过知识库流水线进行处理
- 集成微信数据集
- 如有更多 RHPE 训练数据可补充添加

### 7. 性能分析
- 比较 `code/result/` 下不同时间戳运行的结果
- 分析 `code/result/ablation_20260424/` 与 `ablation_20260426/` 中的消融实验
- 查看 MLP 集成多样性分析结果

---

*报告结束*
