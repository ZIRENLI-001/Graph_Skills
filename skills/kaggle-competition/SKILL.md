---
name: kaggle-competition
description: >
  Kaggle/数据科学竞赛全流程辅助。当用户提到Kaggle比赛、数据科学竞赛、
  机器学习竞赛、赛题分析、特征工程、模型融合、LB分数优化、AIMO、ARC Prize、
  Konwinski Prize、LLM竞赛、Agent竞赛时激活此技能。
  涵盖赛题解析、方案制定、数据梳理、开源方案分析、方案迭代全流程。
---

# Kaggle 竞赛全流程指南

基于 AIMO1/2/3、ARC Prize 2024/2025、Konwinski Prize、LMSYS Chatbot Arena、BirdCLEF、
Santa 2024、FathomNet 2025、Playground 2025 等真实赛题的获奖方案提炼。

- 官方网站: https://www.kaggle.com/competitions
- 方案集合: https://farid.one/kaggle-solutions/
- 竞赛趋势: https://mlcontests.com/

## 1. 竞赛类型分类与典型赛题

| 赛道 | 典型赛题 | 奖金 | 核心方法 | 代表获奖方案 |
|------|---------|------|---------|-------------|
| 表格 | Playground S5, Home Credit | $10k-$105k | GBDT集成+Stacking | Deotte 2025: 72模型三层Stacking + RAPIDS cuML |
| 医学图像 | ISIC 2024 皮肤癌 | $50k | Vision+TTA重度集成 | 1st: 多模型Ensemble+TTA |
| 音频 | BirdCLEF 2024 | $50k | 频谱图+CNN+蒸馏 | 1st: CE替代BCE+sigmoid推理; 3rd: 伪标签+蒸馏(CPU) |
| NLP/LLM | LMSYS Chatbot Arena | $100k | LLM微调+蒸馏 | 1st: Gemma2-9b+Llama3-70B+Qwen2-72B LoRA蒸馏 |
| 数学推理 | AIMO 1/2/3 | $220万 | 数学LLM+TIR | AIMO2 1st: OpenMath-Nemotron-14B+GenSelect |
| AGI推理 | ARC Prize 2024/2025 | $60万+ | TTT+合成数据 | NVARC: Qwen-4B+合成数据+TTT(24%) |
| Agent代码 | Konwinski Prize | $100万 | 代码Agent | 1st: Qwen2.5-Coder-32B(7.5%) |
| 多模态 | FathomNet 2025 | - | ViT+层级分类 | 1st: MCEAM+分类学层级辅助分类 |
| 组合优化 | Santa 2024 | $50k | 局部搜索(非ML) | TSP启发式: k-opt+double-bridge |
| Agent社会 | AgentSociety(WWW2025) | - | LLM用户建模 | 清华FIB Lab主办 |

### AIMO 演进对比

| 维度 | AIMO1 | AIMO2 | AIMO3 |
|------|-------|-------|-------|
| 冠军 | Numina | NemoSkills(NVIDIA) | 进行中 |
| 分数 | 29/50 | 34/50 | LB 44/50 |
| 难度 | AMC12/AIME | 国家奥赛 | 国家→IMO |
| 硬件 | P100/2×T4, 9h | 4×L4(96GB), 5h | H100 |
| 答案 | 3位数 | 3位数 | 5位数 |
| 基座 | DeepSeek-Math-7B | Qwen2.5-14B | GPT-OSS-120B/Qwen3 |
| 关键技术 | 两阶段SFT+SC-TIR | OpenMathReasoning+TIR+GenSelect | TBD |

详情 → `references/competition_analysis.md`

## 2. 赛题解析流程（5步法）

```
Step 1: 目标理解
├── 分类/回归/排序/生成/Agent？
├── 单标签/多标签？代码竞赛？
└── 案例: AIMO=数学生成, ARC=视觉推理, Konwinski=代码修复Agent

Step 2: 评估指标深度分析
├── 指标特性 → 损失函数对齐 → 后处理策略
└── 对齐表:
    | 指标 | 损失函数 | 后处理 | 案例 |
    |------|---------|--------|------|
    | AUC | BCE | Rank averaging | ISIC 2024 |
    | LogLoss | BCE | clip[0.01,0.99] ×0.99 | LMSYS |
    | RMSE | MSE | - | Home Credit |
    | QWK | MSE+阈值 | OptimizedRounder | 作文评分 |
    | Accuracy | CE | majority voting | AIMO, ARC |
    | F1-macro | BCE/class | 每类阈值搜索 | Jigsaw |

Step 3: 数据结构分析
├── 数据量级、特征类型、类别平衡、时序性、分组关系
├── LLM赛: 问题格式、答案格式、是否需要代码执行
└── Agent赛: 环境接口、action空间、观测空间

Step 4: 约束条件 ← 极其重要
├── 代码竞赛？(AIMO/ARC/Konwinski都是)
├── GPU限制？(ARC: 4×小GPU 12h ~$0.20/task)
├── 推理时限？(BirdCLEF: CPU-only)
├── 外部数据/模型白名单？(AIMO2: DeepSeek R1白名单后LB巨变)
├── 禁止联网？(大部分代码赛)
└── 禁止闭源API？(Konwinski)

Step 5: 历史相似赛题
├── farid.one/kaggle-solutions 搜索
├── github.com/anuj0456/kaggle_competition_solutions
└── 分析 top solution → 提取可迁移方法
```

详情 → `references/competition_analysis.md`

## 3. 方案制定（算法选型决策树）

```
赛题类型判断
│
├── 表格数据
│   ├── < 10万行 → CatBoost + TabPFN
│   ├── 10万-1000万 → LightGBM + XGBoost + CatBoost
│   ├── > 1000万 → LightGBM GPU + cuDF加速
│   ├── AutoGluon(2024: 15/18表格赛获奖, 7金牌)
│   ├── 特征工程 > 模型选择(ICLR 2025 TabReD)
│   └── 最终: 3-5 GBDT + 1-2 NN + TabPFN → Stacking
│
├── CV/图像
│   ├── 分类 → EfficientNet-B5/B7 / ConvNeXt-L / ViT-L (timm)
│   ├── 检测 → YOLOv8 / DETR
│   ├── 分割 → UNet++ / SegFormer
│   ├── 医学图像 → 重度TTA + 多模型Ensemble
│   ├── 音频 → Log mel spectrogram → 2D图像分类
│   └── 多模态 → ViT + MCEAM + 层级辅助分类
│
├── NLP
│   ├── 文本分类 → DeBERTa-v3-large(300M, Kaggle NLP之王)
│   ├── 新选择 → ModernBERT(比DeBERTa-v3-base更快更强)
│   ├── 生成/QA → Qwen2.5/Qwen3 + QLoRA / DeepSeek
│   ├── 蒸馏 → 70B QLoRA → logit蒸馏到9B → 量化推理
│   └── 混合 → DeBERTa + LLM Ensemble
│
├── 数学/推理赛(AIMO)
│   ├── 基座: DeepSeek-R1-Distill-Qwen / DeepSeek-Math / Qwen-Math
│   ├── 训练: 两阶段SFT(CoT→TIR) + DPO缩短输出 + GRPO
│   ├── 数据: OpenMathReasoning(306k问题/3.2M解) + NuminaMath(800k+)
│   ├── 推理: SC-TIR(多次采样+代码执行+投票) 或 GenSelect(选择器模型)
│   ├── 加速: TensorRT-LLM+FP8(1.5x) + ReDrafter投机解码(1.8x)
│   └── AIMO3: H100 → 可跑GPT-OSS-120B/Qwen3-235B
│
├── AGI推理赛(ARC Prize)
│   ├── 核心: Test-Time Training(所有top方案共同要素)
│   ├── 合成数据: 离线生成ARC风格任务(概念分解+渐进难度)
│   ├── 模型: Qwen-4B(NVARC) / LLaDA-8B(ARChitects) / CodeT5-660M(MindsAI)
│   ├── Trick: 极简tokenizer(16token), 2D-RoPE, per-task LoRA TTT
│   ├── 约束: 4×小GPU 12h ~$0.20/task
│   └── 核心思想: 重计算离线(合成数据), 轻计算在线(小模型TTT)
│
├── Agent代码赛(Konwinski)
│   ├── 模型: Qwen2.5-Coder-32B / DeepSeek-Coder
│   ├── 禁止闭源API, 纯开源
│   ├── 关键: 上下文管理是最重要组件
│   └── 策略: 代码理解→问题定位→patch生成→验证
│
├── 组合优化赛(Santa)
│   ├── 不是ML问题! 是组合优化
│   └── 方法: 局部搜索(TSP启发式), 模拟退火, 并行搜索
│
└── 通用时间线
    ├── 第1天: Baseline + 可信CV
    ├── 第1周: 特征/数据迭代 + 单模型
    ├── 第2周起: 多模型 + Ensemble
    └── 最后3天: 冻结, 只融合和后处理
```

详情 → `references/solution_design.md`
LLM/Agent赛道详情 → `references/llm_agent_competitions.md`

## 4. 数据梳理与特征工程

### EDA 标准流程

```
数据加载(polars/cuDF)
  → 数据概览(shape, dtypes, missing率, 唯一值)
  → 目标变量分析(分布, 类别平衡, 时序关系)
  → 数值特征(分布, 偏度, 相关性矩阵, 异常值)
  → 类别特征(基数, 频率, 与目标关系)
  → Train-Test分布对比(Adversarial Validation)
```

### GM 级特征工程技巧

来自 Chris Deotte 等多位 Kaggle Grandmaster 实战验证:

| 技巧 | 描述 | 价值 |
|------|------|------|
| Groupby聚合 | 按类别键算mean/std/count/min/max/quantile | ★★★★★ |
| 浮点位提取 | `(feat*100).astype(int)%10` 揭示隐藏编码 | ★★★★ |
| 类别交叉 | 拼接2-3个类别列再编码 | ★★★★ |
| 海量特征生成 | 10000+候选 → importance筛选Top500 | ★★★★ |
| 目标编码+CV | fold内算target mean, 防泄漏 | ★★★★ |
| Adversarial Valid. | 训练分类器区分train/test, 找分布差异 | ★★★★ |
| NaN指示列 | 缺失值二值化 | ★★★ |
| 聚类特征 | KMeans标签作新特征 | ★★★ |
| 数值作类别 | 数值离散化后也作类别处理 | ★★★ |
| 差值/比值 | `a-b`, `a/(b+1e-8)` | ★★★ |

### 数据泄漏检测

- **目标泄漏**: 特征在预测时是否可用？
- **训练-测试污染**: Scaler/Encoder是否在全量数据上fit？
- **检测**: 单特征AUC > 0.9 → 高度怀疑
- **外部泄漏**: AIMO2中DeepSeek R1白名单后LB格局巨变

### GPU 加速数据处理

- `cuDF-pandas`: 零代码切换GPU加速pandas (Deotte方案核心)
- `polars scan_parquet()`: 懒加载, 内存友好
- `cuML`: GPU版sklearn (KNN/SVR/Ridge/PCA等)

详情 → `references/data_exploration.md`
脚本 → `scripts/eda_template.py`, `scripts/feature_engineering.py`

## 5. 开源方案分析

### 浏览策略

```
第1天:
  ├── Top 20 Discussion (按Vote排序)
  ├── EDA Notebook (理解数据)
  └── Baseline Notebook (快速起步)
每周:
  ├── 新高分Notebook
  ├── GM/Master发帖
  └── 规则/数据更新公告
赛后:
  ├── 所有金牌writeup
  └── 提取可复用trick
```

### 方案评估 5 维矩阵

| 维度 | 评估标准 |
|------|---------|
| 分数 | Public LB + CV分数 |
| 可信度 | 作者等级(GM>Master>Expert) + 社区投票 |
| 可复现性 | 代码完整、依赖少、运行快 |
| 差异化 | 与自己方案的互补性 |
| 融合价值 | 模型类型多样性(GBDT vs NN vs LM) |

### Trick 分类提取

- **数据类**: 增强方法、清洗策略、外部数据、伪标签
- **模型类**: 架构修改、预训练权重、损失函数
- **训练类**: 学习率策略、warmup、EMA、SWA
- **推理类**: TTA、后处理、量化、蒸馏

### 关键资源

- [farid.one/kaggle-solutions](https://farid.one/kaggle-solutions/) — 最全方案集合
- [kaggle_competition_solutions](https://github.com/anuj0456/kaggle_competition_solutions) — 方案链接库
- [mlcontests.com](https://mlcontests.com/) — 竞赛趋势年度报告

详情 → `references/opensource_analysis.md`

## 6. 方案迭代策略

### 迭代优先级金字塔

```
收益从高到低:
 Level 1: ▲ 可信CV (GM共识: 没有可信CV一切白费)
 Level 2: ▲ 特征工程 (TabReD: 特征 > 模型选择)
 Level 3: ▲ 数据清洗/增强/伪标签
 Level 4: ▲ 单模型超参 (Optuna 50→200 trials)
 Level 5: ▲ 多模型训练 (seed avg + 不同模型)
 Level 6: ▲ 后处理 (阈值/裁剪/排序)
 Level 7: ▲ Ensemble/Stacking
```

### CV 策略速查

| 场景 | CV策略 | 案例 |
|------|--------|------|
| 普通分类 | StratifiedKFold(5) | ISIC, Jigsaw |
| 时序数据 | TimeSeriesSplit+gap | Home Credit |
| 分组数据 | GroupKFold | BirdCLEF(按站点) |
| 小数据 | RepeatedStratifiedKFold | - |
| LLM赛 | 固定验证集(AMC/AIME/MATH) | AIMO系列 |

### GM 核心原则 (Chris Deotte)

1. 先建立可信的本地验证 — CV不可信则一切白费
2. 用GPU加速实验迭代(cuML, cuDF) — 迭代速度是元游戏
3. 显式分析train/test分布差异
4. 理解并直接优化竞赛指标

详情 → `references/iteration_strategy.md`
脚本 → `scripts/cv_strategy.py`

## 7. 模型融合

### 三层 Stacking 方案 (Deotte 2025 Playground 1st)

```
Level 1 (72个base模型):
  ├── LightGBM/XGBoost/CatBoost × 多组超参
  ├── ExtraTrees / RandomForest
  ├── Neural Network (MLP)
  ├── TabPFN, KNN, SVR, Ridge
  └── 全部用 RAPIDS cuML GPU加速
Level 2: XGBoost + NN + AdaBoost (L1 OOF + 原始特征)
Level 3: Weighted Mean
```

### 融合方法对比

| 方法 | 优点 | 缺点 | 适用 |
|------|------|------|------|
| 加权平均 | 简单快速 | 无法建模非线性 | 快速baseline |
| Rank Averaging | AUC最优 | 仅排序指标 | AUC评估 |
| Blending | 5x快于Stacking | 数据利用率低 | 大数据集 |
| Stacking | 最优精度 | 过拟合风险 | 金牌冲刺 |
| Hill Climbing | 贪心最优组合 | 计算量大 | 模型多时选子集 |

### 后处理技巧 (Deotte曾仅靠后处理获solo金牌)

| 指标 | 后处理 | 细节 |
|------|--------|------|
| AUC | Rank averaging | 转排名再平均, 绝对值无关 |
| LogLoss | clip + 缩放 | [0.01,0.99] + ×0.99 |
| Accuracy | Majority voting | AIMO: M次采样多数投票 / GenSelect |
| F1/QWK | 阈值优化 | CV上搜最优阈值 |
| 所有 | TTA | 多增强预测取均值(CV赛标配) |

详情 → `references/iteration_strategy.md`
脚本 → `scripts/ensemble_template.py`

## 8. LLM / Agent 赛道专项

### 数学推理赛 (AIMO 系列)

**AIMO2 NemoSkills 三大支柱**:
1. **OpenMathReasoning数据集**: 306k问题 + 5.5M训练样本(3.2M CoT + 1.7M TIR + 566k GenSelect)
2. **TIR迭代pipeline**: LLM生成Python代码 → 执行 → 反馈 → 继续推理
3. **GenSelect选择器**: 从64候选中选最优解, AIME24准确率93.3%(超majority voting)

**推理加速**:
- TensorRT-LLM + FP8(1.5x) + ReDrafter投机解码(1.8x) = 总计2.7x
- lmdeploy/TurboMind(AIMO2 2nd/5th/7th使用)
- W4KV8量化比FP16快55%

### AGI 推理赛 (ARC Prize)

**Test-Time Training 流程**:
```
对每个测试任务:
  1. 取示例 input-output 对
  2. 数据增强(几何变换 × 颜色置换) → 扩展样本
  3. LoRA 微调模型(几分钟)
  4. 预测测试 input 的 output
  5. 多 augmentation 预测 → ensemble
```

**核心思想**: 重计算离线(合成数据), 轻计算在线(小模型TTT)

### Agent 代码赛 (Konwinski Prize)

- 上下文管理是最关键组件
- Top5全部使用Qwen或DeepSeek模型
- Round1最高7.5%(vs SWE-Bench 75%) — 真实新issue远比benchmark难

### Test-Time Compute Scaling (ICLR 2025)

1. **并行扩展**: 生成N个解 → majority voting(AIMO: 64次采样)
2. **序列扩展**: CoT推理, 迭代修正(DeepSeek-R1, QwQ-32B)
3. **TTT**: 每任务微调(ARC所有top方案)
4. **GenSelect**: 训练选择器替代投票(AIMO2 1st)

**关键发现**: 小模型+高级推理算法 = Pareto最优(14B胜过更大模型)

详情 → `references/llm_agent_competitions.md`

## 9. 竞赛 Checklist

```
赛前(第1天):
  □ 阅读规则(约束条件极重要: GPU/时间/网络/外部数据/闭源API)
  □ 下载并理解数据结构
  □ 搭建端到端 Baseline(能提交)
  □ 建立可信的本地 CV
  □ 浏览 Top 20 Discussion + EDA Notebook
  □ 搜索历史相似赛题的获奖方案

赛中(每日):
  □ 实验日志(方法, CV, LB, 耗时)
  □ 新 Discussion/Notebook 检查
  □ CV-LB 一致性追踪

赛末(最后3天):
  □ 冻结新实验
  □ Ensemble/Stacking 优化
  □ 后处理微调
  □ 选择2个最终提交(1保守CV + 1激进LB)
  □ 确认提交格式正确
```

## 10. 常见问题速查表

| 问题 | 诊断 | 解决方案 |
|------|------|---------|
| 过拟合 | train↑ val↓ | 正则化/dropout/数据增强/减特征 |
| 欠拟合 | train和val都低 | 增加模型容量/更多特征/减正则化 |
| CV-LB不一致 | CV↑ LB↓ | 检查CV策略是否匹配test分布 |
| LB Shake-up | Pub好 Priv差 | 信任CV, 选保守方案(案例: 某选手升1700名) |
| OOM | GPU内存溢出 | FP16→GradAccum→GradCkpt→torch.compile |
| 代码赛超时 | 推理超限 | 蒸馏/量化/减TTA(BirdCLEF: CPU推理) |
| 外部模型变更 | LB格局巨变 | 关注Discussion白名单变化(AIMO2: DeepSeek R1) |
| Agent低准确率 | 复杂issue | 上下文管理优化(Konwinski: 最高7.5%) |
| 数据太大 | pandas慢 | polars/cuDF-pandas GPU加速 |
| LLM输出过长 | 超时/超显存 | DPO缩短输出/early stopping/动态速度调节 |
| 提交失败 | 格式错 | 对齐sample_submission列名/类型/行数/NaN |

详情 → `references/troubleshooting.md`

## 11. 工具库速查

### 2025 获奖方案使用频率 Top 库

| 库 | 次数 | 用途 |
|---|---|---|
| NumPy/Pandas | 61-62 | 数据处理 |
| PyTorch | 44 | 深度学习 |
| scikit-learn | 33 | ML算法/指标 |
| Transformers | 25 | NLP/LLM |
| timm | 13 | 预训练视觉模型 |
| XGBoost/LightGBM | 14 | GBDT |
| CatBoost | 8 | GBDT |
| W&B | 11 | 实验追踪 |
| Optuna | 6 | 超参搜索 |
| Polars | 6 | 快速数据处理(增长中) |
| RAPIDS cuML/cuDF | - | GPU加速ML/数据(GM核心武器) |
| AutoGluon | - | AutoML(2024: 7金牌) |
| TabPFN | - | 小数据表格(Stacking组件) |

### LLM 推理框架对比

| 框架 | 性能 | 竞赛使用 |
|------|------|---------|
| TensorRT-LLM | NVIDIA优化, FP8 | AIMO2 1st |
| lmdeploy | 高吞吐 | AIMO2 2nd/5th/7th |
| SGLang | ~16200 tok/s H100 | 增长中 |
| vLLM | ~12500 tok/s H100 | 通用 |

### LLM 量化方法对比

| 方法 | 加速 | 竞赛使用 |
|------|------|---------|
| FP8 | 1.5x vs FP16 | AIMO2 1st |
| AWQ 4bit | 大 | AIMO top方案通用 |
| 8bit KV cache | 节省长序列内存 | AIMO2 2nd/7th |
| QLoRA 4bit | 训练省内存 | KDD Cup 2024 1st |

### LoRA 最佳实践

- `target_modules='all-linear'` — MLP层LoRA优于仅attention(Biderman 2024)
- `lora_r=16, lora_alpha=16`, 同时训练embedding+lm_head
- LoRA adapter跨fold平均(LMSYS 1st验证有效)
- Spectrum(30% SNR层): +4% vs QLoRA on GSM8K
