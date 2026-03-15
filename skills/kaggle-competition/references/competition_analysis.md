# 赛题解析方法论

## 1. 赛题五要素分析法

### 1.1 目标理解

| 任务类型 | 描述 | 典型赛题 |
|---------|------|---------|
| 二分类 | 预测0/1 | ISIC 2024(皮肤癌), Home Credit |
| 多分类 | 预测N个类别 | BirdCLEF(182种鸟), FathomNet |
| 回归 | 预测连续值 | Playground(播客时长, 卡路里) |
| 排序 | 输出偏好排序 | LMSYS Chatbot Arena |
| 生成 | 生成文本/代码 | AIMO(数学答案), Konwinski(代码patch) |
| 推理 | 抽象推理 | ARC Prize(视觉pattern) |
| 优化 | 最小化/最大化目标 | Santa 2024(最小化困惑度) |
| Agent | 自主决策执行 | Konwinski(修复GitHub issue), AgentSociety |

### 1.2 评估指标深度分析

| 指标 | 数学定义 | 优化策略 | 后处理 | 典型赛题 |
|------|---------|---------|--------|---------|
| **AUC-ROC** | TPR vs FPR曲线下面积 | BCE loss | Rank averaging(绝对值无关) | ISIC 2024 |
| **LogLoss** | -Σ[y·log(p)+(1-y)·log(1-p)] | BCE loss | clip[0.01,0.99], ×0.99校准 | LMSYS Arena |
| **RMSE** | √(Σ(y-ŷ)²/n) | MSE loss | 无特殊处理 | Home Credit |
| **MAE** | Σ|y-ŷ|/n | L1 loss / Huber | 中位数预测更优 | - |
| **F1-macro** | 各类F1均值 | BCE per class | 每类独立阈值搜索 | Jigsaw 2025 |
| **F1-micro** | 全局TP/FP/FN | BCE | 全局阈值搜索 | - |
| **QWK** | 加权Kappa系数 | MSE + 阈值优化 | OptimizedRounder | 作文评分 |
| **MAP@K** | K个预测的平均精度 | 自定义排序loss | Top-K选择+置信度阈值 | LLM Science Exam |
| **Accuracy** | 正确预测比例 | CE loss | Majority voting | AIMO, ARC |
| **MCC** | Matthews相关系数 | BCE | 阈值搜索(平衡FP/FN) | - |
| **Gini Stability** | Gini系数+时间稳定性 | MSE + 时序特征 | 时间段加权 | Home Credit 2024 |

### 指标陷阱

- **AUC**: 不关心绝对值, 只看排序 → Rank averaging是最优融合方式
- **LogLoss**: 对极端错误惩罚极大 → 必须clip, 标签噪声时乘以0.99
- **QWK**: 连续预测需要离散化 → OptimizedRounder在CV上搜最优阈值
- **F1-macro**: 各类别权重相等 → 少数类也很重要, per-class阈值必须独立搜索
- **MAP@K**: 排序位置很重要 → 高置信度预测放前面
- **Gini Stability**: 惩罚时间维度性能衰退 → 特征工程需考虑时间稳定性

### 1.3 约束条件分析清单

```
□ 竞赛类型: Featured / Code Competition / Playground / Research
□ 代码竞赛约束:
  ├── GPU类型和数量? (P100/T4/L4/H100, 1-4卡)
  ├── 总运行时间? (通常9-12小时)
  ├── 是否可联网? (通常不可)
  ├── 可用Python包版本?
  └── 提交格式? (Notebook必须能完整运行)
□ 外部数据/模型:
  ├── 是否允许外部数据?
  ├── 模型白名单/黑名单? (AIMO2: DeepSeek R1白名单后格局巨变)
  ├── 是否禁止闭源API? (Konwinski: 禁止)
  └── 预训练模型来源限制?
□ 推理约束:
  ├── 推理时间限制?
  ├── CPU-only推理? (BirdCLEF)
  └── 内存限制?
□ 团队规则:
  ├── 最大团队人数?
  ├── 每日提交次数?
  └── 合并截止日期?
```

### 1.4 历史相似赛题检索

**搜索路径**:
1. [farid.one/kaggle-solutions](https://farid.one/kaggle-solutions/) — 按关键词/类型搜索
2. [kaggle_competition_solutions](https://github.com/anuj0456/kaggle_competition_solutions) — GitHub链接库
3. Kaggle Competition页面 → Past Competitions → 按标签筛选
4. [mlcontests.com](https://mlcontests.com/) — 趋势分析

**匹配维度**:
- 数据类型匹配(表格/图像/文本/音频)
- 评估指标匹配
- 约束条件匹配(代码赛/GPU限制)
- 数据规模匹配

---

## 2. 经典赛题案例库

### 2.1 表格赛

| 赛题 | 年份 | 奖金 | 获奖方案要点 |
|------|------|------|------------|
| **Playground S5 Podcast** | 2025 | - | Deotte 1st: 72模型三层Stacking, RAPIDS cuML GPU加速, TabPFN组件 |
| **Home Credit** | 2024 | $105k | Gini Stability指标, LightGBM+CatBoost, 时序特征工程 |
| **Optiver Trading** | 2024 | $100k | 金融时序, 滞后特征+滚动统计, LightGBM |
| **Playground各期** | 2025 | - | 2648-4381队, AutoGluon有竞争力 |

### 2.2 CV/图像赛

| 赛题 | 年份 | 获奖方案要点 |
|------|------|------------|
| **ISIC 2024** | 2024 | 多Vision模型重度Ensemble + TTA |
| **FathomNet 2025** | 2025 | ViT + MCEAM多尺度注意力 + 分类学层级辅助分类 |
| **BirdCLEF 2024** | 2024 | CE loss替代BCE, sigmoid推理, 音频分段; 3rd: 伪标签+EfficientViT蒸馏(CPU) |

### 2.3 NLP/LLM 赛

| 赛题 | 年份 | 获奖方案要点 |
|------|------|------------|
| **LMSYS Arena** | 2024 | 1st: Llama3-70B+Qwen2-72B teacher → 蒸馏Gemma2-9b, LoRA adapter平均, 8×A100 |
| **LLM Science Exam** | 2024 | 1st: 5×7B+1×13B LoRA+RAG(2.5TB Wikipedia); 4th: DeBERTa-v3(300M)媲美70B |
| **Jigsaw 2025** | 2025 | NTT DOCOMO: 13个AI模型+伪标签+per-rule专用模型 |

### 2.4 数学/推理赛

| 赛题 | 年份 | 获奖方案要点 |
|------|------|------------|
| **AIMO1** | 2024 | Numina: DeepSeek-Math-7B 两阶段SFT(CoT→TIR), SC-TIR推理, 800k+数据集 |
| **AIMO2** | 2025 | NemoSkills: OpenMath-Nemotron-14B, OpenMathReasoning数据集(306k/3.2M), GenSelect, FP8+ReDrafter=2.7x |
| **AIMO3** | 2025-26 | 进行中: H100, GPT-OSS-120B(43/50), 国家→IMO难度, 5位数答案 |
| **ARC 2024** | 2024 | ARChitects(53.5%): NeMo-8B+TTT, 转导+归纳ensemble |
| **ARC 2025** | 2025 | NVARC(24%): Qwen-4B+合成数据+TTT, 极简16-token tokenizer |

### 2.5 Agent/代码赛

| 赛题 | 年份 | 获奖方案要点 |
|------|------|------------|
| **Konwinski Prize** | 2025 | Round1 1st(7.5%): Qwen2.5-Coder-32B, 上下文管理核心, Top5全Qwen/DeepSeek |
| **Santa 2024** | 2024 | 组合优化(非ML): TSP启发式, deletion+reinsertion, k-opt, double-bridge |
| **LLM Zoomcamp** | 2024 | Claude-3.5: zero-shot CoT + 代码生成执行 |

### 2.6 多模态/Agent社会

| 赛题 | 年份 | 获奖方案要点 |
|------|------|------------|
| **FathomNet 2025** | 2025 | ViT + 多尺度输入 + 分类学层级 |
| **AgentSociety** | 2025 | WWW2025, LLM Agent用户建模+推荐 |

---

## 3. AIMO 演进分析

### 3.1 分数演进
```
AIMO1: 29/50 (Private) — AMC12/AIME难度
AIMO2: 34/50 (Private) — 国家奥赛难度, 7队超29分
AIMO3: 44/50 (Public LB) — 国家→IMO难度, 最后6题极难
商业模型: 47/50 (AIMO2题目, OpenAI o3首次尝试)
```

### 3.2 模型演进
```
AIMO1: DeepSeek-Math-7B (专用数学模型)
AIMO2: Qwen2.5-14B (通用基座+数学数据SFT)
       DeepSeek-R1-Distill-Qwen-14B-AWQ (2nd place)
       DeepSeek R1白名单后LB巨变
AIMO3: GPT-OSS-120B (OpenAI开源, Apache2.0)
       Qwen3-235B-A22B (MoE)
       Qwen3.5-27B
       DeepSeek-R1-0528 (AIME2025 87.5%)
       DeepSeekMath-V2 (IMO2025金牌)
```

### 3.3 策略演进
```
AIMO1: 两阶段SFT + SC-TIR(majority voting)
AIMO2: 大规模数据(5.5M) + TIR迭代 + GenSelect(替代voting)
       + 推理加速(FP8+投机解码) + 动态时间管理
AIMO3: 更大模型(120B) + H100硬件 + weighted entropy
       + 更多推理时间 → test-time compute scaling
```

### 3.4 关键转折点

1. **AIMO1→2**: 数据规模从60k→306k问题, 解从几十万→3.2M, 数据质量决定上限
2. **AIMO2中期**: DeepSeek R1被白名单, LB格局巨变 → 关注规则变化
3. **AIMO2→3**: 硬件从L4→H100, 可跑120B模型, 计算约束大幅放松
4. **趋势**: 闭源vs开源差距从18/50缩小, AIMO3目标是进一步缩小
