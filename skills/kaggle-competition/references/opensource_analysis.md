# 开源方案分析框架

## 1. Discussion/Notebook 浏览策略

### 时间线策略

```
第1天（赛前调研）:
  ├── 按 Vote 数排序浏览 Top 20 Discussion
  ├── 重点关注:
  │   ├── 赛题 Overview/FAQ（主办方说明）
  │   ├── EDA Notebook（理解数据特性、分布、质量）
  │   ├── Baseline Notebook（快速起步）
  │   └── 评估指标讨论（优化策略、损失函数对齐）
  └── 记录关键发现到实验日志

每周更新:
  ├── 新发布的高分 Notebook
  ├── 标记为 "Useful" 的 Discussion
  ├── GM/Master 的新发帖（高价值信息）
  ├── 规则变更/数据更新公告
  └── 白名单/黑名单模型变化（LLM赛尤其重要）

赛末（最后1周）:
  ├── 不再追新方案，冻结已有方案
  └── 专注融合和后处理

赛后（学习阶段）:
  ├── 所有金牌方案 writeup
  ├── Discussion 中的 solution sharing
  ├── 提取可复用的 trick 到个人笔记
  └── 标记值得跟踪的 Kaggler
```

### 高效浏览技巧

- 按 "Most Votes" 排序, 比 "Recent" 更有信息密度
- 关注帖子中的 **CV 分数** — 有CV分数的帖子通常更可信
- GM/Master 的帖子优先级最高（他们通常不会分享误导信息）
- 关注评论区的修正和讨论, 常有额外洞察

---

## 2. 方案评估 5 维矩阵

| 维度 | 评估标准 | 权重 | 如何判断 |
|------|---------|------|---------|
| **分数** | Public LB + 声称的CV分数 | 高 | LB分数可见; CV分数取决于作者声明 |
| **可信度** | 作者等级 + 社区反馈 | 中 | GM>Master>Expert>Contributor; 看Vote数和评论 |
| **可复现性** | 代码完整度、依赖复杂度、运行时间 | 高 | 能否fork直接运行? 依赖是否复杂? |
| **差异化** | 与当前方案的互补性 | 高 | 使用了不同的trick? 不同的模型类型? |
| **融合价值** | 模型类型多样性 | 中 | GBDT vs NN vs LLM? 不同特征工程? |

### 评估打分模板

```
方案名称: ___
作者/等级: ___
LB分数: ___ / CV分数: ___

□ 可复现性: □可直接运行 □需小修改 □需大修改 □无法复现
□ 差异化: □完全不同 □部分不同 □与当前类似
□ 融合价值: □高(不同模型类型) □中(同类但不同超参) □低(高度重叠)
□ 风险: □低(成熟方法) □中(新trick) □高(可能过拟合)

关键trick提取:
1. ___
2. ___
3. ___
```

---

## 3. Trick 分类提取模板

### 数据类 Trick

| Trick | 描述 | 适用 | 案例 |
|-------|------|------|------|
| 数据增强 | 增加训练样本多样性 | CV/NLP | Mixup, CutMix, 翻转旋转 |
| 伪标签 | 用模型预测未标注数据, 加入训练 | 全部 | BirdCLEF 2024 3rd, Jigsaw 2025 |
| 外部数据 | 使用允许的额外数据 | 允许时 | LLM Science: 2.5TB Wikipedia |
| 清洗策略 | 去除噪声标签/异常样本 | 标签有噪声时 | Confident Learning |
| 合成数据 | 生成训练数据 | LLM/推理赛 | ARC Prize: 合成ARC风格任务 |

### 模型类 Trick

| Trick | 描述 | 适用 | 案例 |
|-------|------|------|------|
| 架构修改 | 修改模型结构 | CV/NLP | ARChitects: 2D-RoPE替换1D |
| 预训练权重 | 选择最优预训练 | 深度学习 | timm预训练, HuggingFace |
| 自定义Loss | 修改损失函数 | 指标与CE不对齐 | BirdCLEF: CE替代BCE |
| 多任务学习 | 同时优化多个目标 | 有辅助任务 | FathomNet: 层级分类 |
| 蒸馏 | 大模型→小模型 | 推理受限 | LMSYS: 70B→9B |

### 训练类 Trick

| Trick | 描述 | 适用 | 案例 |
|-------|------|------|------|
| CosineAnnealing | 余弦退火学习率 | 几乎所有 | 深度学习标配 |
| Warmup | 开始时低学习率 | 大模型 | transformer标配 |
| EMA | 指数移动平均权重 | 提升稳定性 | CV赛常用 |
| SWA | 随机权重平均 | 后期优化 | 提升泛化 |
| Label Smoothing | 标签平滑 | 防过拟合 | 分类任务 |
| Gradient Accum. | 梯度累积 | 显存不足 | batch_size×4 |

### 推理类 Trick

| Trick | 描述 | 适用 | 案例 |
|-------|------|------|------|
| TTA | 多增强预测平均 | CV赛 | ISIC: 必用 |
| 后处理 | 裁剪/阈值/排序 | 全部 | Deotte: 仅后处理获金 |
| 量化 | 减少模型精度 | LLM赛 | FP8/AWQ/GPTQ |
| 投机解码 | 小模型预测+大模型验证 | LLM赛 | AIMO2: ReDrafter 1.8x |
| Majority Voting | 多次推理投票 | 数学/推理 | AIMO: 64次采样 |

---

## 4. 开源方案快速复现 SOP

```
Step 1: Fork Notebook
  └─ Kaggle: 点击 "Copy & Edit"
  └─ GitHub: Fork + Clone

Step 2: 环境搭建
  └─ 检查 Python 版本和依赖
  └─ 对齐 GPU 类型和内存
  └─ 安装缺失包

Step 3: 数据路径修改
  └─ 修改 input/output 路径
  └─ 确认数据版本一致

Step 4: 运行验证
  └─ 先跑1 fold验证(快速检查)
  └─ 对比 CV 分数是否与作者声称一致
  └─ 如果偏差 > 1%, 检查数据处理差异

Step 5: 分数对齐
  └─ CV 分数对齐 → 方案可信
  └─ 提交 LB → 确认方案有效
  └─ 记录基线分数, 作为后续改进参照
```

---

## 5. 常用开源框架

| 框架 | 用途 | 赛道 |
|------|------|------|
| timm | 预训练视觉模型(2025: 13次获奖) | CV |
| HuggingFace Transformers | NLP/LLM模型(2025: 25次获奖) | NLP/LLM |
| LLaMA-Factory | LoRA/QLoRA微调(YAML配置) | LLM |
| vLLM / lmdeploy / SGLang | LLM推理加速 | LLM |
| TensorRT-LLM | NVIDIA推理优化 | LLM |
| albumentations | 图像增强 | CV |
| Optuna | 超参搜索(2025: 6次获奖) | 全部 |
| RAPIDS cuML/cuDF | GPU加速ML/数据 | 表格 |
| AutoGluon | AutoML(2024: 7金牌) | 表格 |
| Polars | 快速数据处理(2025: 6次获奖) | 全部 |
| NeMo RL + Skills | NVIDIA RL框架 | 数学/推理 |

---

## 6. 关键资源列表

| 资源 | URL | 用途 |
|------|-----|------|
| Kaggle Solutions | farid.one/kaggle-solutions | 最全方案集合, 按赛题搜索 |
| Competition Solutions | github.com/anuj0456/kaggle_competition_solutions | GitHub方案链接库 |
| ML Contests | mlcontests.com | 竞赛趋势年度报告 |
| Papers with Code | paperswithcode.com | 论文+代码+排行榜 |
| AIMO Prize | aimoprize.com | AIMO竞赛官方 |
| ARC Prize | arcprize.org | ARC竞赛官方+技术报告 |
| OpenMathReasoning | huggingface.co/datasets/nvidia/OpenMathReasoning | AIMO2训练数据 |
| NuminaMath | huggingface.co/blog/winning-aimo-progress-prize | AIMO1方案详解 |
