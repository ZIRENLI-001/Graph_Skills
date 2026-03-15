# 方案制定指南

## 1. Baseline 快速搭建模板

### 1.1 表格赛 Baseline（30分钟可提交）

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features = [c for c in train.columns if c not in ['id', 'target']]
target = 'target'

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train[target])):
    X_tr, X_va = train.loc[tr_idx, features], train.loc[va_idx, features]
    y_tr, y_va = train.loc[tr_idx, target], train.loc[va_idx, target]

    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42+fold
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(test[features])[:, 1] / 5

# 评估 + 提交
from sklearn.metrics import roc_auc_score
print(f'CV AUC: {roc_auc_score(train[target], oof_preds):.5f}')
submission = pd.DataFrame({'id': test['id'], 'target': test_preds})
submission.to_csv('submission.csv', index=False)
```

### 1.2 CV 赛 Baseline（1小时可提交）

```python
import timm
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A

# 使用 timm 预训练模型
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)

# 基础增强
transform_train = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
])

# 标准训练循环: AdamW + CosineAnnealingLR + 混合精度
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = torch.cuda.amp.GradScaler()  # FP16混合精度
```

### 1.3 NLP 赛 Baseline（1小时可提交）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)
trainer = Trainer(model=model, args=training_args, ...)
```

### 1.4 LLM 赛 Baseline

```python
from vllm import LLM, SamplingParams

model = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            quantization="awq", dtype="float16",
            max_model_len=4096, gpu_memory_utilization=0.9)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048)

for problem in problems:
    outputs = model.generate([problem['prompt']], sampling_params)
    answer = extract_answer(outputs[0].outputs[0].text)
```

---

## 2. 算法选型详解

### 2.1 GBDT 三件套对比

| 维度 | LightGBM | XGBoost | CatBoost |
|------|----------|---------|----------|
| **速度** | ★★★★★ | ★★★ | ★★★ |
| **精度** | ★★★★ | ★★★★★ | ★★★★ |
| **类别特征** | 内置(有限) | 需编码 | ★★★★★(原生) |
| **小数据** | ★★★ | ★★★ | ★★★★★ |
| **大数据** | ★★★★★ | ★★★★ | ★★★ |
| **GPU支持** | ✓ | ✓ | ✓ |
| **2025获奖** | 14次 | 14次 | 8次 |
| **适用场景** | 默认首选, 大数据集 | 精细调参, 需要高精度 | 类别特征多, 小数据 |

### LightGBM 推荐超参搜索空间 (Optuna)

```python
params = {
    'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
    'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 15, 127),
    'max_depth': trial.suggest_int('max_depth', 3, 12),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
}
```

### 2.2 AutoGluon（2024 表格赛 7 金牌）

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target', eval_metric='roc_auc')
predictor.fit(train_data, time_limit=3600, presets='best_quality')
predictions = predictor.predict_proba(test_data)
```

- 2024年15/18表格赛获奖
- 曾击败Optuna调优的XGBoost+LightGBM+TabM集成
- 适合快速获得强baseline

### 2.3 Vision 模型选型

| 模型 | 参数量 | 精度 | 速度 | 适用场景 |
|------|--------|------|------|---------|
| EfficientNet-B0 | 5M | ★★★ | ★★★★★ | 快速baseline |
| EfficientNet-B5/B7 | 30M/66M | ★★★★ | ★★★ | 通用分类 |
| ConvNeXt-Large | 198M | ★★★★★ | ★★★ | 高精度 |
| ViT-Large | 304M | ★★★★★ | ★★ | 大数据集 |
| EfficientViT | 可变 | ★★★★ | ★★★★★ | 推理受限(CPU) |

**timm 库**: 2025年13次出现在获奖方案中, 是CV赛标准工具

### 2.4 NLP 模型选型

| 模型 | 参数量 | 精度 | 推理成本 | 适用场景 |
|------|--------|------|---------|---------|
| DeBERTa-v3-large | 300M | ★★★★★ | ★★★★★ | NLP分类之王, 媲美70B |
| ModernBERT | ~150M | ★★★★ | ★★★★★ | DeBERTa潜在替代 |
| Gemma2-9B | 9B | ★★★★ | ★★★ | 蒸馏目标模型 |
| Qwen2.5-72B | 72B | ★★★★★ | ★ | Teacher模型(QLoRA) |
| DeepSeek-R1 | 多种 | ★★★★★ | ★★ | 数学/推理 |

---

## 3. 真实获奖方案详解

### 3.1 AIMO1 — Numina (29/50, 1st)

**架构**: DeepSeek-Math-Base 7B → 两阶段SFT

```
Stage 1: CoT SFT
  └─ NuminaMath-CoT数据集(~1M数学问题+文本解答)
  └─ 输出: NuminaMath-7B-CoT

Stage 2: TIR SFT
  └─ 从NuminaMath-CoT选~60k数值答案题
  └─ GPT-4生成TORA格式推理路径(代码+执行+结果)
  └─ 过滤错误答案, 重复3轮
  └─ 输出: NuminaMath-7B-TIR
```

**推理 — SC-TIR**:
```
对每个问题:
  1. 复制M份(majority voting宽度, 如M=256)
  2. 采样completion直到产生Python代码块
  3. 执行代码, 拼接输出(含traceback)
  4. 重复N次(推理深度)
  5. 提取数值答案 → 多数投票
```

**关键**: 8×H100训练10小时, 数据集800k+, 验证集选自AMC/AIME/MATH

### 3.2 AIMO2 — NemoSkills (34/50, 1st, $262k)

**三大支柱**:

**Pillar 1: OpenMathReasoning数据集**
- 306k唯一数学问题(AoPS论坛)
- Qwen2.5-32B-Instruct做预处理
- DeepSeek-R1 + QwQ-32B 生成解(temp=0.7, top-p=0.95, max 16384 tokens)
- 每题最多32候选 → 3.2M长推理CoT解

**Pillar 2: TIR迭代Pipeline**
- 1.7M TIR解通过迭代训练-生成-过滤创建
- 直接prompt推理模型做TIR失败 → 需要迭代pipeline
- 主SFT后轻量TIR微调(15k样本)
- CoT和TIR checkpoint线性合并

**Pillar 3: GenSelect选择器**
- QwQ-32B训练, 从2-16个候选摘要中选最优解
- 摘要由Qwen2.5-32B-Instruct生成(4候选/解, max 2048 tokens)
- AIME24准确率: GenSelect 93.3% vs majority voting基线
- 566k GenSelect训练样本

**训练细节**:
```
基座: Qwen2.5-14B-Base
SFT: 6 epochs on 5.5M样本(3.2M CoT + 1.7M TIR + 566k GenSelect)
RoPE base: 改为500K
第二轮SFT: 仅高难度奥赛题, 过滤>5000 token的解
Kaggle模型: 2.2M CoT + 15K TIR
```

**推理加速**:
```
TensorRT-LLM + FP8量化: 1.5x speedup vs FP16
ReDrafter投机解码: 3-token proposal, 65%接受率, 1.8x speedup
总计: 2.7x加速
Early stopping: 答案收敛4+次则停止
时间缓冲: 管理5小时窗口
```

### 3.3 AIMO2 2nd — imagination-research (清华/MSR)

```
基座: DeepSeek-R1-Distill-Qwen-14B-AWQ
训练: Stage1 SFT → Stage2 DPO(缩短输出长度)
推理: lmdeploy/TurboMind引擎
量化: W4KV8(4bit权重 + 8bit KV cache) → 比FP16快55%
采样: 32次/题(16直接推理 + 16代码求解)
Early stopping: 5/7答案一致则停止
动态调节: adjust_speed模块, 根据剩余时间/题目动态调整
```

### 3.4 ARC Prize 2025 — NVARC (24%, 1st)

```
模型: Qwen-2-VL-4B → 极简tokenizer(仅16 token: 0-9颜色+格式)
合成数据:
  └─ 概念分解 → 基础任务生成 → 组合复杂任务
  └─ 用Qwen等模型验证 → 渐进难度 → 百万级语料
TTT:
  └─ 每任务独立微调(几分钟)
  └─ 几何变换 + 颜色置换数据增强
约束: 4×小GPU, 12h, ~$0.20/task
核心: 重计算离线(合成数据), 轻计算在线(小模型TTT)
工具: NeMo RL + NeMo Skills
```

### 3.5 ARC Prize 2025 — ARChitects (16.5%, 2nd)

```
模型: LLaDA-8B masked diffusion LLM
架构创新:
  └─ 2D-RoPE: 替换原始1D位置编码, 适配2D网格
  └─ Masked diffusion: de-mask方式生成解答
TTT:
  └─ Per-task独立微调(不混合多任务)
  └─ LoRA微调(比全量更稳定)
  └─ 损失函数改进: 更好利用multi-example格式
```

### 3.6 LMSYS Chatbot Arena — sayoulala (1st, $100k)

```
蒸馏Pipeline:
  Teacher: Llama3-70B-Instruct + Qwen2-72B-Instruct (LoRA微调)
  Student: Gemma2-9B-it (LoRA微调)
  Loss: CrossEntropyLoss + KLDivLoss + CosineEmbeddingLoss

训练:
  5-fold CV with LoRA
  关键trick: LoRA adapter跨fold平均
  硬件: 8×A100 80GB

推理: 8-bit量化

代码: github.com/shyoulala/LMSYS_BlackPearl
```

### 3.7 LLM Science Exam — H2O (1st)

```
模型: 5×7B LLM + 1×13B LLM (LoRA微调)
RAG: 2.5TB Wikipedia数据, 多种检索方式:
  ├── BM-25
  ├── Elasticsearch
  ├── BERT embeddings
  └── LLMs reranking (MTEB leaderboard模型)

关键发现: DeBERTa-v3(300M) + 好检索 ≈ 70B LLM
4th place仅用DeBERTa + Elasticsearch, token长度1280即可
```

### 3.8 Konwinski Prize — Round1 1st (7.5%)

```
模型: Qwen2.5-Coder-32B
核心: 上下文管理(最关键组件)
约束: 禁止闭源API, 纯开源
Top5: 全部使用Qwen或DeepSeek
差距: 7.5% vs SWE-Bench 75% → 真实新issue远比benchmark难
```

### 3.9 Playground 2025 — Deotte (1st)

```
模型: 72个base模型三层Stacking
Level 1: XGBoost×N, LightGBM×N, CatBoost×N, ET, RF, NN, TabPFN, KNN, SVR, Ridge
Level 2: XGBoost + NN + AdaBoost (输入: L1 OOF + 原始特征)
Level 3: Weighted Mean
加速: RAPIDS cuML GPU加速所有sklearn兼容模型
```

---

## 4. LoRA/QLoRA 关键配置

### 推荐配置

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",  # MLP层LoRA优于仅attention (Biderman 2024)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 同时训练embedding和lm_head层
model.enable_input_require_grads()
```

### 方法对比

| 方法 | 特点 | 精度 | 速度 |
|------|------|------|------|
| LoRA(all-linear) | 所有线性层 | ★★★★★ | ★★★★ |
| QLoRA 4bit | 4bit基座+LoRA | ★★★★ | ★★★ |
| Spectrum(30% SNR) | 选30%高SNR层 | ★★★★(+4% vs QLoRA) | ★★★★ |
| CLoQ INT2 | 无需反向传播 | ★★★★ | ★★★★★ |
| LoRA adapter平均 | 跨fold平均权重 | 提升稳定性 | - |

### 蒸馏 Pipeline (LMSYS 1st 验证)

```
Step 1: Teacher训练
  └─ 70B模型 + QLoRA微调 (8×A100)
Step 2: 生成蒸馏数据
  └─ Teacher推理 → 收集logit分布
Step 3: Student训练
  └─ 9B模型 + LoRA
  └─ Loss = CE(hard labels) + KL(teacher logits) + Cosine(representations)
Step 4: 量化推理
  └─ Student → 8bit/4bit量化 → 部署
```
