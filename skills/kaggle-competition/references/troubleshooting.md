# 常见问题与解决方案

## 1. 过拟合 / 欠拟合诊断

```
Train Loss ↓, Val Loss ↓ → 正常训练, 继续
Train Loss ↓, Val Loss ↑ → 过拟合!
Train Loss ↑, Val Loss ↑ → 学习率太大/模型有bug
Train Loss 平, Val Loss 平 → 欠拟合, 模型容量不足
```

### 过拟合解决方案

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| Early Stopping | 在val loss开始上升时停止 | 所有模型 |
| 降低模型复杂度 | 减少num_leaves/层数/参数 | 数据量小 |
| 正则化 | L1/L2, dropout, weight_decay | 所有 |
| 数据增强 | 增加训练样本多样性 | CV/NLP |
| 减少特征 | 删除噪声特征 | 表格赛 |
| 增加训练数据 | 伪标签、外部数据 | 数据不足 |
| Seed averaging | 多seed平均减少方差 | 所有 |
| Label smoothing | 防止模型过于自信 | 分类 |

### 欠拟合解决方案

| 方法 | 描述 |
|------|------|
| 增加模型容量 | 更多num_leaves/层数/参数 |
| 减少正则化 | 降低dropout/weight_decay |
| 更多特征 | 特征工程 |
| 降低学习率 | 让模型训练更充分 |
| 增加训练轮数 | 更多epochs |
| 换更强模型 | TabPFN→NN, DeBERTa→LLM |

---

## 2. CV-LB 不一致原因清单

| # | 原因 | 表现 | 解决方案 |
|---|------|------|---------|
| 1 | CV策略不匹配test | CV高LB低 | 用匹配test分布的CV策略 |
| 2 | 过拟合Public LB | LB高CV低 | 停止追LB, 信任CV |
| 3 | 时序泄漏 | CV异常高 | TimeSeriesSplit + gap |
| 4 | 分组泄漏 | CV异常高 | GroupKFold |
| 5 | 数据分布漂移 | 两者不稳定 | Adversarial Validation |
| 6 | Public LB样本少 | LB波动大 | 多次提交取趋势, 不追单次 |
| 7 | 评估指标实现差异 | 细微偏差 | 检查自定义metric是否与官方一致 |
| 8 | 后处理过拟合 | LB好CV差 | 后处理阈值只在CV上调 |

### GM 核心原则

> **永远信任CV, 不追LB**
>
> 案例: 某选手在ISIC 2024通过信任CV, 在Shake-up中上升~1700名

---

## 3. LB Shake-up 风险评估

### 风险因素

| 因素 | 高风险 | 低风险 |
|------|--------|--------|
| Public LB样本量 | < 20% test | > 40% test |
| 数据类型 | 时序/分组 | i.i.d |
| 类别平衡 | 极不平衡 | 平衡 |
| 评估指标 | 对极端值敏感(LogLoss) | 对排序鲁棒(AUC) |

### 防范策略

1. **两个最终提交**: 1个保守(最稳CV) + 1个激进(最高LB)
2. **多fold模拟**: 用不同CV fold模拟public/private split
3. **避免后处理过拟合**: 阈值只在CV上搜索, 不在LB上调
4. **关注CV趋势**: CV持续上升 > 单次LB跳跃

---

## 4. OOM 解决方案阶梯

按优先级从高到低尝试:

### Step 1: FP16 混合精度

```python
# PyTorch
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
**效果**: 显存减半, 速度略提升, 精度损失极小

### Step 2: Gradient Accumulation

```python
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    loss = model(input, target) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
**效果**: 等效batch_size×4, 显存不增加

### Step 3: Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```
**效果**: 节省~60%显存, 增加~20-30%训练时间

### Step 4: torch.compile

```python
model = torch.compile(model)  # PyTorch 2.0+
```
**效果**: 算子融合, 优化CUDA kernel, 减少峰值显存

### Step 5: 手动清理

```python
del variable
torch.cuda.empty_cache()
import gc; gc.collect()
```

### Step 6: 减少模型/数据

- 减小batch_size
- 减小输入分辨率/序列长度
- 使用更小模型
- 量化(FP8/4bit)

### Kaggle 硬件规格

| 资源 | 规格 |
|------|------|
| GPU RAM | 29GB (T4×2 或 P100) |
| GPU RAM | 96GB (L4×4, AIMO2) |
| CPU Cores | 4 |
| RAM | 30GB |
| 运行时间 | 通常9-12小时 |

---

## 5. LLM 特有问题

### 生成长度控制

| 问题 | 解决方案 |
|------|---------|
| 输出过长导致超时 | DPO训练缩短输出(AIMO2 2nd) |
| 输出过长导致OOM | max_tokens限制 + 动态截断 |
| Early stopping | 答案收敛4+次则停止(AIMO2 1st) |
| 动态调速 | 根据剩余时间调整采样数(AIMO2 2nd adjust_speed) |

### 幻觉减少

| 方法 | 描述 |
|------|------|
| TIR | 让模型生成代码验证计算(AIMO核心) |
| Self-consistency | 多次采样, 答案一致性检查 |
| CoT + 代码双通道 | 16次文本推理 + 16次代码求解(AIMO2 2nd) |

### 格式对齐

```python
# 答案提取正则
import re

def extract_answer(text):
    """从LLM输出中提取数值答案"""
    # 尝试多种格式
    patterns = [
        r'\\boxed\{(\d+)\}',           # LaTeX boxed
        r'[Aa]nswer[:\s]*(\d+)',        # "Answer: 123"
        r'[Tt]he answer is[:\s]*(\d+)', # "The answer is 123"
        r'(\d+)\s*$',                   # 末尾数字
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None
```

---

## 6. Agent 特有问题

| 问题 | 解决方案 |
|------|---------|
| 上下文窗口溢出 | 动态截断, 只保留最相关代码片段 |
| 工具调用失败 | 重试 + fallback策略 |
| 环境交互超时 | 设置timeout + 降级策略 |
| 修复patch不正确 | 测试验证 + 多次尝试 |
| 问题定位错误 | 多策略搜索(文本匹配+AST分析+LLM理解) |

---

## 7. 代码赛调试

### 本地模拟 Kaggle 环境

```bash
# Docker模拟Kaggle环境
docker pull gcr.io/kaggle-gpu-images/python
docker run --gpus all -v $(pwd):/kaggle/working -it kaggle/python

# 或者直接在Kaggle Notebook中调试
# 提示: 先在小数据上验证, 再全量运行
```

### 离线包管理

```python
# 在有网环境下载包
pip download -d ./packages transformers torch

# 在离线Kaggle Notebook中安装
!pip install --no-index --find-links=./packages transformers
```

### 常见代码赛错误

| 错误 | 原因 | 解决 |
|------|------|------|
| ImportError | 包不在Kaggle环境中 | 提前下载上传为Dataset |
| CUDA OOM | 模型太大 | 量化/减小batch/gradient ckpt |
| Timeout | 推理太慢 | 量化/投机解码/减少采样次数 |
| FileNotFoundError | 路径错误 | 使用 `/kaggle/input/` 前缀 |
| 提交格式错 | 列名/类型不匹配 | 对齐sample_submission.csv |

---

## 8. 提交格式排查清单

```
□ 文件名是否正确? (通常 submission.csv)
□ 列名是否与 sample_submission.csv 完全一致?
□ 行数是否匹配?
□ ID列是否完整且无重复?
□ 预测值类型是否正确? (float vs int)
□ 是否有NaN值? (通常不允许)
□ 预测值是否在合理范围? (概率应在[0,1])
□ 文件编码是否为UTF-8?
```

---

## 9. Deadline 前策略

### 72小时前

```
□ 冻结所有新实验
□ 确认最优单模型和Ensemble
□ 开始最终Stacking
□ 准备两个候选提交
```

### 24小时前

```
□ 完成所有Ensemble
□ 后处理微调(仅在CV上)
□ 测试提交成功(格式正确)
□ 选定最终2个提交
```

### 6小时前

```
□ 不要做任何大的改动!
□ 确认提交状态正常
□ 备份代码和模型
□ 提前选好最终提交, 不要最后一刻改
```

### 常见翻车

- 最后一刻改提交 → 选了个过拟合的版本
- 最后一刻Ensemble → 代码bug导致分数暴跌
- 没提前测试 → 提交格式错误, 来不及修
- 选了LB最高 → Shake-up后大幅下降

---

## 10. 大数据集处理优化

### Polars vs Pandas

| 操作 | Pandas | Polars | 加速比 |
|------|--------|--------|--------|
| 读取CSV(1GB) | 15s | 3s | 5x |
| GroupBy聚合 | 8s | 0.8s | 10x |
| Join操作 | 12s | 1.5s | 8x |
| 内存占用 | 高 | 低(零拷贝) | 2-4x |

### cuDF-pandas

```python
# 一行切换, 所有pandas操作GPU加速
%load_ext cudf.pandas
import pandas as pd  # 自动GPU版

# 效果: 10000+特征生成 小时→分钟
# Deotte 2025 Playground 1st 核心加速手段
```

### 大文件处理

```python
import polars as pl

# 懒加载: 不立即读入内存
df = pl.scan_parquet('huge_data.parquet')

# 分块处理
for chunk in pd.read_csv('huge.csv', chunksize=100000):
    process(chunk)

# Parquet格式: 比CSV小5-10x, 读取快10x
df.to_parquet('data.parquet', compression='snappy')
```
