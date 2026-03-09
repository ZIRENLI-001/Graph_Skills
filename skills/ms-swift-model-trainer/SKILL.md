---
name: ms-swift-model-trainer
description: >
  使用 ms-swift (ModelScope SWIFT) 框架对 LLM/MLLM 进行 SFT、DPO、GRPO 训练。
  支持 LoRA/全参数微调 600+ 语言模型和 300+ 多模态模型。
  涵盖数据准备、格式验证、训练执行、模型导出全流程。
  当用户需要微调、训练、对齐大语言模型或多模态模型时激活此 Skill。
---

# ms-swift Model Trainer

使用 [ms-swift](https://github.com/modelscope/ms-swift) 框架训练和微调大语言模型（LLM）及多模态大模型（MLLM）。支持 SFT、DPO、GRPO 等训练方法，覆盖从数据准备到模型导出的完整流程。

## 1. 训练概述

ms-swift 是 ModelScope 社区提供的大模型微调与部署框架（AAAI 2025），v4.0+ 版本支持：

- **训练类型**: CPT（继续预训练）、SFT（监督微调）、DPO、GRPO、KTO、SimPO、ORPO、GKD 等
- **模型支持**: Qwen3/3.5、DeepSeek-R1、Llama4、InternLM3、GLM-5 等 600+ LLMs；Qwen3-VL、InternVL3.5 等 300+ MLLMs
- **训练方式**: LoRA、QLoRA、全参数微调、Megatron 并行训练
- **加速**: vLLM 推理加速（GRPO）、DeepSpeed ZeRO、序列并行

### 安装

```bash
pip install ms-swift
# 或从源码安装（获取最新功能）
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift && pip install -e .
```

## 2. 文档参考

- **官方文档**: https://swift.readthedocs.io/en/latest/
- **GitHub**: https://github.com/modelscope/ms-swift
- **训练方法选择** → `references/training_methods.md`
- **数据集格式详解** → `references/dataset_formats.md`
- **训练模式与参数** → `references/training_patterns.md`
- **硬件选型** → `references/hardware_guide.md`
- **模型导出与量化** → `references/model_export.md`
- **故障排查** → `references/troubleshooting.md`

## 3. 数据集验证（关键步骤）

**必须在训练前验证数据格式。** 超过 50% 的训练失败源于数据格式问题。DPO 尤其严格，要求精确的字段结构。

### 运行验证脚本

```bash
python scripts/dataset_validator.py --dataset_path /path/to/your/dataset.jsonl --task sft
python scripts/dataset_validator.py --dataset_path /path/to/your/dataset.jsonl --task dpo
python scripts/dataset_validator.py --dataset_path /path/to/your/dataset.jsonl --task grpo
```

### 验证输出示例

```
Dataset: /path/to/dataset.jsonl
Samples: 10,000
Format detected: messages

Compatibility:
  ✓ SFT    — READY (messages format with user/assistant roles)
  ✗ DPO    — NEEDS MAPPING (missing rejected_response field)
  ✗ GRPO   — NEEDS MAPPING (missing solution field)

Suggestion: Add 'rejected_response' field for DPO training.
```

### ms-swift 支持的四种数据格式

| 格式 | 必需字段 | 适用场景 |
|------|---------|---------|
| **messages**（推荐） | `messages: [{role, content}]` | 所有任务 |
| **shareGPT** | `conversation: [{human, assistant}]` | 多轮对话 |
| **alpaca** | `instruction, input, output` | 指令微调 |
| **query/response** | `query, response` | 简单问答 |

详见 → `references/dataset_formats.md`

## 4. 训练执行

### 4.1 SFT（监督微调）

**适用场景**: 有高质量示例数据（客服对话、代码生成、领域问答等），要让模型学习特定行为模式。

#### CLI 方式

```bash
# 单卡 LoRA 训练
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-en \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir output/sft_lora

# 多卡全参数训练
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-en \
    --train_type full \
    --deepspeed zero3 \
    --output_dir output/sft_full
```

#### Python API 方式

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import sft_main, TrainArguments

result = sft_main(TrainArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    dataset=['AI-ModelScope/alpaca-gpt4-data-en#1000'],
    train_type='lora',
    torch_dtype='bfloat16',
    num_train_epochs=3,
    learning_rate=1e-4,
    output_dir='output/sft_lora',
))
```

参考模板 → `scripts/train_sft_example.py`

### 4.2 DPO（直接偏好优化）

**适用场景**: 有偏好对数据（chosen/rejected），希望模型输出符合人类偏好。通常在 SFT 之后进行。

#### 数据格式要求

DPO 数据必须包含 `messages`（含 chosen response）和 `rejected_response` 字段：

```jsonl
{"messages": [{"role": "user", "content": "写一首诗"}, {"role": "assistant", "content": "优质回答..."}], "rejected_response": "较差回答..."}
```

#### CLI 方式

```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --train_type lora \
    --learning_rate 5e-6 \
    --output_dir output/dpo_lora
```

参考模板 → `scripts/train_dpo_example.py`

### 4.3 GRPO（组相对策略优化）

**适用场景**: 使用可验证的奖励信号（如数学题正确性）来优化模型推理能力。无需训练单独的奖励模型。

#### CLI 方式

```bash
# 使用 vLLM 加速推理（推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --train_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --num_generations 8 \
    --output_dir output/grpo_lora
```

#### 自定义 ORM（Outcome Reward Model）

GRPO 的额外数据字段会自动传递到 ORM 函数：

```python
def custom_orm(completions, solution, **kwargs):
    """自定义奖励函数。completions 为必需参数，其余为数据集额外字段。"""
    rewards = []
    for completion in completions:
        # 提取模型回答
        content = completion[0]['content'] if isinstance(completion, list) else completion
        # 与标准答案比较
        reward = 1.0 if solution.strip() in content else 0.0
        rewards.append(reward)
    return rewards
```

参考模板 → `scripts/train_grpo_example.py`

## 5. 训练监控

ms-swift 默认集成 TensorBoard，训练日志自动保存在 `output_dir/runs/` 目录。

### TensorBoard（默认）

```bash
tensorboard --logdir output/vx-xxx/runs/
```

### 可选集成

```bash
# 使用 Wandb
swift sft --model ... --report_to wandb

# 使用 SwanLab
swift sft --model ... --report_to swanlab
```

### 关键监控指标

- `train/loss` — 训练损失，应持续下降
- `train/learning_rate` — 学习率变化曲线
- `eval/loss` — 验证集损失，用于检测过拟合
- `train/grad_norm` — 梯度范数，异常值提示训练不稳定

## 6. 训练模式与 OOM 处理

### LoRA vs 全参数微调

| 模型规模 | 推荐方式 | 显存需求（LoRA） | 显存需求（Full） |
|---------|---------|----------------|----------------|
| < 3B | Full 或 LoRA | ~8GB | ~24GB |
| 3B-7B | LoRA | ~16GB | ~60GB |
| 7B-14B | LoRA | ~24GB | ~120GB |
| 14B+ | LoRA + QLoRA | ~24GB (4bit) | 不推荐单卡 |

### OOM 应对策略

1. **减小 batch_size**: `--per_device_train_batch_size 1`
2. **增大梯度累积**: `--gradient_accumulation_steps 8`
3. **启用 LoRA**: `--train_type lora`
4. **使用 QLoRA**: `--quant_bits 4`
5. **启用 gradient checkpointing**: `--gradient_checkpointing true`（默认开启）
6. **使用 DeepSpeed**: `--deepspeed zero2` 或 `--deepspeed zero3`

有效 batch_size = per_device_train_batch_size × gradient_accumulation_steps × GPU数

参考 → `references/training_patterns.md`

## 7. 训练前检查清单

在启动训练前确认以下事项：

- [ ] **数据集格式已验证** — 运行 `scripts/dataset_validator.py`
- [ ] **模型名称/路径正确** — 支持 ModelScope ID（默认）或 HuggingFace ID（加 `--use_hf true`）
- [ ] **显存充足** — 参考 `references/hardware_guide.md`
- [ ] **output_dir 已设置** — 避免覆盖之前的训练结果
- [ ] **DPO 前置**: 已有 SFT 基线模型，或使用预训练 Instruct 模型
- [ ] **GRPO 前置**: vLLM 已安装 (`pip install vllm`)，GPU 数量 ≥ 2（推荐 4+）

## 8. 模型保存与导出

### 本地保存

训练完成后，checkpoint 自动保存在 `output_dir` 中，包含：
- 模型权重（LoRA adapter 或完整权重）
- `args.json` — 训练参数记录
- tokenizer 文件

### Hub 推送

```bash
# 推送到 ModelScope Hub
swift sft --model ... --push_to_hub true --hub_model_id your-username/model-name

# 推送到 HuggingFace Hub
swift sft --model ... --push_to_hub true --hub_model_id your-username/model-name --use_hf true
```

### LoRA 合并导出

```bash
swift export --adapters output/vx-xxx/checkpoint-xxx --merge_lora true --output_dir output/merged
```

### 量化导出

```bash
# GPTQ 量化
swift export --model output/merged --quant_method gptq --quant_bits 4

# AWQ 量化
swift export --model output/merged --quant_method awq --quant_bits 4
```

参考 → `references/model_export.md`

## 9. 训练后推理

```bash
# 使用训练后的 adapter 推理（自动读取 args.json 中的模型和参数）
CUDA_VISIBLE_DEVICES=0 swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

# 使用合并后的模型推理
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model output/merged \
    --stream true
```

## 10. 故障排查

| 问题 | 解决方案 |
|------|---------|
| **CUDA OOM** | 减小 batch_size=1，增大 gradient_accumulation，启用 LoRA/QLoRA |
| **数据格式错误** | 运行 `scripts/dataset_validator.py` 检查格式 |
| **多卡训练挂起** | 检查 NCCL：`export NCCL_P2P_DISABLE=1`，确认 GPU 可见性 |
| **vLLM 报错（GRPO）** | 确认 vLLM 版本兼容，尝试 `--vllm_mode server` 替代 `colocate` |
| **Loss 不下降** | 检查学习率（LoRA 建议 1e-4，Full 建议 1e-5）、数据质量 |
| **eval_loss 上升** | 过拟合，减少 epoch 数或增大数据集 |

详见 → `references/troubleshooting.md`
