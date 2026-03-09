# 训练方法概览

ms-swift 支持多种训练方法，本文档帮助你选择适合的方法。

## 方法选择决策树

```
你有什么数据？
├── 有高质量输入-输出对 → SFT
├── 有偏好对（好/差回答） → DPO / SimPO / ORPO
├── 有可验证的奖励信号 → GRPO
├── 只有正/负标签 → KTO
└── 有老师模型蒸馏 → GKD
```

## SFT (Supervised Fine-Tuning) — 监督微调

### 原理
使用有标注的输入-输出对直接训练模型，让模型学习特定的行为模式。

### 适用场景
- 客服对话、代码生成、领域问答
- 让模型掌握特定输出风格或领域知识
- 作为 DPO/GRPO 等对齐训练的前置步骤

### 数据要求
- messages 格式（推荐）：`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- 建议样本数：1,000 - 100,000 条
- 质量 > 数量

### 典型参数
| 参数 | LoRA | Full |
|------|------|------|
| learning_rate | 1e-4 | 1e-5 |
| num_train_epochs | 3-5 | 2-3 |
| batch_size (effective) | 32-128 | 64-256 |
| warmup_ratio | 0.05 | 0.03 |

### CLI 示例
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset your_data.jsonl \
    --train_type lora \
    --output_dir output/sft
```

---

## DPO (Direct Preference Optimization) — 直接偏好优化

### 原理
通过偏好对（chosen/rejected）训练模型，使其生成更符合人类偏好的回答。DPO 将奖励模型和策略优化合并为一步，无需单独训练奖励模型。

### 适用场景
- 有人工标注的偏好数据
- 希望模型输出更安全、更有帮助
- SFT 后的精细调优

### 与 SFT 的关系
建议先进行 SFT 获得基线模型，再用 DPO 进行偏好对齐。也可以直接在 Instruct 模型上进行 DPO。

### 数据要求
- 必须包含 `rejected_response` 字段
- 格式：`{"messages": [..., {"role": "assistant", "content": "chosen"}], "rejected_response": "rejected"}`
- 建议样本数：5,000 - 50,000 条

### 典型参数
| 参数 | 推荐值 |
|------|--------|
| learning_rate | 5e-6 ~ 5e-7 |
| num_train_epochs | 1-3 |
| beta | 0.1 (默认) |
| warmup_ratio | 0.1 |

### CLI 示例
```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset dpo_data.jsonl \
    --train_type lora \
    --output_dir output/dpo
```

---

## GRPO (Group Relative Policy Optimization) — 组相对策略优化

### 原理
对每个 prompt 生成一组候选回答，使用可验证的奖励函数评估每个回答，然后基于组内相对排名更新策略。无需训练独立的奖励模型。

### 适用场景
- 数学推理（可验证答案正确性）
- 代码生成（可执行测试用例）
- 任何可以用规则验证的任务

### 与 PPO/DPO 的对比
| 特性 | GRPO | PPO | DPO |
|------|------|-----|-----|
| 需要奖励模型 | 否 | 是 | 否 |
| 需要偏好数据 | 否 | 否 | 是 |
| 在线生成 | 是 | 是 | 否 |
| 计算效率 | 高 | 低 | 高 |
| 适合推理任务 | 强 | 中 | 弱 |

### 数据要求
- 只需 prompt + 标准答案：`{"messages": [{"role": "user", "content": "问题"}], "solution": "答案"}`
- messages 中通常不包含 assistant 回复
- 额外字段自动传递给 ORM

### 奖励函数设计
ORM 函数签名：
```python
def my_orm(completions, solution, **kwargs):
    """completions: 必需参数; solution: 数据集额外字段"""
    rewards = []
    for completion in completions:
        reward = 1.0 if check_answer(completion, solution) else 0.0
        rewards.append(reward)
    return rewards
```

### 典型参数
| 参数 | 推荐值 |
|------|--------|
| learning_rate | 5e-6 ~ 1e-6 |
| num_generations | 4-16 |
| num_train_epochs | 1-2 |
| use_vllm | true (推荐) |
| vllm_mode | colocate |

### CLI 示例
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset math_data.jsonl \
    --train_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --output_dir output/grpo
```

---

## 其他 RLHF 方法（简介）

### KTO (Kahneman-Tversky Optimization)
- 只需正/负标签（不需要配对的 chosen/rejected）
- 数据格式：`{"messages": [...], "label": true/false}`
- `swift rlhf --rlhf_type kto`

### SimPO (Simple Preference Optimization)
- 无需参考模型，比 DPO 更高效
- 数据格式与 DPO 相同
- `swift rlhf --rlhf_type simpo`

### ORPO (Odds Ratio Preference Optimization)
- 将 SFT 和偏好对齐合并为一步
- 数据格式与 DPO 相同
- `swift rlhf --rlhf_type orpo`

### GKD (Generalized Knowledge Distillation)
- 用教师模型蒸馏学生模型
- `swift rlhf --rlhf_type gkd`

### DAPO / RLOO / Reinforce++
- GRPO 家族的变体算法
- `swift rlhf --rlhf_type dapo|rloo|reinforce_plus_plus`
