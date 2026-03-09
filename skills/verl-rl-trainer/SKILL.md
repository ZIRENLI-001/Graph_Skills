---
name: verl-rl-trainer
description: >
  使用 verl (Volcano Engine RL) 框架对 LLM 进行强化学习后训练。
  支持 PPO、GRPO、DAPO、REINFORCE++ 等多种 RL 算法。
  基于 Ray 分布式架构，集成 vLLM/SGLang 推理加速和 FSDP/Megatron 训练后端。
  当用户需要对大语言模型进行 RL 强化学习训练、RLHF 对齐时激活此 Skill。
---

# verl RL Trainer

使用 [verl](https://github.com/verl-project/verl) 框架对大语言模型进行强化学习后训练（RL Post-Training）。verl 由字节跳动 Seed MLSys 团队开发，采用 HybridFlow 架构（EuroSys 2025），支持 PPO、GRPO、DAPO 等算法，可扩展至 671B 模型和数百 GPU。

## 1. 训练概述

verl 是面向 LLM 的生产级 RL 后训练框架，核心特性：

- **RL 算法**: PPO、GRPO、GSPO、DAPO、DrGRPO、ReMax、REINFORCE++、RLOO、PRIME
- **训练后端**: PyTorch FSDP、FSDP2、Megatron-LM
- **推理引擎**: vLLM、SGLang、HF Transformers（用于 rollout 生成）
- **模型支持**: Qwen-3、Qwen-2.5、Llama3.1、Gemma2、DeepSeek、InternLM 等
- **奖励类型**: 基于模型的奖励（Reward Model）和基于函数的可验证奖励（数学/代码）
- **分布式**: 基于 Ray 的弹性分布式训练，支持多节点扩展
- **监控**: wandb、mlflow、tensorboard、swanlab

### 安装

```bash
# 方式一：Docker（推荐）
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
  --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl && docker exec -it verl bash
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .

# 方式二：Conda + Pip
conda create -n verl python==3.12
conda activate verl
git clone https://github.com/volcengine/verl && cd verl
bash scripts/install_vllm_sglang_mcore.sh
# 仅 FSDP（不含 Megatron）：
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

核心依赖：`torch>=2.0.0`、`ray>=2.41.0`、`vllm>=0.8.2`、`transformers>=4.40.0`、`hydra-core`、`flash-attn`

## 2. 文档参考

- **官方文档**: https://verl.readthedocs.io/
- **GitHub**: https://github.com/verl-project/verl
- **算法详解** → `references/algorithms.md`
- **配置参数详解** → `references/config_guide.md`
- **硬件与分布式** → `references/hardware_distributed.md`
- **故障排查** → `references/troubleshooting.md`

## 3. 数据准备（关键步骤）

verl 使用 **Parquet 格式** 的数据文件。训练前必须将数据预处理为正确格式。

### 数据格式要求

每条数据需包含 `prompt` 字段（chat 模板格式的列表）：

```json
{
  "data_source": "gsm8k",
  "prompt": [
    {"role": "system", "content": "你是一个数学助手"},
    {"role": "user", "content": "计算 2+3=?"}
  ],
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "5"},
  "extra_info": {"solution": "5"}
}
```

### 运行数据预处理

```bash
# GSM8K 数学数据集
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

# 自定义数据集
python3 scripts/data_prepare_example.py \
    --input_path /path/to/your/data.jsonl \
    --output_dir ~/data/custom \
    --task_type math
```

参考模板 → `scripts/data_prepare_example.py`

### 数据格式验证

```bash
python3 scripts/dataset_validator.py --dataset_path ~/data/gsm8k/train.parquet
```

## 4. 训练执行

**统一入口**: 所有算法均通过 `python -m verl.trainer.main_ppo` 启动，通过 Hydra 配置选择算法。

### 4.1 GRPO（组相对策略优化）— 推荐

**适用场景**: 使用可验证奖励（数学正确性、代码测试通过率）优化推理能力。无需训练 Critic 模型，资源需求更低。

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    trainer.n_gpus_per_node=4 \
    trainer.logger=['console','wandb']
```

参考模板 → `scripts/train_grpo_example.py`

### 4.2 PPO（近端策略优化）

**适用场景**: 有训练好的 Reward Model，进行经典 RLHF 训练。需要额外的 Critic 模型。

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.ppo_micro_batch_size=2 \
    trainer.n_gpus_per_node=4 \
    trainer.logger=console
```

参考模板 → `scripts/train_ppo_example.py`

### 4.3 其他算法

通过 `algorithm.adv_estimator` 切换：

| 算法 | 参数值 | 说明 |
|------|--------|------|
| PPO | `gae` | 经典 RLHF，需要 Critic |
| GRPO | `grpo` | 组相对优化，无需 Critic |
| REINFORCE++ | `reinforce_plus_plus` | 改进的 REINFORCE |
| RLOO | `rloo` 或 `rloo_vectorized` | Leave-One-Out 基线 |
| DAPO | 基于 GRPO 配置 | 动态采样的 GRPO 变体 |

详见 → `references/algorithms.md`

### 4.4 使用 LoRA

在任意训练命令中添加：

```bash
actor_rollout_ref.model.lora_rank=64 \
actor_rollout_ref.model.lora_alpha=32
```

### 4.5 多节点训练

```bash
# 通过 Ray 提交多节点任务
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    ...
```

## 5. 自定义奖励函数

verl 支持函数式可验证奖励，适用于数学、代码等有明确正确答案的场景：

```python
def compute_score(solution_str, ground_truth, method='strict'):
    """自定义奖励函数示例"""
    # 从模型输出中提取答案
    answer = extract_answer(solution_str)
    # 与标准答案比较
    if answer == ground_truth:
        return 1.0
    return 0.0
```

也可使用基于模型的奖励（Reward Model）：

```bash
reward_model.enable=True \
reward_model.model.path=your-reward-model-path \
reward_model.micro_batch_size=2
```

## 6. 训练监控

### 关键指标

- `critic/vf_loss` — Critic 损失（PPO 专用）
- `actor/entropy` — 策略熵，过低说明探索不足
- `actor/pg_loss` — 策略梯度损失
- `actor/kl_divergence` — 与参考模型的 KL 散度
- `reward/mean` — 平均奖励，应持续上升
- `reward/std` — 奖励标准差

### 日志系统

```bash
# WandB（推荐）
trainer.logger=['console','wandb']

# TensorBoard
trainer.logger=['console','tensorboard']

# MLflow
trainer.logger=['console','mlflow']
```

## 7. 训练前检查清单

- [ ] **数据已转换为 Parquet 格式** — 运行 `scripts/data_prepare_example.py`
- [ ] **数据格式已验证** — 运行 `scripts/dataset_validator.py`
- [ ] **模型路径正确** — HuggingFace 模型 ID 或本地路径
- [ ] **GPU 数量充足** — GRPO 推荐 4+ GPU，PPO 推荐 4+ GPU
- [ ] **Ray 已启动** — 多节点需先启动 Ray cluster
- [ ] **vLLM/SGLang 已安装** — rollout 生成依赖推理引擎
- [ ] **显存规划** — 参考 `references/hardware_distributed.md`

## 8. 算法选择指南

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 数学推理优化 | GRPO | 可验证奖励，无需 Critic |
| 代码生成优化 | GRPO | 测试用例作为奖励信号 |
| 通用 RLHF 对齐 | PPO | 经典方法，有成熟 RM |
| 资源受限 | GRPO + LoRA | 无需 Critic，显存友好 |
| 大规模训练 | PPO/GRPO + Megatron | 支持模型并行 |
| 改进 GRPO 探索 | DAPO | 动态采样，更好探索 |

## 9. 故障排查

| 问题 | 解决方案 |
|------|---------|
| **CUDA OOM** | 减小 `ppo_micro_batch_size`，降低 `gpu_memory_utilization`，启用 LoRA |
| **Ray 连接失败** | 检查 `ray status`，确认 head node 地址正确 |
| **vLLM 报错** | 确认 vLLM 版本 ≥0.8.2，检查 `tensor_model_parallel_size` 设置 |
| **奖励始终为 0** | 检查奖励函数逻辑，验证 ground_truth 字段存在 |
| **KL 散度爆炸** | 降低学习率，增大 `kl_loss_coef` |
| **训练不稳定** | 减小 `clip_ratio`，使用 `loss_agg_mode=token-mean` |

详见 → `references/troubleshooting.md`
