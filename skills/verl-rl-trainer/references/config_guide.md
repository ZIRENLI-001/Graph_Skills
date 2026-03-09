# verl 配置参数详解

verl 使用 **Hydra** 配置系统，所有参数通过命令行 `key=value` 覆盖传递。

## 配置层级结构

```
├── algorithm          # 算法选择与超参
├── data               # 数据路径与格式
├── actor_rollout_ref  # Actor/Rollout/Reference 模型
│   ├── model          # 模型配置
│   ├── actor          # 训练参数
│   ├── rollout        # 推理生成参数
│   └── ref            # 参考模型参数
├── critic             # Critic 模型（PPO 专用）
├── reward_model       # 奖励模型
└── trainer            # 训练器设置
```

## algorithm — 算法配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adv_estimator` | `gae` | 算法选择：`gae`(PPO), `grpo`, `reinforce_plus_plus`, `rloo` |
| `gamma` | `1.0` | 折扣因子 |
| `lam` | `1.0` | GAE lambda 参数 |
| `kl_ctrl.type` | `fixed` | KL 控制器类型：`fixed` 或 `adaptive` |
| `kl_ctrl.kl_coef` | `0.001` | KL 惩罚系数 |

## data — 数据配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_files` | 必填 | 训练数据 Parquet 文件路径 |
| `val_files` | 必填 | 验证数据 Parquet 文件路径 |
| `train_batch_size` | `256` | 全局训练 batch size |
| `max_prompt_length` | `512` | Prompt 最大长度 |
| `max_response_length` | `1024` | 生成回答最大长度 |
| `filter_overlong_prompts` | `True` | 过滤超长 prompt |

**重要**: `train_batch_size` 是全局值，自动分配到各 GPU。

## actor_rollout_ref.model — 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `path` | 必填 | HuggingFace 模型 ID 或本地路径 |
| `lora_rank` | `0` | LoRA 秩，0 表示不使用 |
| `lora_alpha` | `16` | LoRA alpha 值 |
| `enable_gradient_checkpointing` | `True` | 梯度检查点，省显存 |
| `enable_activation_offload` | `False` | 激活值卸载到 CPU |

## actor_rollout_ref.actor — Actor 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `strategy` | `fsdp` | 训练策略：`fsdp`, `fsdp2`, `megatron` |
| `ppo_mini_batch_size` | `64` | 全局 mini-batch size |
| `ppo_micro_batch_size` | `2` | 每 GPU micro-batch size |
| `ppo_epochs` | `1` | 每次 rollout 更新轮数 |
| `clip_ratio` | `0.2` | PPO 裁剪比例 |
| `use_kl_loss` | `False` | 是否在 loss 中加 KL 项 |
| `kl_loss_coef` | `0.001` | KL loss 系数 |
| `loss_agg_mode` | `token-mean` | 损失聚合：`token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean` |
| `lr` | `1e-6` | 学习率 |

**设计原则**：
- `ppo_mini_batch_size` = 全局值（自动分配）
- `ppo_micro_batch_size` = 每 GPU 本地值（影响显存）
- `train_batch_size` 必须能被 `ppo_mini_batch_size` 整除
- `ppo_mini_batch_size` 必须能被 `ppo_micro_batch_size × GPU数` 整除

## actor_rollout_ref.rollout — Rollout 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `name` | `vllm` | 推理引擎：`vllm` 或 `sglang` |
| `n` | `1` | 每个 prompt 生成回答数（GRPO 需 >1） |
| `temperature` | `1.0` | 采样温度 |
| `top_k` | `-1` | Top-K 采样（-1 禁用） |
| `top_p` | `1.0` | Top-P 采样 |
| `tensor_model_parallel_size` | `1` | 推理张量并行度 |
| `gpu_memory_utilization` | `0.4` | vLLM KV cache 显存占比 |

## critic — Critic 配置（PPO 专用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.path` | PPO 必填 | Critic 模型路径 |
| `ppo_epochs` | `1` | Critic 更新轮数 |
| `ppo_mini_batch_size` | 同 actor | Critic mini-batch size |
| `ppo_micro_batch_size` | `2` | Critic 每 GPU micro-batch |
| `cliprange_value` | `0.5` | 值函数裁剪范围 |

## trainer — 训练器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `total_epochs` | `1` | 总训练轮数 |
| `n_gpus_per_node` | `8` | 每节点 GPU 数 |
| `nnodes` | `1` | 节点数 |
| `save_freq` | `-1` | 保存频率（步数）|
| `test_freq` | `-1` | 评估频率（步数）|
| `val_before_train` | `True` | 训练前先评估一次 |
| `logger` | `console` | 日志：`console`, `wandb`, `tensorboard`, `mlflow`, `swanlab` |
| `project_name` | `verl` | WandB/MLflow 项目名 |
| `experiment_name` | `grpo` | 实验名 |

## 常用配置组合示例

### GRPO 数学训练（4×A100）

```bash
algorithm.adv_estimator=grpo \
data.train_batch_size=256 \
data.max_prompt_length=512 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size=2 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
trainer.n_gpus_per_node=4
```

### PPO RLHF（8×A100）

```bash
algorithm.adv_estimator=gae \
data.train_batch_size=512 \
actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size=2 \
actor_rollout_ref.rollout.name=vllm \
critic.model.path=Qwen/Qwen2.5-7B-Instruct \
critic.ppo_micro_batch_size=2 \
trainer.n_gpus_per_node=8
```

### LoRA 小规模训练（2×GPU）

```bash
algorithm.adv_estimator=grpo \
data.train_batch_size=64 \
actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
actor_rollout_ref.model.lora_rank=64 \
actor_rollout_ref.model.lora_alpha=32 \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
actor_rollout_ref.actor.ppo_micro_batch_size=1 \
actor_rollout_ref.rollout.n=4 \
trainer.n_gpus_per_node=2
```
