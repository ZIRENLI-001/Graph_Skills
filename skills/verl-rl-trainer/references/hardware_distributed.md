# 硬件需求与分布式训练指南

## 硬件需求估算

### GRPO（无 Critic）

GRPO 需要加载：Actor + Reference Model + Rollout 引擎

| 模型规模 | 最少 GPU | 推荐 GPU | 显存/卡 | 说明 |
|---------|---------|---------|--------|------|
| 0.5B-3B | 2×A100 40G | 4×A100 40G | ~20GB | 小模型快速实验 |
| 3B-7B | 4×A100 40G | 4×A100 80G | ~40GB | 中等规模 |
| 7B-14B | 4×A100 80G | 8×A100 80G | ~60GB | 需要 TP 或 LoRA |
| 14B-72B | 8×A100 80G | 16×A100 80G | ~80GB | 需要多节点或 TP>1 |
| 72B+ | 多节点 | 32+ GPU | ~80GB | Megatron 并行 |

### PPO（有 Critic）

PPO 额外需要 Critic 模型，显存需求约为 GRPO 的 1.5-2 倍。

| 模型规模 | 最少 GPU | 推荐 GPU |
|---------|---------|---------|
| 0.5B-3B | 4×A100 40G | 4×A100 80G |
| 3B-7B | 4×A100 80G | 8×A100 80G |
| 7B+ | 8×A100 80G | 16+ A100 80G |

### 使用 LoRA 降低显存

LoRA 可将训练显存降低 40-60%：

```bash
actor_rollout_ref.model.lora_rank=64 \
actor_rollout_ref.model.lora_alpha=32
```

| 模型 | Full 显存 | LoRA 显存 | 节省 |
|------|----------|----------|------|
| 3B | ~40GB | ~20GB | 50% |
| 7B | ~80GB | ~40GB | 50% |
| 14B | ~160GB | ~80GB | 50% |

## 分布式训练配置

### 单节点

```bash
trainer.n_gpus_per_node=4    # 使用 4 个 GPU
trainer.nnodes=1              # 单节点
```

### 多节点（Ray Cluster）

#### 1. 启动 Ray 集群

```bash
# Head 节点
ray start --head --port=6379

# Worker 节点
ray start --address="HEAD_IP:6379"

# 验证集群
ray status
```

#### 2. 提交训练任务

```bash
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- python3 -m verl.trainer.main_ppo \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    ...
```

### 训练后端选择

| 后端 | 参数值 | 适用场景 |
|------|--------|---------|
| FSDP | `fsdp` | 默认，适合大多数场景 |
| FSDP2 | `fsdp2` | PyTorch 2.x 新版 FSDP |
| Megatron-LM | `megatron` | 超大规模模型（70B+），需要 TP/PP |

```bash
# FSDP（默认）
actor_rollout_ref.actor.strategy=fsdp

# Megatron（大规模）
actor_rollout_ref.actor.strategy=megatron
```

### 推理引擎并行

Rollout 引擎支持张量并行：

```bash
# 单卡推理
actor_rollout_ref.rollout.tensor_model_parallel_size=1

# 2 卡张量并行（大模型）
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

**注意**: `tensor_model_parallel_size` 必须能被 `n_gpus_per_node` 整除。

## 显存优化策略

### 1. 调整 vLLM 显存分配

```bash
# 降低 KV cache 显存占比（默认 0.4）
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
```

### 2. 减小 micro batch size

```bash
actor_rollout_ref.actor.ppo_micro_batch_size=1
critic.ppo_micro_batch_size=1
```

### 3. 梯度检查点（默认开启）

```bash
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

### 4. 激活值卸载

```bash
actor_rollout_ref.model.enable_activation_offload=True
```

### 5. 减少生成长度

```bash
data.max_response_length=512   # 从 1024 降到 512
```

### 6. 减少每组采样数

```bash
actor_rollout_ref.rollout.n=4  # 从 8 降到 4
```

## 性能调优建议

1. **Batch size 与 GPU 数量匹配**：`train_batch_size` 应为 GPU 数的整数倍
2. **vLLM GPU 利用率**：设为 0.3-0.5，过高会与训练争显存
3. **Rollout n 值**：GRPO 推荐 5-16，过大增加计算但基线更准
4. **梯度累积**：通过 `ppo_mini_batch_size / (ppo_micro_batch_size × GPU数)` 隐式控制
5. **混合精度**：默认 bf16，A100/H100 最佳
