# 常见训练模式与参数配置

## LoRA 配置模板

### 标准 LoRA
```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --learning_rate 1e-4 \
    --output_dir output
```

### 高 rank LoRA（更强表达能力）
```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --learning_rate 5e-5 \
    --output_dir output
```

### QLoRA（4-bit 量化 + LoRA，节省显存）
```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --quant_bits 4 \
    --learning_rate 1e-4 \
    --output_dir output
```

## 全参数微调配置

```bash
# 单卡（小模型 < 3B）
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --train_type full \
    --learning_rate 1e-5 \
    --output_dir output

# 多卡 + DeepSpeed（大模型）
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --deepspeed zero3 \
    --learning_rate 1e-5 \
    --output_dir output
```

## 多卡训练配置

### DDP（数据并行）
```bash
# 自动使用 DDP
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --output_dir output
```

### DeepSpeed ZeRO

| 级别 | 分片内容 | 适用场景 |
|------|---------|---------|
| ZeRO-2 | 优化器状态 + 梯度 | LoRA 多卡 |
| ZeRO-3 | 优化器 + 梯度 + 参数 | 全参数微调 |

```bash
# ZeRO-2（推荐 LoRA 多卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --deepspeed zero2 \
    --output_dir output

# ZeRO-3（全参数微调必需）
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --deepspeed zero3 \
    --output_dir output
```

## Batch Size 与梯度累积

有效 batch_size = `per_device_train_batch_size` × `gradient_accumulation_steps` × `GPU数量`

### 推荐配置

| GPU 数 | batch_size | grad_accum | 有效 batch |
|--------|-----------|------------|-----------|
| 1 | 1 | 16 | 16 |
| 1 | 2 | 8 | 16 |
| 2 | 1 | 8 | 16 |
| 4 | 1 | 8 | 32 |
| 4 | 2 | 4 | 32 |
| 8 | 2 | 4 | 64 |

建议有效 batch_size 在 16-128 之间。

## 学习率调度

ms-swift 默认使用 cosine 学习率调度。

```bash
swift sft \
    --lr_scheduler_type cosine \    # 'cosine' | 'linear' | 'constant'
    --warmup_ratio 0.05 \           # warmup 占比
    --learning_rate 1e-4 \
    --output_dir output
```

### 推荐学习率

| 训练方式 | 学习率范围 |
|---------|-----------|
| LoRA SFT | 1e-4 ~ 5e-5 |
| Full SFT | 1e-5 ~ 5e-6 |
| DPO | 5e-6 ~ 5e-7 |
| GRPO | 5e-6 ~ 1e-6 |

## 常用模型推荐参数

### Qwen2.5-7B-Instruct (LoRA SFT)
```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --lora_rank 8 \
    --torch_dtype bfloat16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output
```

### DeepSeek-R1-Distill-Qwen-7B (LoRA SFT)
```bash
swift sft \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --train_type lora \
    --torch_dtype bfloat16 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --output_dir output
```

### Llama-3.1-8B-Instruct (LoRA SFT)
```bash
swift sft \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --use_hf true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --output_dir output
```

## 序列并行（长序列训练）

处理超长序列时启用序列并行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --sequence_parallel_size 4 \
    --max_length 32768 \
    --output_dir output
```

## Gradient Checkpointing

默认开启。在显存紧张时，这是最有效的省显存方法之一：

```bash
swift sft \
    --gradient_checkpointing true \   # 默认 true
    --output_dir output
```

关闭可以加速训练（但消耗更多显存）：
```bash
swift sft \
    --gradient_checkpointing false \
    --output_dir output
```
