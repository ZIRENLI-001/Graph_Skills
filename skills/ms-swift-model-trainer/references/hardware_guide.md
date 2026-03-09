# 硬件选型与显存估算指南

## 显存需求估算

### LoRA 微调显存需求

| 模型规模 | bf16 LoRA | 4-bit QLoRA | 推荐 GPU |
|---------|-----------|-------------|---------|
| 0.5B-1.5B | ~6GB | ~4GB | T4 16GB |
| 3B | ~12GB | ~8GB | V100 16GB |
| 7B-8B | ~18GB | ~12GB | A10G 24GB / RTX 4090 |
| 14B | ~30GB | ~18GB | A100 40GB |
| 32B-34B | ~60GB | ~35GB | A100 80GB |
| 70B-72B | ~120GB | ~60GB | 2×A100 80GB |

### 全参数微调显存需求

| 模型规模 | bf16 Full | 推荐 GPU |
|---------|-----------|---------|
| 0.5B-1.5B | ~12GB | V100 16GB |
| 3B | ~30GB | A100 40GB |
| 7B-8B | ~60GB | A100 80GB |
| 14B | ~120GB | 2×A100 80GB + ZeRO-3 |
| 70B-72B | ~600GB | 8×A100 80GB + ZeRO-3 |

### GRPO 额外显存需求

GRPO 需要同时运行训练和推理，显存需求更高：

| 模型规模 | LoRA + vLLM (colocate) | 推荐配置 |
|---------|------------------------|---------|
| 1.5B | ~16GB | 1×A10G |
| 7B | ~40GB | 2×A100 40GB |
| 7B | ~60GB (无vLLM) | 4×A10G |
| 14B | ~80GB | 4×A100 80GB |
| 70B | ~320GB | 8×A100 80GB |

## GPU 型号对比

| GPU | 显存 | bf16 算力 | 适合场景 |
|-----|------|----------|---------|
| T4 | 16GB | 65 TFLOPS | 小模型 LoRA, QLoRA |
| V100 | 16/32GB | 125 TFLOPS | 中小模型训练 |
| A10G | 24GB | 250 TFLOPS | 7B LoRA 主力 |
| RTX 3090 | 24GB | 285 TFLOPS | 消费级最佳 |
| RTX 4090 | 24GB | 330 TFLOPS | 消费级旗舰 |
| A100 | 40/80GB | 312 TFLOPS | 生产环境主力 |
| H100 | 80GB | 990 TFLOPS | 高性能训练 |
| H200 | 141GB | 990 TFLOPS | 大模型训练 |

## 多卡配置建议

### 2 卡配置
- 7B LoRA SFT / DPO
- 14B QLoRA SFT
- 7B GRPO (vLLM colocate)

### 4 卡配置
- 7B 全参数 SFT (ZeRO-3)
- 14B LoRA SFT / DPO
- 7B GRPO (推荐)
- 14B GRPO (vLLM colocate)

### 8 卡配置
- 14B+ 全参数微调
- 70B LoRA SFT
- 34B+ GRPO

## 省显存技巧优先级

1. **使用 LoRA** — 最显著的效果，降低 60-80% 显存
2. **QLoRA (4-bit)** — 在 LoRA 基础上再减 40%
3. **Gradient Checkpointing** — 减少 30%（默认开启）
4. **减小 batch_size** — 直接减少
5. **减小 max_length** — 减少激活内存
6. **DeepSpeed ZeRO** — 多卡分片
7. **混合精度 bf16** — 比 fp32 减半

## 训练时间估算

以 7B 模型 + LoRA + 单卡 A100 80GB 为例：

| 数据量 | epochs | 预计时间 |
|--------|--------|---------|
| 1,000 | 3 | ~10 min |
| 10,000 | 3 | ~1.5 hr |
| 50,000 | 3 | ~7 hr |
| 100,000 | 3 | ~14 hr |

影响因素：max_length、batch_size、模型大小、GPU 速度。
