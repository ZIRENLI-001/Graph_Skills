# 模型导出、合并与量化指南

## LoRA 权重合并

训练后得到的 LoRA adapter 需要合并到基础模型中才能独立使用。

### 合并命令

```bash
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true \
    --output_dir output/merged_model
```

合并后的 `output/merged_model` 目录包含完整模型权重，可以像普通模型一样加载。

### 使用合并后的模型

```bash
# 推理
swift infer --model output/merged_model --stream true

# 部署
swift deploy --model output/merged_model --port 8000
```

## 量化导出

量化可以显著减小模型大小和推理显存需求。

### GPTQ 量化

```bash
# 4-bit GPTQ（推荐）
swift export \
    --model output/merged_model \
    --quant_method gptq \
    --quant_bits 4 \
    --output_dir output/model-gptq-int4

# 8-bit GPTQ
swift export \
    --model output/merged_model \
    --quant_method gptq \
    --quant_bits 8 \
    --output_dir output/model-gptq-int8
```

### AWQ 量化

```bash
swift export \
    --model output/merged_model \
    --quant_method awq \
    --quant_bits 4 \
    --output_dir output/model-awq-int4
```

### 量化方法对比

| 方法 | 精度损失 | 推理速度 | 兼容性 |
|------|---------|---------|--------|
| GPTQ 4-bit | 低 | 快 | vLLM, transformers |
| GPTQ 8-bit | 极低 | 中 | vLLM, transformers |
| AWQ 4-bit | 低 | 最快 | vLLM, transformers |
| GGUF Q4_K_M | 中低 | 快 | llama.cpp, ollama |

## GGUF 转换

GGUF 格式用于 llama.cpp、Ollama 等本地推理框架。

### 步骤

1. 先合并 LoRA（如果使用了 LoRA）
2. 使用 llama.cpp 的转换工具

```bash
# 1. 合并 LoRA
swift export --adapters output/vx-xxx/checkpoint-xxx --merge_lora true --output_dir output/merged

# 2. 转换为 GGUF（需要安装 llama.cpp）
python llama.cpp/convert_hf_to_gguf.py output/merged --outtype f16 --outfile model.gguf

# 3. 量化 GGUF
llama.cpp/build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
```

### 常用 GGUF 量化级别

| 级别 | 大小 (7B) | 质量 | 适用场景 |
|------|----------|------|---------|
| Q2_K | ~2.8GB | 较低 | 极端显存限制 |
| Q4_K_M | ~4.1GB | 良好 | 推荐日常使用 |
| Q5_K_M | ~4.8GB | 很好 | 质量与大小平衡 |
| Q6_K | ~5.5GB | 优秀 | 高质量需求 |
| Q8_0 | ~7.2GB | 接近原始 | 最高精度 |
| F16 | ~14GB | 原始 | 无损 |

## Hub 上传

### ModelScope Hub

```bash
# 训练时自动上传
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --push_to_hub true \
    --hub_model_id your-username/model-name \
    --output_dir output

# 手动上传合并后的模型
# 需要先 modelscope login
modelscope upload your-username/model-name output/merged_model
```

### HuggingFace Hub

```bash
# 训练时自动上传到 HuggingFace
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --push_to_hub true \
    --hub_model_id your-username/model-name \
    --use_hf true \
    --output_dir output

# 手动上传（需要 huggingface-cli login）
huggingface-cli upload your-username/model-name output/merged_model
```

### 上传前检查清单

- [ ] 模型文件完整（config.json, model weights, tokenizer files）
- [ ] 如果是 LoRA，确认已合并（或上传 adapter 供他人使用）
- [ ] 添加 model card（README.md），说明训练方法和数据
- [ ] 设置合适的 license
- [ ] 不要上传含敏感信息的训练日志
