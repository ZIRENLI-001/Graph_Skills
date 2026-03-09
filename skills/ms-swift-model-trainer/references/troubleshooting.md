# 常见问题与解决方案

## OOM (Out of Memory) 排查

### 排查流程

```
CUDA OOM
├── 尝试 --per_device_train_batch_size 1
│   ├── 仍然 OOM → 模型太大
│   │   ├── 启用 LoRA: --train_type lora
│   │   ├── 启用 QLoRA: --quant_bits 4
│   │   ├── 减小 max_length
│   │   └── 使用更大显存的 GPU
│   └── 解决 → 增大 gradient_accumulation_steps 补偿
└── 训练中 OOM（某些 step）
    ├── 数据长度不均匀 → 设置 --max_length 截断
    ├── gradient checkpointing 被关闭 → 确认 --gradient_checkpointing true
    └── 多卡训练 → 尝试 --deepspeed zero2 或 zero3
```

### 常用解决命令

```bash
# 方案 1: 减小 batch_size + 增大梯度累积
swift sft --per_device_train_batch_size 1 --gradient_accumulation_steps 16

# 方案 2: 启用 QLoRA
swift sft --train_type lora --quant_bits 4

# 方案 3: 限制序列长度
swift sft --max_length 1024

# 方案 4: 多卡 + DeepSpeed
NPROC_PER_NODE=4 swift sft --deepspeed zero2
```

## 数据格式错误

### 常见错误

**错误: `KeyError: 'messages'`**
- 原因: 数据集不包含 `messages` 字段
- 解决: 使用 `dataset_validator.py` 检查格式，或使用 `--columns` 映射字段名
- 示例: `--columns '{"conversations": "messages"}'`

**错误: `rejected_response not found` (DPO)**
- 原因: DPO 数据缺少 `rejected_response` 字段
- 解决: 确保每条数据都包含 `rejected_response` 字段
- 使用 `data_prepare_example.py` 中的 `prepare_dpo_data()` 函数转换

**错误: 数据解析失败**
- 原因: JSONL 文件中有格式错误的行
- 排查:
```bash
# 检查 JSONL 文件中的错误行
python -c "
import json
with open('data.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"
```

**错误: 空样本或空内容**
- 原因: 数据中有空的 messages 列表或空 content
- 解决: 数据清洗时过滤空样本

### 预防措施

训练前始终运行:
```bash
python scripts/dataset_validator.py --dataset_path your_data.jsonl --task sft
```

## 多卡训练问题

### NCCL 通信错误

**错误: `NCCL error` / `RuntimeError: NCCL communicator was aborted`**

```bash
# 方案 1: 禁用 P2P（跨 PCIe 通信时常见）
export NCCL_P2P_DISABLE=1

# 方案 2: 使用 SHM（共享内存）
export NCCL_SHM_DISABLE=0

# 方案 3: 指定网络接口
export NCCL_SOCKET_IFNAME=eth0

# 方案 4: 设置超时时间
export NCCL_TIMEOUT=1800
```

### 多卡训练挂起

- 检查 GPU 可见性: `echo $CUDA_VISIBLE_DEVICES`
- 确保 `NPROC_PER_NODE` 与可见 GPU 数量匹配
- 检查是否有僵尸进程占用 GPU: `nvidia-smi`

### DeepSpeed 问题

**错误: `DeepSpeed ZeRO-3 is incompatible with ...`**
- 不是所有模型都支持 ZeRO-3，尝试 ZeRO-2
- 或使用 LoRA + DDP（不使用 DeepSpeed）

## vLLM 配置问题（GRPO）

### 版本兼容性

**错误: `vLLM version not compatible`**
- 确认 vLLM 版本与模型兼容
- 安装推荐版本: `pip install vllm>=0.6.0`

### colocate 模式 OOM

**问题: vLLM colocate 模式显存不足**
- colocate 模式下训练和推理共享 GPU 显存
- 解决方案:
  1. 减小 `num_generations`（如从 8 降到 4）
  2. 切换到 `--vllm_mode server`（独立 GPU 运行推理）
  3. 使用 QLoRA 减小训练显存

### server 模式配置

```bash
# 终端 1: 启动 vLLM server
CUDA_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2.5-7B-Instruct --port 8000

# 终端 2: 启动 GRPO 训练
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_url http://localhost:8000 \
    ...
```

## Loss 相关问题

### Loss 不下降

| 可能原因 | 解决方案 |
|---------|---------|
| 学习率太小 | LoRA 推荐 1e-4，Full 推荐 1e-5 |
| 学习率太大 | Loss 震荡或发散，减小学习率 |
| 数据质量差 | 检查数据，去除噪声样本 |
| batch_size 太小 | 增大 gradient_accumulation_steps |
| 模型已经很好 | Instruct 模型在相关任务上已经表现好 |

### eval_loss 上升（过拟合）

| 可能原因 | 解决方案 |
|---------|---------|
| epoch 过多 | 减少 num_train_epochs |
| 数据集太小 | 增大数据集或使用数据增强 |
| 学习率太高 | 减小学习率 |
| LoRA rank 太大 | 减小 lora_rank |

### Loss 变为 NaN

- 通常是学习率过大或数据异常
- 尝试: 减小 learning_rate，检查数据中是否有异常值
- 使用 bf16 而非 fp16（bf16 动态范围更大）

## 常见错误信息速查

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `CUDA out of memory` | 显存不足 | 见 OOM 排查 |
| `No such file or directory` | 模型/数据路径错误 | 检查路径拼写 |
| `Connection error` | 模型下载失败 | 检查网络，或使用本地模型 |
| `Tokenizer not found` | tokenizer 文件缺失 | 检查模型目录完整性 |
| `AssertionError: adapter` | LoRA 加载失败 | 检查 adapter 路径和兼容性 |
| `torch.cuda.OutOfMemoryError` | GPU 显存溢出 | 同 CUDA OOM |
| `RuntimeError: Expected all tensors` | 数据类型不匹配 | 确认 --torch_dtype 设置 |
| `ValueError: target modules` | LoRA target 不支持 | 检查模型是否支持该 LoRA 配置 |
