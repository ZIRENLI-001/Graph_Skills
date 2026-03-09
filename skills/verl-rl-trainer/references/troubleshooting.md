# verl 故障排查指南

## 安装问题

### flash-attn 编译失败

```
ERROR: Failed building wheel for flash-attn
```

**解决**：
```bash
# 使用预编译版本
pip install flash-attn --no-build-isolation
# 或指定 CUDA 版本
MAX_JOBS=4 pip install flash-attn
```

### vLLM 版本不兼容

```
ImportError: cannot import name 'xxx' from 'vllm'
```

**解决**：verl 要求 vLLM ≥ 0.8.2，建议使用官方安装脚本：
```bash
bash scripts/install_vllm_sglang_mcore.sh
```

### Ray 版本冲突

**解决**：确保 `ray>=2.41.0`：
```bash
pip install "ray[default]>=2.41.0"
```

## 训练启动问题

### Ray 集群连接失败

```
ConnectionError: Could not connect to Ray cluster
```

**解决**：
```bash
# 检查 Ray 状态
ray status

# 重启 Ray
ray stop && ray start --head

# 多节点：确认 worker 已连接
ray status  # 应显示所有节点
```

### NCCL 通信超时

```
RuntimeError: NCCL communicator was aborted
```

**解决**：
```bash
# 禁用 P2P（跨节点常见）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 增加超时
export NCCL_TIMEOUT=1800
```

### Hydra 配置错误

```
omegaconf.errors.ConfigAttributeError
```

**解决**：检查参数名拼写，使用 `--help` 查看所有可用参数：
```bash
python -m verl.trainer.main_ppo --help
```

## 显存问题

### CUDA OOM

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**排查步骤**（按优先级）：

1. **减小 micro batch size**：
```bash
actor_rollout_ref.actor.ppo_micro_batch_size=1
critic.ppo_micro_batch_size=1
```

2. **降低 vLLM 显存**：
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
```

3. **启用 LoRA**：
```bash
actor_rollout_ref.model.lora_rank=64
```

4. **减少生成长度**：
```bash
data.max_response_length=512
```

5. **减少每组采样数**：
```bash
actor_rollout_ref.rollout.n=4
```

6. **启用激活卸载**：
```bash
actor_rollout_ref.model.enable_activation_offload=True
```

### vLLM 与训练显存冲突

**症状**：训练阶段 OOM，但 rollout 阶段正常。

**解决**：降低 `gpu_memory_utilization`，给训练留更多显存：
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.2
```

## 训练质量问题

### 奖励始终为 0

**可能原因**：
1. 奖励函数逻辑错误 — 检查 `compute_score` 函数
2. 数据中缺少 `ground_truth`/`solution` 字段
3. 模型输出格式与奖励函数期望不匹配

**调试**：
```python
# 打印模型输出和奖励
print(f"Output: {completion}")
print(f"Ground truth: {ground_truth}")
print(f"Reward: {reward}")
```

### 奖励不增长

**可能原因**：
1. 学习率太低 — 尝试增大 `actor.lr`
2. KL 惩罚过大 — 减小 `kl_loss_coef`
3. `max_response_length` 太短 — 模型无法输出完整答案
4. Batch size 过小 — 增大 `train_batch_size`

### KL 散度爆炸

**症状**：`kl_divergence` 快速增长到很大值。

**解决**：
```bash
# 增大 KL 惩罚
actor_rollout_ref.actor.kl_loss_coef=0.01

# 使用自适应 KL 控制器
algorithm.kl_ctrl.type=adaptive

# 降低学习率
actor_rollout_ref.actor.lr=5e-7
```

### 训练不稳定（loss 震荡）

**解决**：
```bash
# 减小裁剪范围
actor_rollout_ref.actor.clip_ratio=0.1

# 使用 token-mean 损失聚合
actor_rollout_ref.actor.loss_agg_mode=token-mean

# 增大 batch size 以稳定梯度
data.train_batch_size=512
```

### 策略坍缩（entropy 过低）

**症状**：`actor/entropy` 持续下降接近 0，模型输出变得重复。

**解决**：
- 增大 `rollout.temperature`（如 1.0 → 1.2）
- 增加 `rollout.n`（更多采样增加多样性）
- 考虑使用 DAPO 算法（内置抗坍缩机制）

## 数据问题

### Parquet 读取错误

```
pyarrow.lib.ArrowInvalid: ...
```

**解决**：
```bash
# 验证 Parquet 文件
python -c "import pandas as pd; df = pd.read_parquet('train.parquet'); print(df.head())"

# 检查必需字段
python -c "import pandas as pd; df = pd.read_parquet('train.parquet'); print(df.columns.tolist())"
```

### Prompt 超长被过滤过多

**症状**：实际训练样本远少于数据集大小。

**解决**：
```bash
# 增大 max_prompt_length
data.max_prompt_length=1024

# 或关闭过滤（不推荐）
data.filter_overlong_prompts=False
```

## 多节点问题

### Worker 节点超时

**解决**：
```bash
# 增加 Ray 超时
export RAY_BACKEND_LOG_LEVEL=debug

# 检查网络连通性
ping HEAD_NODE_IP

# 确认端口开放
nc -zv HEAD_NODE_IP 6379
```

### 数据路径在 Worker 不存在

**解决**：使用共享存储（NFS/HDFS）或确保每个节点有相同路径下的数据副本。

## 常用调试命令

```bash
# 查看 GPU 使用情况
nvidia-smi -l 1

# 查看 Ray 集群状态
ray status

# 查看 Ray 任务日志
ray job logs <JOB_ID>

# 停止 Ray 任务
ray job stop <JOB_ID>

# 清理 Ray 临时文件
ray stop && rm -rf /tmp/ray
```
