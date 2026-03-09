# Agent Lightning 算法详解

## 算法总览

| 算法 | 类型 | 需要 GPU | 优化目标 | 适用场景 |
|------|------|---------|---------|---------|
| APO | 提示优化 | 否 | 提示模板文本 | 提示词迭代改进 |
| VERL (GRPO) | RL 微调 | 是 | LLM 权重 | 有可验证奖励的 Agent |
| VERL (PPO) | RL 微调 | 是 | LLM 权重 | 有 Reward Model 的 Agent |
| VERL (REINFORCE++) | RL 微调 | 是 | LLM 权重 | 改进的 REINFORCE |
| Tinker | 快速 SFT | 是/云 | LLM 权重 | 快速监督微调 |
| Baseline | 调试 | 否 | 无 | `trainer.dev()` 验证 |

## APO — Automatic Prompt Optimization

### 核心思想

APO 通过「文本梯度」迭代优化提示词模板，无需 GPU 或模型微调。类似梯度下降，但在文本空间中操作。

### 优化循环

```
Round N:
  1. Evaluate — 用当前提示模板运行 Agent，收集奖励
  2. Compute Textual Gradients — LLM 分析失败案例，生成改进建议
  3. Apply Edits — LLM 根据建议生成新提示模板候选
  4. Select Best — 评估候选，保留最优
```

### 配置

```python
from agentlightning.algorithm import APO

algorithm = APO(
    llm_proxy_base_url="https://api.openai.com/v1",
    llm_proxy_api_key="your-api-key",
    initial_prompt_template="Given: {request}\nSelect the best option.",
    n_optimization_rounds=3,       # 优化轮数
)
```

### 关键特性

- **无需 GPU**: 使用 API 模型（如 GPT-4）作为优化器
- **模板变量**: 支持任意数量的 `{variable}` 占位符
- **限制**: 目前只优化单个提示模板
- **效果参考**: 会议室预订 Agent 准确率 56.9% → 72.1%（2 轮）

### 适用场景

- Agent 逻辑固定，只需改进提示词
- 快速原型验证，不想投入 GPU 资源
- 与任何 LLM API 兼容（OpenAI、Anthropic、Azure 等）

## VERL — Reinforcement Learning 微调

### 核心思想

基于 verl 引擎，对 Agent 使用的 LLM 进行端到端 RL 微调。Agent Lightning 的独特贡献是 **LightningRL**——将多轮 Agent 轨迹分解为可训练的 RL 数据。

### LightningRL 方法

```
Agent 轨迹:
  [LLM调用1] → [工具调用] → [LLM调用2] → [工具调用] → [最终奖励]
                            ↓
Credit Assignment (信用分配):
  LLM调用1 → 中间奖励1    LLM调用2 → 中间奖励2
                            ↓
Token-Level RL (GRPO/PPO):
  对每个 LLM 调用的 token 进行策略优化
```

### 支持的 RL 算法

| 算法 | `adv_estimator` | 需要 Critic | 说明 |
|------|----------------|-------------|------|
| GRPO | `grpo` | 否 | 推荐，组内相对排序 |
| PPO | `gae` | 是 | 经典 RLHF |
| REINFORCE++ | `reinforce_plus_plus` | 否 | 改进基线 |

### 配置示例

```python
from agentlightning.algorithm import VERL

verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",       # 算法选择
        "use_kl_in_reward": False,     # 是否在奖励中加 KL
    },
    "data": {
        "train_batch_size": 32,        # 全局 batch size
        "max_prompt_length": 4096,     # prompt 最大长度
        "max_response_length": 2048,   # 回答最大长度
    },
    "actor_rollout_ref": {
        "model": {
            "path": "Qwen/Qwen2.5-1.5B-Instruct",  # 模型路径
        },
        "rollout": {
            "name": "vllm",            # 推理引擎
            "n": 4,                    # GRPO 每组采样数
            "multi_turn": {
                "format": "hermes",    # 多轮对话格式
            },
            "gpu_memory_utilization": 0.6,
            "tensor_model_parallel_size": 1,
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "lr": 1e-6,               # 学习率
            "clip_ratio_low": 0.2,     # PPO 裁剪下界
            "clip_ratio_high": 0.3,    # PPO 裁剪上界
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
    },
}

algorithm = VERL(config=verl_config)
```

### 多轮对话格式

VERL 支持多轮 Agent 对话，通过 `multi_turn.format` 指定格式：
- `"hermes"`: Hermes 格式（推荐）
- 与 vLLM/SGLang 的 chat template 对应

### 选择性多 Agent 优化

```python
"rollout": {
    "agent_match": "write",  # 正则匹配：只训练名称含 "write" 的 Agent
}
```

这允许在多 Agent 系统中冻结部分 Agent，只训练目标 Agent。

### 效果参考

- SQL Agent (Spider): Llama 3.2 3B 训练后显著提升
- 训练时间: 单 80GB GPU ~12 小时

## Tinker — 快速 SFT

### 核心思想

极速监督微调，比 Azure OpenAI fine-tuning 快数百倍：
- 训练时间: 秒级 vs 小时级
- 部署时间: ~15 秒 vs 数分钟

### 适用场景

- 有高质量示范数据，需快速微调
- 迭代速度优先
- Azure OpenAI 环境

## Baseline — 调试算法

### 用途

专用于 `trainer.dev()` dry-run：
- 不执行实际训练
- 打印所有 span 信息
- 验证数据流和 Agent 封装
- 最多处理 10 个任务

```python
trainer.dev()  # 自动使用 Baseline 算法
```

## 算法选择决策树

```
需要训练 Agent
├─ 只想优化提示词？
│   └─ 是 → APO（无需 GPU）
├─ 需要微调 LLM 权重？
│   ├─ 有可验证奖励（数学/代码/SQL）？
│   │   └─ 是 → VERL (GRPO)（推荐）
│   ├─ 有 Reward Model？
│   │   └─ 是 → VERL (PPO)
│   └─ 有高质量示范数据？
│       └─ 是 → Tinker (SFT)
├─ 多 Agent 系统？
│   └─ VERL + agent_match 选择性优化
└─ 先调试验证？
    └─ trainer.dev()（Baseline）
```

## 奖励设计

### 终端奖励

最简单的方式，`rollout()` 返回 float：

```python
def rollout(self, task, resources):
    # Agent 逻辑...
    if correct:
        return 1.0
    else:
        return 0.0
```

### 中间奖励

使用 `emit_reward()` 在执行过程中发射多个奖励：

```python
def rollout(self, task, resources):
    # 第一步完成
    agl.emit_reward(0.3)

    # 第二步完成
    agl.emit_reward(0.5)

    # 最终完成
    agl.emit_reward(1.0)
```

### 密集奖励 vs 稀疏奖励

- **稀疏奖励**: 只有最终结果的对/错，简单但训练信号弱
- **密集奖励**: 每步都有奖励，训练更稳定但设计更复杂
- **AIR (Automatic Intermediate Rewarding)**: LightningRL 自动从稀疏奖励生成中间奖励
