# Agent Lightning 配置参数详解

## 配置方式

Agent Lightning 使用 **Python dict** 作为配置（非 YAML），直接传递给算法构造函数。

## Trainer 配置

```python
trainer = agl.Trainer(
    agent=my_agent,           # LitAgent 实例或 @rollout 函数
    algorithm=algorithm,       # LitAlgorithm 实例
    train_tasks=train_data,    # list[dict]，训练任务
    val_tasks=val_data,        # list[dict]，验证任务
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `agent` | `LitAgent` | 封装后的 Agent |
| `algorithm` | `LitAlgorithm` | 训练算法实例 |
| `train_tasks` | `list[dict]` | 训练任务数据 |
| `val_tasks` | `list[dict]` | 验证任务数据 |

### Trainer 方法

| 方法 | 说明 |
|------|------|
| `trainer.dev()` | Dry-run：Baseline 算法，≤10 任务，打印 span |
| `trainer.fit()` | 正式训练 |

## APO 算法配置

```python
from agentlightning.algorithm import APO

algorithm = APO(
    llm_proxy_base_url="https://api.openai.com/v1",
    llm_proxy_api_key="sk-...",
    initial_prompt_template="Given: {request}\nAnswer:",
    n_optimization_rounds=3,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_proxy_base_url` | str | 必填 | LLM API base URL |
| `llm_proxy_api_key` | str | 必填 | API key |
| `initial_prompt_template` | str | 必填 | 初始提示模板，含 `{variable}` 占位符 |
| `n_optimization_rounds` | int | 3 | 优化轮数 |

## VERL 算法配置

配置为嵌套 Python dict，结构如下：

```python
verl_config = {
    "algorithm": {...},
    "data": {...},
    "actor_rollout_ref": {
        "model": {...},
        "rollout": {...},
        "actor": {...},
    },
    "trainer": {...},
}
```

### algorithm 节

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `adv_estimator` | str | `"grpo"` | 算法：`grpo`, `gae`(PPO), `reinforce_plus_plus` |
| `use_kl_in_reward` | bool | `False` | 奖励中是否包含 KL 惩罚 |

### data 节

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_batch_size` | int | 32 | 全局训练 batch size |
| `max_prompt_length` | int | 4096 | prompt 最大 token 数 |
| `max_response_length` | int | 2048 | 回答最大 token 数 |

### actor_rollout_ref.model 节

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | str | 必填 | HuggingFace 模型 ID 或本地路径 |

### actor_rollout_ref.rollout 节

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"vllm"` | 推理引擎：`vllm`, `sglang` |
| `n` | int | 4 | GRPO 每组采样数 |
| `multi_turn.format` | str | `"hermes"` | 多轮对话格式 |
| `gpu_memory_utilization` | float | 0.6 | vLLM GPU 显存占比 |
| `tensor_model_parallel_size` | int | 1 | 张量并行度 |
| `agent_match` | str | `None` | 正则匹配目标 Agent 名称 |

### actor_rollout_ref.actor 节

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ppo_mini_batch_size` | int | 32 | PPO mini batch（全局） |
| `lr` | float | 1e-6 | 学习率 |
| `clip_ratio_low` | float | 0.2 | PPO 裁剪下界 |
| `clip_ratio_high` | float | 0.3 | PPO 裁剪上界 |

### trainer 节

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_gpus_per_node` | int | 1 | 每节点 GPU 数 |
| `val_before_train` | bool | `True` | 训练前先验证 |

## LightningStore 配置

### 内存存储（默认）

无需额外配置，自动使用。

### MongoDB 存储

```bash
# CLI 启动
agl store --backend mongo --n-workers 4

# 需要 MongoDB 已运行
# mongod --dbpath /data/db
```

### 自定义存储

```python
from agentlightning import LightningStore

class MyRedisStore(LightningStore):
    def __init__(self, redis_url):
        self.redis = Redis(redis_url)
    # 实现必需方法...
```

## 完整配置示例

### APO 提示优化

```python
import agentlightning as agl
from agentlightning.algorithm import APO

@agl.rollout
def agent(task, prompt_template, llm):
    prompt = prompt_template.format(question=task["question"])
    resp = llm.chat.completions.create(
        model="main_llm",
        messages=[{"role": "user", "content": prompt}],
    )
    return 1.0 if correct(resp, task) else 0.0

algorithm = APO(
    llm_proxy_base_url="https://api.openai.com/v1",
    llm_proxy_api_key="sk-xxx",
    initial_prompt_template="Answer the question: {question}",
    n_optimization_rounds=3,
)

trainer = agl.Trainer(
    agent=agent,
    algorithm=algorithm,
    train_tasks=[{"question": "...", "answer": "..."}, ...],
    val_tasks=[...],
)
trainer.fit()
```

### VERL GRPO 训练

```python
import agentlightning as agl
from agentlightning.algorithm import VERL

class MyAgent(agl.LitAgent):
    def rollout(self, task, resources):
        llm = resources["main_llm"]
        resp = llm.chat.completions.create(
            model="main_llm",
            messages=[{"role": "user", "content": task["prompt"]}],
        )
        return evaluate(resp, task)

config = {
    "algorithm": {"adv_estimator": "grpo"},
    "data": {
        "train_batch_size": 32,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
    },
    "actor_rollout_ref": {
        "model": {"path": "Qwen/Qwen2.5-1.5B-Instruct"},
        "rollout": {
            "name": "vllm",
            "n": 4,
            "multi_turn": {"format": "hermes"},
            "gpu_memory_utilization": 0.6,
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "lr": 1e-6,
        },
    },
    "trainer": {"n_gpus_per_node": 1},
}

algorithm = VERL(config=config)
trainer = agl.Trainer(
    agent=MyAgent(),
    algorithm=algorithm,
    train_tasks=train_data,
    val_tasks=val_data,
)
trainer.dev()
trainer.fit()
```

### 多 Agent 选择性优化

```python
config = {
    ...
    "actor_rollout_ref": {
        "rollout": {
            "agent_match": "sql_writer",  # 只优化 sql_writer Agent
        },
    },
}
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API key（APO 需要） |
| `CUDA_VISIBLE_DEVICES` | GPU 可见性 |
| `RAY_ADDRESS` | Ray 集群地址（分布式训练） |
