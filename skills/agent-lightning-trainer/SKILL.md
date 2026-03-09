---
name: agent-lightning-trainer
description: >
  使用 Microsoft Agent Lightning 框架对任意 AI Agent 进行强化学习训练。
  支持 APO（自动提示优化）、VERL（RL 微调 GRPO/PPO）、Tinker（快速 SFT）等算法。
  完全解耦 Agent 执行与训练，兼容 LangChain、AutoGen、CrewAI 等任意 Agent 框架。
  当用户需要对 AI Agent 进行 RL 训练、提示优化、Agent 对齐时激活此 Skill。
---

# Agent Lightning Trainer

使用 [Agent Lightning](https://github.com/microsoft/agent-lightning) 框架对 AI Agent 进行强化学习训练。由 Microsoft Research 开发，核心理念是**完全解耦 Agent 执行与训练**——无需修改现有 Agent 代码即可接入 RL 训练流程。支持任意 Agent 框架（LangChain、AutoGen、CrewAI、LangGraph、纯 Python），通过薄封装即可训练。

## 1. 框架概述

Agent Lightning 是面向 AI Agent 的生产级 RL 训练框架（EuroSys 2025 HybridFlow 架构），核心特性：

- **框架无关**: 兼容 LangChain、AutoGen、CrewAI、LangGraph 或纯 Python Agent
- **零代码修改**: 仅需用 `@rollout` 装饰器或继承 `LitAgent` 封装现有 Agent
- **完全解耦**: Agent 执行与 RL 训练通过 LightningStore 完全分离
- **多算法支持**: APO（提示优化）、VERL（GRPO/PPO/REINFORCE++ RL 微调）、Tinker（快速 SFT）
- **选择性优化**: 多 Agent 系统中可通过正则匹配选择优化目标，冻结其余 Agent
- **可观测性**: 基于 OpenTelemetry 的全链路追踪（LLM 调用、工具调用、奖励）
- **弹性扩展**: 从单进程调试到多机分布式训练，Runner 与 Algorithm 独立扩展

### 安装

```bash
# 基础安装
pip install agentlightning

# VERL RL 训练依赖（手动安装推荐）
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install vllm==0.10.2
pip install verl==0.5.0

# APO 依赖
pip install poml openai>=2.0
```

**系统要求**: Linux（Ubuntu 22.04+），Python 3.10+。GPU 仅 RL 微调需要，APO 和评估可 CPU 运行。

## 2. 文档参考

- **官方文档**: https://microsoft.github.io/agent-lightning/latest/
- **GitHub**: https://github.com/microsoft/agent-lightning
- **架构与核心概念** → `references/architecture.md`
- **算法详解（APO/VERL/Tinker）** → `references/algorithms.md`
- **配置参数详解** → `references/config_guide.md`
- **故障排查** → `references/troubleshooting.md`

## 3. 核心概念

### 3.1 架构总览

```
┌─────────────────────┐         ┌──────────────────────┐
│   Runner Bundle     │         │  Algorithm Bundle    │
│  ┌───────────────┐  │         │  ┌────────────────┐  │
│  │  LitAgent     │  │         │  │  LitAlgorithm  │  │
│  │  (你的Agent)   │  │  Store  │  │  (APO/VERL/...)│  │
│  ├───────────────┤  │◄───────►│  ├────────────────┤  │
│  │  Tracer       │  │         │  │  Adapter       │  │
│  │  Hook         │  │         │  │  LLM Proxy     │  │
│  │  Runner       │  │         │  │                │  │
│  └───────────────┘  │         │  └────────────────┘  │
└─────────────────────┘         └──────────────────────┘
        ▲                                ▲
        └────────── Trainer 协调 ─────────┘
```

- **LitAgent**: 封装你的 Agent 逻辑，只需实现 `rollout()` 方法
- **LightningStore**: 中心数据库/消息队列，存储任务、结果、资源、追踪数据
- **Algorithm**: 训练大脑——决定执行什么任务、从结果中学习、更新资源
- **Trainer**: 协调 Runner 和 Algorithm 的生命周期

### 3.2 封装 Agent 的两种方式

**方式一：函数式（`@rollout` 装饰器）— 快速上手**

```python
from agentlightning import rollout

@rollout
def my_agent(task, prompt_template, llm):
    """你的 Agent 逻辑"""
    prompt = prompt_template.format(question=task["question"])
    response = llm.chat.completions.create(
        model="main_llm",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    # 计算奖励
    reward = 1.0 if answer.strip() == task["expected"] else 0.0
    return reward  # 返回 float 奖励
```

**方式二：类式（继承 `LitAgent`）— 灵活控制**

```python
import agentlightning as agl

class MyAgent(agl.LitAgent):
    def rollout(self, task, resources):
        llm = resources["main_llm"]
        # Agent 逻辑...
        return reward

    def training_rollout(self, task, resources):
        """训练时的行为（可选，如增加探索）"""
        pass

    def validation_rollout(self, task, resources):
        """验证时的行为（可选，如确定性推理）"""
        pass
```

### 3.3 奖励信号

```python
# 方式一：直接返回 float
def rollout(self, task, resources):
    return 1.0  # 整个 rollout 的奖励

# 方式二：手动发射奖励 span
import agentlightning as agl
def rollout(self, task, resources):
    # ... Agent 逻辑 ...
    agl.emit_reward(0.8)  # 在任意位置发射奖励
```

## 4. 训练执行

### 4.1 APO（自动提示优化）— 无需 GPU

**适用场景**: 优化 Agent 的提示词模板，通过文本梯度迭代改进。无需 GPU，使用 API 模型即可。

```python
import agentlightning as agl
from agentlightning.algorithm import APO

@agl.rollout
def booking_agent(task, prompt_template, llm):
    prompt = prompt_template.format(request=task["request"])
    response = llm.chat.completions.create(
        model="main_llm",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    return 1.0 if task["expected_room"] in answer else 0.0

# 配置 APO
algorithm = APO(
    llm_proxy_base_url="https://api.openai.com/v1",
    llm_proxy_api_key="your-key",
    initial_prompt_template="Given: {request}\nSelect the best room.",
    n_optimization_rounds=3,
)

trainer = agl.Trainer(
    agent=booking_agent,
    algorithm=algorithm,
    train_tasks=train_data,
    val_tasks=val_data,
)

# 先 dry-run 验证
trainer.dev()

# 正式训练
trainer.fit()
```

APO 优化循环：评估 → 计算文本梯度（LLM 批评） → 应用编辑 → 选择最优提示

参考模板 → `scripts/train_apo_example.py`

### 4.2 VERL（RL 微调）— 需要 GPU

**适用场景**: 通过 GRPO/PPO 等 RL 算法微调 LLM 权重，让 Agent 从交互中学习。

```python
import agentlightning as agl
from agentlightning.algorithm import VERL

class SQLAgent(agl.LitAgent):
    def rollout(self, task, resources):
        llm = resources["main_llm"]
        # 构建 SQL 查询 Agent...
        response = llm.chat.completions.create(
            model="main_llm",
            messages=messages,
        )
        sql = response.choices[0].message.content
        # 执行并评估
        reward = evaluate_sql(sql, task["expected_sql"])
        return reward

# VERL 配置（Python dict，非 YAML）
verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",
    },
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
            "tensor_model_parallel_size": 1,
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "lr": 1e-6,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
    },
}

algorithm = VERL(config=verl_config)
trainer = agl.Trainer(
    agent=SQLAgent(),
    algorithm=algorithm,
    train_tasks=train_data,
    val_tasks=val_data,
)

trainer.dev()   # dry-run
trainer.fit()   # 正式训练
```

参考模板 → `scripts/train_verl_example.py`

### 4.3 选择性多 Agent 优化

在多 Agent 系统中，只训练特定 Agent：

```python
verl_config = {
    ...
    "actor_rollout_ref": {
        "rollout": {
            "agent_match": "write",  # 正则匹配，只优化名称含 "write" 的 Agent
        },
    },
}
```

### 4.4 Dry-Run 调试

```python
trainer.dev()
```

`dev()` 方法：
- 替换为轻量 Baseline 算法
- 只入队最多 10 个任务
- 打印每个 span 的详细信息
- 适合在提交 GPU 训练前验证连接和数据流

## 5. LLM Proxy — 统一模型接口

LLM Proxy 是 Agent 与训练之间的桥梁，提供统一的 OpenAI 兼容 API：

```python
# Agent 中使用 LLM Proxy
response = llm.chat.completions.create(
    model="main_llm",  # 逻辑名，由 Algorithm 动态映射到实际模型
    messages=[...],
)
```

**核心能力**:
- **后端抽象**: 统一 OpenAI、Anthropic、本地模型接口，自带重试/限流/缓存
- **资源管理**: Algorithm 可动态切换模型（如换为新微调权重），Agent 代码无需修改
- **遥测**: 自动为每次 LLM 调用生成 OpenTelemetry span

## 6. 追踪与可观测性

Agent Lightning 基于 OpenTelemetry 构建全链路追踪：

```python
# 自动追踪：LLM 调用、工具调用会被自动记录

# 手动发射 span
import agentlightning as agl

agl.emit_reward(0.8)                          # 奖励
agl.emit_action("search", {"query": "..."})   # 动作
agl.emit_observation("result", {"data": "..."})  # 观测
```

### Hook 系统

无需修改 Agent 代码即可注入生命周期回调：

```python
class MyHook(agl.Hook):
    def on_rollout_start(self, context):
        print(f"Rollout started: {context}")

    def on_rollout_end(self, context):
        print(f"Rollout ended: {context}")
```

## 7. 执行策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `SharedMemoryExecutionStrategy` | 单进程，Runner 在主线程，Algorithm 在 Python 线程 | 调试、单元测试 |
| Client-Server | 多进程/多机分布式，角色独立部署 | 生产环境、大规模训练 |

### Client-Server 模式

```bash
# 启动持久化 Store（MongoDB 后端）
agl store --backend mongo --n-workers 4

# 分别启动 Algorithm 和 Runner
python train.py --role algorithm
python train.py --role runner --num-runners 8
```

## 8. 算法选择指南

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 提示词优化 | APO | 无需 GPU，通过文本梯度迭代改进提示 |
| Agent RL 微调（有可验证奖励） | VERL (GRPO) | 端到端 RL，无需 Critic |
| Agent RL 微调（有 RM） | VERL (PPO) | 经典 RLHF |
| 快速 SFT | Tinker | 秒级训练，比 Azure OpenAI fine-tune 快数百倍 |
| Agent 调试/验证 | Baseline (dev) | `trainer.dev()` 快速验证数据流 |
| 多 Agent 选择性训练 | VERL + agent_match | 正则匹配目标 Agent |

## 9. 训练前检查清单

- [ ] **Agent 已封装** — 使用 `@rollout` 或继承 `LitAgent`，`rollout()` 返回 float 奖励
- [ ] **数据已准备** — 训练/验证任务列表（dict 列表）
- [ ] **Dry-run 通过** — `trainer.dev()` 无错误，span 正常打印
- [ ] **APO**: OpenAI API key 已配置
- [ ] **VERL**: GPU 可用，vLLM/flash-attn 已安装，模型路径正确
- [ ] **多 Agent**: `agent_match` 正则匹配正确的目标 Agent
- [ ] **系统**: Linux 环境，Python 3.10+

## 10. 故障排查

| 问题 | 解决方案 |
|------|---------|
| **安装失败** | 确认 Linux + Python 3.10+，flash-attn 需 `--no-build-isolation` |
| **CUDA OOM** | 减小 `train_batch_size`、`ppo_mini_batch_size`，降低 `gpu_memory_utilization` |
| **rollout 返回无奖励** | 确保 `rollout()` 返回 float 或调用 `agl.emit_reward()` |
| **LLM Proxy 连接失败** | 检查 `llm_proxy_base_url` 和 API key 配置 |
| **Span 未记录** | 确认使用 `resources["main_llm"]` 而非直接调用模型 API |
| **多 Agent 优化目标错误** | 检查 `agent_match` 正则表达式 |
| **trainer.dev() 无输出** | 检查任务数据格式，确保 task 是 dict |

详见 → `references/troubleshooting.md`
