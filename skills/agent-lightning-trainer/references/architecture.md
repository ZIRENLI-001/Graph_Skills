# Agent Lightning 架构与核心概念

## 设计哲学

Agent Lightning 的核心思想是**训练-执行解耦（Training-Agent Disaggregation）**：

- Agent 的**运行逻辑**与**学习过程**完全分离
- Agent 通过自身在真实环境中的交互来改进
- 无需修改 Agent 源码，只需薄封装即可接入训练

## 架构总览

系统由两个可执行 Bundle 组成，通过 LightningStore 通信：

```
┌─────────────────────────────────┐
│           Trainer               │
│  (生命周期管理、协调、错误处理)    │
└─────────┬───────────┬───────────┘
          │           │
          ▼           ▼
┌─────────────────┐ ┌─────────────────┐
│  Runner Bundle  │ │ Algorithm Bundle│
│                 │ │                 │
│  ┌───────────┐  │ │  ┌───────────┐  │
│  │ LitAgent  │  │ │  │LitAlgorithm│ │
│  │ (Agent逻辑)│  │ │  │(训练算法)  │  │
│  └───────────┘  │ │  └───────────┘  │
│  ┌───────────┐  │ │  ┌───────────┐  │
│  │  Tracer   │  │ │  │  Adapter  │  │
│  │ (追踪记录) │  │ │  │ (数据转换) │  │
│  └───────────┘  │ │  └───────────┘  │
│  ┌───────────┐  │ │  ┌───────────┐  │
│  │   Hook    │  │ │  │ LLM Proxy │  │
│  │ (生命周期) │  │ │  │ (模型桥接) │  │
│  └───────────┘  │ │  └───────────┘  │
│  ┌───────────┐  │ │                 │
│  │  Runner   │  │ │                 │
│  │ (任务执行) │  │ │                 │
│  └───────────┘  │ │                 │
└────────┬────────┘ └────────┬────────┘
         │                   │
         ▼                   ▼
    ┌─────────────────────────────┐
    │       LightningStore        │
    │  (中心数据库 / 消息队列)      │
    │  任务 | 结果 | 资源 | 追踪    │
    └─────────────────────────────┘
```

## 核心组件详解

### LitAgent — 你的 Agent 封装

`LitAgent` 是连接你的 Agent 与训练框架的核心抽象。

**函数式 API**:
```python
from agentlightning import rollout

@rollout
def my_agent(task, prompt_template, llm):
    # task: dict, 来自数据集
    # prompt_template: str, 由算法管理和优化
    # llm: OpenAI-compatible client, 由 LLM Proxy 提供
    prompt = prompt_template.format(question=task["question"])
    response = llm.chat.completions.create(
        model="main_llm",
        messages=[{"role": "user", "content": prompt}],
    )
    return compute_reward(response, task)
```

`@rollout` 装饰器将函数包装为 `FunctionalLitAgent`，自动注入 `task`、`prompt_template`、`llm` 等参数。

**类式 API**:
```python
import agentlightning as agl

class MyAgent(agl.LitAgent):
    def rollout(self, task, resources):
        """默认执行逻辑"""
        llm = resources["main_llm"]
        return reward

    def training_rollout(self, task, resources):
        """训练时行为（可选），如增加探索"""
        pass

    def validation_rollout(self, task, resources):
        """验证时行为（可选），如确定性推理"""
        pass
```

**奖励返回方式**:
- `float` — 整个 rollout 的最终奖励
- `None` — 需手动调用 `agl.emit_reward(value)` 发射奖励 span

### LightningStore — 中心存储

所有数据的 single source of truth：
- 任务队列（待执行、进行中、已完成）
- Rollout 结果（spans、奖励）
- 资源版本管理（模型权重、提示模板）
- 追踪数据（OpenTelemetry spans）

**后端选择**:

| 后端 | 特点 | 适用场景 |
|------|------|---------|
| 内存（默认） | 线程锁同步，无需依赖 | 单进程调试 |
| MongoDB | 持久化，支持多 worker | 生产环境、Client-Server 模式 |
| 自定义 | 继承 `LightningStore` | Redis、SQL 等 |

```bash
# 启动 MongoDB 后端 Store
agl store --backend mongo --n-workers 4
```

### Trainer — 训练协调器

管理完整训练流程：

```python
trainer = agl.Trainer(
    agent=my_agent,
    algorithm=algorithm,
    train_tasks=train_data,    # list[dict]
    val_tasks=val_data,        # list[dict]
)

trainer.dev()   # dry-run：Baseline 算法，最多 10 任务，打印 span
trainer.fit()   # 正式训练
```

`dev()` vs `fit()`:
- `dev()`: 替换为 Baseline 算法，快速验证数据流和 Agent 封装
- `fit()`: 使用配置的算法，完整训练循环

Trainer 管理 **Runner Fleet**：生成多个 Runner 实例，每个 Runner 领取任务、执行 Agent、流式上传 span。

### Tracer — 链路追踪

基于 OpenTelemetry，自动记录：
- LLM 调用（输入/输出/token 数/延迟）
- 工具调用
- 自定义 span

```python
import agentlightning as agl

# 手动发射 span
agl.emit_reward(0.8)
agl.emit_action("search", {"query": "天气"})
agl.emit_observation("result", {"data": "晴天"})
```

### Adapter — 数据转换

将 Store 中的原始 span 转换为训练可用的格式：

| Adapter | 输入 | 输出 | 特点 |
|---------|------|------|------|
| `TracerTraceToTriplet` | Tracer span | (prompt, response, reward) 三元组 | 重建执行树层级 |
| `LlmProxyTraceToTriplet` | LLM Proxy span | (prompt, response, reward) 三元组 | 扁平结构，更高效 |

### LLM Proxy — 模型桥接

Agent 与 Algorithm 之间的桥梁：

```python
# Agent 中使用逻辑模型名
response = llm.chat.completions.create(
    model="main_llm",  # 逻辑名，Algorithm 动态映射到实际模型
    messages=[...],
)
```

核心能力：
1. **后端抽象**: OpenAI/Anthropic/本地模型统一接口，自带重试/限流/缓存
2. **资源管理**: Algorithm 可热切换模型（如换为新微调权重）
3. **遥测**: 自动为每次调用生成 OpenTelemetry span

### Hook — 生命周期回调

无需修改 Agent 代码注入回调：

```python
class LoggingHook(agl.Hook):
    def on_trace_start(self, ctx):
        print("Trace started")
    def on_rollout_start(self, ctx):
        print("Rollout started")
    def on_rollout_end(self, ctx):
        print(f"Rollout ended, reward: {ctx.reward}")
    def on_trace_end(self, ctx):
        print("Trace ended")
```

### NamedResources — 资源管理

Algorithm 向 Agent 传递命名资源（不仅仅是单个提示模板）：

```python
resources = {
    "main_llm": llm_client,
    "prompt_template": "你是一个...",
    "retriever_config": {...},
}
```

## 执行策略

### SharedMemoryExecutionStrategy

单进程，Runner 在主线程，Algorithm 在 Python 线程：
- 适合调试和单元测试
- 所有数据在内存中共享
- 断点调试友好

### Client-Server

多进程/多机分布式：
- Runner 和 Algorithm 独立部署和扩展
- 通过 LightningStore REST API 通信
- 适合生产环境和大规模训练

```bash
# 启动 Store 服务
agl store --backend mongo --n-workers 4

# 启动 Algorithm
python train.py --role algorithm

# 启动多个 Runner
python train.py --role runner --num-runners 8
```

## LightningRL — 分层 RL

Agent Lightning 的独创 RL 方法：

1. **Credit Assignment（信用分配）**: 将 episode 级奖励分解到每个 LLM 调用
   - Automatic Intermediate Rewarding (AIR): 自动生成中间奖励，提供密集反馈
2. **Token-Level Supervision（token 级监督）**: 在每个 action 内部使用单轮 RL（GRPO/PPO）优化 token 生成

这使得 Agent Lightning 能将**任意** Agent 轨迹转换为可训练的 RL 数据。

## 数据流

```
1. Algorithm 发布任务到 Store
2. Runner 从 Store 领取任务
3. Runner 执行 LitAgent.rollout()
4. Tracer 自动记录 span 到 Store
5. Algorithm 从 Store 读取 span
6. Adapter 转换 span 为训练数据
7. Algorithm 更新模型/提示
8. 更新后的资源发布回 Store
9. 循环至训练结束
```
