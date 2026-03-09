# Agent Lightning 故障排查指南

## 安装问题

### pip install agentlightning 失败

**平台限制**: Agent Lightning 仅支持 Linux（Ubuntu 22.04+）。macOS 和 Windows（WSL2 除外）不支持。

```bash
# 确认 Python 版本
python --version  # 需要 3.10+

# 在 WSL2 中安装
wsl --install
# 进入 WSL2 后再安装
pip install agentlightning
```

### flash-attn 编译失败

```bash
# 使用 --no-build-isolation
pip install flash-attn --no-build-isolation

# 或限制并行编译
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### vLLM 版本不兼容

```bash
# Agent Lightning 要求特定版本
pip install vllm==0.10.2
pip install verl==0.5.0

# 注意 torch 版本匹配
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### POML 安装（APO 需要）

```bash
pip install poml
pip install "openai>=2.0"
```

## Agent 封装问题

### rollout() 无返回值

**症状**: 训练正常运行但奖励始终为 0。

**原因**: `rollout()` 方法没有返回 float 奖励。

```python
# 错误：忘记返回奖励
@rollout
def my_agent(task, prompt_template, llm):
    response = llm.chat.completions.create(...)
    # 缺少 return

# 正确
@rollout
def my_agent(task, prompt_template, llm):
    response = llm.chat.completions.create(...)
    return 1.0 if correct else 0.0
```

### 函数式 Agent 参数注入失败

**原因**: `@rollout` 装饰器根据参数名自动注入，参数名必须匹配。

```python
# 可用参数名：
# task — 任务数据 dict
# prompt_template — 提示模板 str
# llm — OpenAI-compatible client

@rollout
def my_agent(task, prompt_template, llm):  # 参数名必须精确匹配
    pass
```

### 类式 Agent resources 为空

**原因**: 使用 `resources["main_llm"]` 而非直接构造 LLM 客户端。

```python
class MyAgent(agl.LitAgent):
    def rollout(self, task, resources):
        # 正确：从 resources 获取
        llm = resources["main_llm"]

        # 错误：直接构造（不会被追踪，无法优化）
        # from openai import OpenAI
        # llm = OpenAI()
```

## LLM Proxy 问题

### LLM Proxy 连接失败

```
ConnectionError: Could not connect to LLM Proxy
```

**解决**:
1. 检查 `llm_proxy_base_url` 是否正确
2. 确认 API key 有效
3. 网络连通性

### 模型名称不匹配

```python
# Agent 中使用逻辑名
response = llm.chat.completions.create(
    model="main_llm",  # 逻辑名，不是实际模型名
    messages=[...],
)
```

### API 限流

LLM Proxy 内置重试和限流，但高并发时仍可能触发 API 限制：
- 减少并发 Runner 数量
- 使用本地模型替代 API

## 训练问题

### trainer.dev() 无输出

**可能原因**:
1. `train_tasks` 为空或格式不对（应为 `list[dict]`）
2. Agent 的 `rollout()` 抛出异常
3. Store 未正确启动

**调试**:
```python
# 检查任务数据
print(f"Tasks: {len(train_tasks)}")
print(f"Sample: {train_tasks[0]}")

# 手动测试 Agent
agent = MyAgent()
result = agent.rollout(train_tasks[0], resources)
print(f"Result: {result}")
```

### CUDA OOM（VERL 训练）

**排查步骤**（按优先级）：

1. **减小 batch size**:
```python
"data": {"train_batch_size": 16},  # 从 32 减到 16
```

2. **减小 mini batch size**:
```python
"actor": {"ppo_mini_batch_size": 16},
```

3. **降低 vLLM 显存**:
```python
"rollout": {"gpu_memory_utilization": 0.4},  # 从 0.6 降到 0.4
```

4. **减少采样数**:
```python
"rollout": {"n": 2},  # 从 4 降到 2
```

5. **减小序列长度**:
```python
"data": {
    "max_prompt_length": 2048,    # 从 4096 降到 2048
    "max_response_length": 1024,  # 从 2048 降到 1024
},
```

### Span 未被记录

**症状**: Algorithm 收到空的 span 列表。

**原因**: Agent 直接调用模型 API 而非通过 LLM Proxy。

```python
# 错误：直接调用，不会被追踪
import openai
client = openai.OpenAI()
client.chat.completions.create(...)

# 正确：通过 resources 获取的 LLM Proxy
llm = resources["main_llm"]
llm.chat.completions.create(model="main_llm", ...)
```

### 多 Agent 优化目标错误

**症状**: 错误的 Agent 被优化或所有 Agent 都被优化。

```python
# 检查 agent_match 正则
"rollout": {
    "agent_match": "write",  # 匹配所有名称含 "write" 的 Agent
    # "write_query" ✓
    # "rewrite_query" ✓
    # "read_query" ✗
}

# 更精确的匹配
"agent_match": "^write_query$"  # 精确匹配
```

## Store 问题

### MongoDB Store 连接失败

```bash
# 确认 MongoDB 运行中
mongosh --eval "db.adminCommand('ping')"

# 启动 Store
agl store --backend mongo --n-workers 4
```

### 内存 Store 数据丢失

内存 Store 是默认模式，进程退出后数据丢失。生产环境请使用 MongoDB 后端。

## Client-Server 模式问题

### Runner 无法连接 Store

```bash
# 检查 Store 服务运行状态
curl http://store-host:port/health

# 确认网络连通
ping store-host
nc -zv store-host port
```

### Runner 与 Algorithm 不同步

**原因**: Store 中的资源版本不匹配。

**解决**: 重启训练，或清理 Store 中的过期资源。

## 常用调试技巧

### 1. 使用 dev() 快速验证

```python
trainer.dev()  # 总是先 dry-run
```

### 2. 添加 Hook 观察行为

```python
class DebugHook(agl.Hook):
    def on_rollout_start(self, ctx):
        print(f"Task: {ctx.task}")
    def on_rollout_end(self, ctx):
        print(f"Reward: {ctx.reward}")
        print(f"Spans: {len(ctx.spans)}")

trainer = agl.Trainer(
    agent=agent,
    algorithm=algorithm,
    hooks=[DebugHook()],
    ...
)
```

### 3. 手动测试 Agent

```python
# 脱离框架单独测试
agent = MyAgent()
task = {"question": "test", "answer": "42"}
resources = {"main_llm": create_test_llm()}
reward = agent.rollout(task, resources)
print(f"Reward: {reward}")
```

### 4. 检查 GPU 状态

```bash
nvidia-smi -l 1
watch -n 1 nvidia-smi
```
