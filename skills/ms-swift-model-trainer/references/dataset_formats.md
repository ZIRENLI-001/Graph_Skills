# 数据集格式详解

ms-swift 支持 4 种标准数据格式，以及 DPO/GRPO 的扩展字段。

## 支持的文件格式

- **JSONL** (.jsonl) — 每行一个 JSON 对象（推荐）
- **JSON** (.json) — JSON 数组
- **CSV** (.csv) — 带表头的 CSV 文件
- **文件夹** — git clone 的开源数据集目录

## 1. Messages 格式（推荐）

最通用的标准格式，支持所有训练任务。

```jsonl
{"messages": [{"role": "system", "content": "你是一个有帮助的助手。"}, {"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}]}
```

### 多轮对话
```jsonl
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}, {"role": "user", "content": "解释一下量子计算"}, {"role": "assistant", "content": "量子计算是利用量子力学原理..."}]}
```

### 角色说明
| 角色 | 说明 |
|------|------|
| `system` | 系统提示（可选，最多一条，放在开头） |
| `user` | 用户输入 |
| `assistant` | 模型回复（SFT 的训练目标） |
| `tool` | 工具调用结果（Agent 场景） |

## 2. ShareGPT / Conversations 格式

常见于开源对话数据集。

```jsonl
{"system": "你是一个助手", "conversation": [{"human": "你好", "assistant": "你好！"}, {"human": "什么是AI？", "assistant": "AI是人工智能..."}]}
```

### 变体（from/value 格式）
```jsonl
{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好！"}]}
```

## 3. Alpaca 格式

适用于指令微调场景。

```jsonl
{"system": "你是一个有帮助的助手", "instruction": "翻译成英文", "input": "你好世界", "output": "Hello World"}
```

- `instruction` 和 `input` 会合并为 query：`query = f'{instruction}\n{input}'`
- `input` 可以为空字符串

## 4. Query/Response 格式

简单问答格式，支持多轮历史。

```jsonl
{"system": "你是一个助手", "query": "什么是深度学习？", "response": "深度学习是机器学习的一个子领域...", "history": [["之前的问题", "之前的回答"]]}
```

### Response 字段自动映射

ms-swift 会自动从以下字段名中识别 response：
`response`, `answer`, `output`, `targets`, `target`, `answer_key`, `answers`, `solution`, `text`, `completion`, `content`

---

## DPO 数据扩展

在标准格式基础上添加 `rejected_response` 字段：

```jsonl
{"messages": [{"role": "user", "content": "写一首诗"}, {"role": "assistant", "content": "春风拂面柳丝长，桃花笑靥映斜阳。"}], "rejected_response": "诗就是一些文字排列。"}
```

### 关键要求
- `messages` 最后一条 assistant 消息的 content 即为 **chosen response**
- `rejected_response` 是与之对应的 **rejected response**
- 每条样本都必须包含 `rejected_response`，否则训练报错

### 从 prompt/chosen/rejected 格式转换

如果原始数据是：
```jsonl
{"prompt": "写一首诗", "chosen": "好的诗...", "rejected": "差的诗..."}
```

使用 `--columns` 映射或用 `data_prepare_example.py` 转换。

---

## GRPO 数据扩展

在标准 messages 基础上添加额外字段，这些字段会自动传递给 ORM 奖励函数。

```jsonl
{"messages": [{"role": "user", "content": "计算 15 × 23 = ?"}], "solution": "345"}
```

### 关键特性
- messages 中通常 **不包含** assistant 回复（模型自行生成）
- 额外字段（如 `solution`）会作为关键字参数传递给 ORM 函数
- 可以添加任意额外字段（如 `test_cases`, `expected_output` 等）

### 多种额外字段
```jsonl
{"messages": [{"role": "user", "content": "写一个加法函数"}], "solution": "def add(a, b): return a + b", "test_cases": "assert add(1,2)==3", "difficulty": "easy"}
```

---

## 列名映射 (--columns)

当数据集字段名不符合标准格式时，使用 `--columns` 参数映射：

```bash
# 将 question 映射为 query，answer 映射为 response
swift sft --dataset data.jsonl --columns '{"question": "query", "answer": "response"}'
```

## 数据集混合

ms-swift 支持同时使用多个数据集，用 `#N` 控制采样数量：

```bash
swift sft --dataset \
    dataset1.jsonl#1000 \    # 取前 1000 条
    dataset2.jsonl#500 \     # 取前 500 条
    AI-ModelScope/alpaca-gpt4-data-en  # 全量使用
```

## 多模态数据

多模态模型支持额外的 `images`、`videos`、`audios` 字段：

```jsonl
{"messages": [{"role": "user", "content": "描述这张图片<image>"}, {"role": "assistant", "content": "这是一张..."}], "images": ["/path/to/image.jpg"]}
```

使用 `<image>`、`<video>`、`<audio>` 标签在 content 中标记媒体位置。
