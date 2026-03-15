# LLM / Agent 赛道专项指南

## 1. LLM 竞赛全景

### 赛题列表

| 赛题 | 年份 | 类型 | 奖金 | 核心挑战 |
|------|------|------|------|---------|
| AIMO 1 | 2024 | 数学推理 | $131k | 数学问题求解, P100/T4, 9h |
| AIMO 2 | 2025 | 数学推理 | $262k | 更难, 4×L4, 5h |
| AIMO 3 | 2025-26 | 数学推理 | $220万 | 国家→IMO难度, H100 |
| LMSYS Arena | 2024 | 偏好预测 | $100k | 预测用户对LLM回复的偏好 |
| LLM Science Exam | 2024 | 科学QA | - | MCQ, RAG+LLM/DeBERTa |
| LLM 20 Questions | 2024 | 推理游戏 | - | 20问猜答案 |
| Konwinski Prize | 2025 | 代码Agent | $100万 | 修复GitHub Issue, 纯开源 |
| ARC Prize 2024 | 2024 | 视觉推理 | $60万+ | 抽象推理, TTT |
| ARC Prize 2025 | 2025 | 视觉推理 | $60万+ | ARC-AGI-2更难 |
| AgentSociety | 2025 | Agent建模 | - | LLM用户行为建模+推荐 |

### 共同约束

- **代码竞赛**: 必须提交完整Notebook
- **GPU限制**: P100/T4/L4/H100, 固定时间
- **无网络**: 大部分禁止联网
- **开源要求**: 代码/模型/数据必须公开(AIMO)
- **闭源API禁止**: Konwinski禁止使用闭源模型

### 模型选择趋势

```
2024: DeepSeek-Math-7B 统治 AIMO1
      DeBERTa-v3 在 LLM Science Exam 媲美70B
      Gemma2-9b 作为蒸馏目标(LMSYS)

2025: Qwen2.5-14B 成为 AIMO2 基座
      DeepSeek-R1-Distill 系列广泛使用
      Qwen2.5-Coder-32B 统治 Konwinski
      GPT-OSS-120B 出现在 AIMO3(OpenAI首个开源模型)
      Qwen3-235B-A22B(MoE) 提供大模型选项

趋势: Qwen + DeepSeek 统治Kaggle LLM赛道
```

---

## 2. 数学推理赛专项 (AIMO 系列)

### 2.1 训练 Pipeline

```
                    ┌──────────────────┐
                    │  基座模型选择     │
                    │ DeepSeek-Math-7B │
                    │ Qwen2.5-14B     │
                    │ DeepSeek-R1-Dist │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Stage 1: CoT SFT │
                    │ 数学文本解题      │
                    │ NuminaMath-CoT    │
                    │ OpenMathReasoning │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Stage 2: TIR SFT │
                    │ 代码解题          │
                    │ Python代码+执行   │
                    │ 迭代过滤          │
                    └────────┬─────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
      ┌────────▼──────┐ ┌───▼───────┐ ┌───▼────────┐
      │  DPO          │ │  GRPO     │ │  GenSelect │
      │ 缩短输出长度   │ │  RL优化   │ │  选择器训练│
      └───────────────┘ └───────────┘ └────────────┘
```

### 2.2 数据集构建

**AIMO1 (Numina)**:
```
NuminaMath-CoT: ~1M数学问题 + 文本解答
NuminaMath-TIR: ~60k问题 + TORA格式(代码+执行+结果)
  └─ GPT-4生成推理路径
  └─ 执行Python代码
  └─ 过滤错误答案
  └─ 重复3轮
```

**AIMO2 (NemoSkills)**:
```
OpenMathReasoning(NVIDIA开源):
  ├── 306k唯一问题(AoPS论坛)
  ├── 3.2M长推理CoT解(DeepSeek-R1 + QwQ-32B生成)
  │   └─ temp=0.7, top-p=0.95, max 16384 tokens, 每题32候选
  ├── 1.7M TIR解(迭代pipeline: 训练→生成→过滤→重训练)
  └── 566k GenSelect训练样本
  └─ 总计5.5M训练样本
```

### 2.3 推理策略详解

#### SC-TIR (Self-Consistency + Tool-Integrated Reasoning)

```python
def sc_tir_inference(model, problem, M=64, N=3):
    """
    M: majority voting宽度(并行采样数)
    N: 推理深度(代码执行+继续轮数)
    """
    answers = []
    for i in range(M):
        prompt = problem
        for step in range(N):
            # 采样completion直到产生Python代码块
            output = model.generate(prompt, temperature=0.7)

            if contains_python_code(output):
                # 执行Python代码
                code = extract_code(output)
                exec_result = execute_python(code)
                prompt = output + f"\n```output\n{exec_result}\n```\n"
            else:
                break

        answer = extract_numerical_answer(output)
        if answer is not None:
            answers.append(answer)

    # 多数投票
    from collections import Counter
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return None
```

#### GenSelect (AIMO2 1st, 优于majority voting)

```python
def genselect_inference(solver_model, selector_model, problem, N=64):
    """
    用solver生成N个候选, 用selector选最优
    """
    # Step 1: 生成N个候选解
    candidates = []
    for i in range(N):
        solution = solver_model.generate(problem, temperature=0.7)
        candidates.append(solution)

    # Step 2: 生成候选摘要
    summaries = []
    for solution in candidates:
        summary = summarizer_model.generate(
            f"Summarize the key steps and answer:\n{solution}",
            max_tokens=2048
        )
        summaries.append(summary)

    # Step 3: selector选择最优
    prompt = f"Given these {len(summaries)} solutions, select the best one:\n"
    for i, s in enumerate(summaries):
        prompt += f"\n--- Solution {i+1} ---\n{s}"
    best_idx = selector_model.generate(prompt)

    return extract_answer(candidates[int(best_idx)])
```

### 2.4 推理加速

| 技术 | 加速比 | 实现 | 使用者 |
|------|--------|------|--------|
| FP8量化 | 1.5x vs FP16 | TensorRT-LLM | AIMO2 1st |
| ReDrafter | 1.8x | 3-token proposal, 65%接受 | AIMO2 1st |
| W4KV8 | 1.55x vs FP16 | lmdeploy/TurboMind | AIMO2 2nd |
| AWQ 4bit | ~2x | AutoAWQ | AIMO top通用 |
| Early stopping | 可变 | 答案收敛4+次停 | AIMO2 1st/2nd |
| 动态调速 | 可变 | adjust_speed模块 | AIMO2 2nd |

### 2.5 AIMO3 当前策略

```
硬件升级: H100 → 可跑120B+模型
模型选择:
  ├── GPT-OSS-120B: OpenAI开源, Apache2.0, ~3h/50题, 43/50(weighted entropy)
  ├── Qwen3-235B-A22B: MoE架构, 可控思考预算
  ├── Qwen3.5-27B: 适合fine-tune + vLLM + TIR
  ├── DeepSeek-R1-0528: AIME2025 87.5%(vs旧版70%)
  └── DeepSeekMath-V2: IMO2025金牌, 基于V3.2-Exp-Base

训练资源: 可申请128×H100(Fields Model Initiative)
```

---

## 3. AGI 推理赛专项 (ARC Prize)

### 3.1 Test-Time Training 详解

```
ARC任务结构:
  ├── 2-5个示例: input grid → output grid
  └── 1个测试: input grid → ? (需预测)

TTT流程(所有top方案共用):
  Step 1: 数据增强
    ├── 几何变换: 旋转(0/90/180/270°) × 翻转(水平/垂直)
    ├── 颜色置换: 随机重映射颜色编号
    └── 从2-5个示例扩展到数百个训练样本

  Step 2: Per-Task微调
    ├── 用LoRA微调(比全量更稳定)
    ├── 每个任务独立微调(不混合)
    ├── 训练时间: 几分钟/任务
    └── 约束: ~$0.20/任务

  Step 3: 预测
    ├── 多augmentation下预测
    ├── 多checkpoint预测
    └── Ensemble → 最终输出

  Step 4: 验证
    └── 用已知示例验证预测一致性
```

### 3.2 各方案技术对比

| 维度 | NVARC (1st, 24%) | ARChitects (2nd, 16.5%) | MindsAI (3rd, 12.6%) |
|------|-----------------|----------------------|---------------------|
| **模型** | Qwen-2-VL-4B | LLaDA-8B (masked diffusion) | CodeT5-660M |
| **Tokenizer** | 极简16 token(0-9+格式) | 标准 | 标准+dropout |
| **位置编码** | 标准 | 2D-RoPE(适配2D网格) | 标准 |
| **TTT方式** | 全量微调 | Per-task LoRA | TTFT+AIRV |
| **数据** | 百万级合成 | 合成+原始 | ARC Mega预训练 |
| **核心创新** | 合成数据pipeline | 2D结构感知 | Tokenizer dropout |

### 3.3 合成数据 Pipeline (NVARC)

```
概念分解
  └─ 将ARC任务分解为基础概念(平移、旋转、填充、计数等)

基础任务生成
  └─ 为每个概念生成简单任务

组合复杂任务
  └─ 将多个概念组合生成更复杂的任务

模型验证
  └─ 用Qwen等模型验证生成的任务是否有唯一解

渐进难度
  └─ 从简单→复杂，逐步提升

规模: 百万级语料
```

### 3.4 关键数据

```
Pure LLM prompting: < 5% (几乎无用)
LLM fine-tuning only: ~10%
TTT (test-time training): > 20% (ARC-AGI-2)
转导+归纳 ensemble: 53.5% (ARC-AGI-1, 2024)
最好的商业模型: 37.6% ($2.20/task, Opus 4.5 Thinking)
最好的refinement: 54% ($30/task, Poetiq on Gemini 3 Pro)
```

---

## 4. Agent 代码赛专项 (Konwinski Prize)

### 4.1 任务描述

```
输入: GitHub Issue描述 + 仓库代码
输出: 修复Issue的代码patch
约束: 禁止闭源API, 纯开源模型
```

### 4.2 Agent 架构

```
┌─────────────────────────────────────────┐
│              上下文管理(核心)             │
│  ├── Issue理解: 提取关键信息            │
│  ├── 代码定位: 找到相关文件和函数       │
│  ├── 窗口管理: 在有限token内提供最相关代码│
│  └── 历史追踪: 记录已尝试的修复        │
└────────────────────┬────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐     ┌──────▼──────┐   ┌────▼────┐
│ 理解   │     │   定位      │   │  生成   │
│ Issue  │ →   │   问题代码   │ → │  Patch  │
└────────┘     └─────────────┘   └────┬────┘
                                      │
                                 ┌────▼────┐
                                 │  验证   │
                                 │  Patch  │
                                 └─────────┘
```

### 4.3 关键发现

- **上下文管理是最重要组件** — Round1冠军的核心优势
- **Top5全部使用Qwen或DeepSeek** — 开源代码模型的统治地位
- **7.5% vs SWE-Bench 75%** — 新issue远比benchmark难, 原因:
  - SWE-Bench可能有数据污染
  - 新issue是真实的、未见过的问题
  - 问题描述可能不完整
- **$500计算投入** — LLM 20 Questions冠军在服务器上花了~$500

---

## 5. LLM 推理优化

### 5.1 推理框架选择

```
需要最高性能 → TensorRT-LLM(NVIDIA GPU优化, FP8)
需要高吞吐   → lmdeploy/TurboMind
需要前缀缓存 → SGLang(RadixAttention, 共享前缀5x加速)
通用/快速上手 → vLLM
```

### 5.2 量化策略选择

```
精度优先 → FP8(损失极小, 1.5x加速)
显存优先 → AWQ 4bit(显存减半+)
长序列   → 8bit KV cache(节省KV内存)
训练时   → QLoRA 4bit(训练大模型)
```

### 5.3 投机解码

```
原理: 小模型快速生成draft tokens → 大模型并行验证
ReDrafter(Apple): 3-token proposal, 65%接受率, 1.8x加速
使用: AIMO2 1st 结合FP8, 总计2.7x加速
```

### 5.4 蒸馏 Pipeline

```
Step 1: Teacher训练
  └─ 大模型(70B) + QLoRA微调 → 收集logit分布

Step 2: Student训练
  └─ 小模型(9B) + LoRA
  └─ Loss = CE(hard labels) + KL(teacher logits) + Cosine(reps)

Step 3: 量化部署
  └─ Student → 8bit/4bit → 推理

案例: LMSYS 1st
  ├── Teacher: Llama3-70B + Qwen2-72B (QLoRA)
  ├── Student: Gemma2-9B (LoRA)
  ├── 5-fold LoRA adapter平均
  └── 8bit推理
```

---

## 6. Test-Time Compute Scaling

### 概念

推理时花更多计算换更好结果(vs 训练时scaling)

### 四种方法

| 方法 | 描述 | 案例 | 效果 |
|------|------|------|------|
| **并行采样** | 生成N个解, 投票 | AIMO: 64次采样 | 最通用 |
| **序列扩展** | CoT推理, 迭代修正 | DeepSeek-R1, QwQ-32B | 推理任务 |
| **TTT** | 每任务微调模型 | ARC所有top方案 | 推理/适应 |
| **GenSelect** | 训练选择器选最优 | AIMO2 1st | 优于voting |

### ICLR 2025 关键发现

> 小模型 + 高级推理算法 = Pareto最优
> 14B模型通过推理优化, 胜过更大模型

- 推理时scaling可能比训练时scaling更高效
- DeepSeek-R1支持思考链开关(toggle latency vs depth)
- 2026趋势: 更多inference-time scaling

---

## 7. Agent 框架与工具

### 竞赛中使用的框架

| 框架 | 描述 | 性能 |
|------|------|------|
| **LLaMA-Factory** | 标准化LoRA/QLoRA微调, YAML配置 | 广泛使用 |
| **NeMo RL + Skills** | NVIDIA RL框架, AIMO/ARC使用 | AIMO1/2 1st |
| **vLLM** | LLM推理服务 | 通用 |
| **lmdeploy** | 高吞吐推理 | AIMO2 2nd/5th/7th |

### 研究中的Agent框架

| 框架 | 描述 | 性能 |
|------|------|------|
| **AutoKaggle** | 5-Agent系统(Reader/Planner/Developer/Reviewer/Summarizer) | 85%提交率, 超AIDE 28% |
| **Agent K v1.0** | 自主Kaggle Agent | GM水平(6金3银7铜) |
| **MLE-bench** | OpenAI, 75个Kaggle赛题基准 | o1+AIDE=16.9%铜牌 |
| **MLE-Dojo** | Gym风格RL环境, 200+赛题 | 研究用 |

### Google/Kaggle AI Agents Intensive (2025.11)

5天课程核心内容:
1. Agent循环: 观察→思考→行动→反馈
2. 工具调用: MCP标准化协议
3. 记忆系统: 短期/长期记忆管理
4. 多Agent协调: 专业化Agent协作
5. 关键发现: 工具设计不当导致上下文膨胀和行为失控
