#!/usr/bin/env python3
"""
ms-swift GRPO (Group Relative Policy Optimization) 训练模板

GRPO 适用场景:
  - 使用可验证的奖励信号优化模型（如数学题正确性、代码执行结果）
  - 无需训练单独的奖励模型（与 PPO 的关键区别）
  - 适合提升模型推理能力（如 DeepSeek-R1 的训练方法）

数据格式:
  {"messages": [{"role": "user", "content": "问题"}], "solution": "标准答案"}
  注意: messages 中通常不包含 assistant 回复，模型会自行生成多个候选回答。
  额外字段（如 solution）会自动传递给 ORM 奖励函数。

硬件要求:
  - GRPO 需要同时进行推理和训练，推荐 4+ GPU
  - 使用 vLLM 加速推理（推荐 colocate 模式）

CLI 用法:
  CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift rlhf \\
      --rlhf_type grpo \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --dataset AI-MO/NuminaMath-TIR#10000 \\
      --train_type lora \\
      --use_vllm true \\
      --vllm_mode colocate \\
      --output_dir output/grpo_lora

Python API 用法:
  python train_grpo_example.py
"""

import os
import re


# =============================================================================
# 1. 自定义 ORM (Outcome Reward Model) 函数
# =============================================================================

def math_orm(completions, solution, **kwargs):
    """数学题奖励函数示例。

    检查模型回答中是否包含正确答案。

    Args:
        completions: 模型生成的候选回答列表（必需参数）
        solution: 数据集中的标准答案（通过额外字段传递）
        **kwargs: 其他额外字段

    Returns:
        list[float]: 每个候选回答的奖励分数
    """
    rewards = []
    for completion in completions:
        # 提取模型回答内容
        if isinstance(completion, list):
            content = completion[-1].get('content', '') if completion else ''
        else:
            content = str(completion)

        # 尝试从 \\boxed{} 中提取答案
        boxed_match = re.search(r'\\boxed\{(.+?)\}', content)
        if boxed_match:
            answer = boxed_match.group(1).strip()
        else:
            # 取最后一行数字
            numbers = re.findall(r'-?\d+\.?\d*', content)
            answer = numbers[-1] if numbers else ''

        # 比较答案
        reward = 1.0 if answer.strip() == str(solution).strip() else 0.0
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """格式奖励函数示例。

    奖励模型输出结构化格式（如使用 <think>...</think> 标签）。
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[-1].get('content', '') if completion else ''
        else:
            content = str(completion)

        # 检查是否包含思考过程标签
        has_think = '<think>' in content and '</think>' in content
        reward = 0.5 if has_think else 0.0

        # 检查答案格式
        has_answer = '\\boxed{' in content or '答案' in content
        reward += 0.5 if has_answer else 0.0

        rewards.append(reward)

    return rewards


def code_orm(completions, test_cases, **kwargs):
    """代码正确性奖励函数示例。

    通过执行测试用例验证代码正确性。
    注意: 生产环境中应在沙箱中执行代码。

    数据集格式:
    {"messages": [{"role": "user", "content": "写一个函数..."}],
     "test_cases": "assert add(1,2)==3\\nassert add(0,0)==0"}
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[-1].get('content', '') if completion else ''
        else:
            content = str(completion)

        # 提取代码块
        code_match = re.search(r'```python\n(.+?)```', content, re.DOTALL)
        code = code_match.group(1) if code_match else content

        try:
            # 警告: 生产环境应使用沙箱执行
            exec_globals = {}
            exec(code, exec_globals)
            exec(test_cases, exec_globals)
            reward = 1.0
        except Exception:
            reward = 0.0

        rewards.append(reward)

    return rewards


# =============================================================================
# 2. GRPO 训练示例
# =============================================================================

def train_grpo_math():
    """数学推理 GRPO 训练示例。

    使用 vLLM colocate 模式加速推理。
    colocate 模式: vLLM 引擎与训练进程共享 GPU，显存动态分配。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    from swift.llm import rlhf_main, RLHFArguments

    result = rlhf_main(RLHFArguments(
        # === RLHF 类型 ===
        rlhf_type='grpo',

        # === 模型配置 ===
        model='Qwen/Qwen2.5-7B-Instruct',

        # === 数据集配置 ===
        dataset=['AI-MO/NuminaMath-TIR#10000'],
        # 本地数据集:
        # dataset=['/path/to/grpo_data.jsonl'],

        # === 训练方式 ===
        train_type='lora',
        # lora_rank=8,

        # === GRPO 超参数 ===
        num_generations=8,                   # 每个 prompt 生成的候选数量
        # temperature=0.7,                   # 生成温度
        # max_new_tokens=2048,               # 最大生成长度

        # === vLLM 加速配置 ===
        use_vllm=True,
        vllm_mode='colocate',                # 'colocate' | 'server'
                                              # colocate: 共享GPU，显存动态分配
                                              # server: 独立vLLM服务器，需额外GPU

        # === 奖励函数 ===
        # 使用内置奖励函数或自定义:
        # reward_funcs=['math_orm', 'format_reward'],
        # external_plugins='train_grpo_example.py',  # 包含自定义ORM的文件

        # === 训练超参数 ===
        torch_dtype='bfloat16',
        num_train_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        # === 保存配置 ===
        output_dir='output/grpo_math',
        save_steps=50,
        logging_steps=1,
    ))

    print(f"\nGRPO Training completed!")


def train_grpo_with_server_mode():
    """使用 vLLM server 模式的 GRPO 训练。

    server 模式: vLLM 在独立进程/GPU上运行，适合大模型。
    需要更多 GPU，但训练和推理不争抢显存。

    CLI 方式:
    # 先启动 vLLM server (单独的 GPU):
    CUDA_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2.5-7B-Instruct --port 8000

    # 再启动 GRPO 训练:
    CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 swift rlhf \\
        --rlhf_type grpo \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset AI-MO/NuminaMath-TIR#10000 \\
        --train_type lora \\
        --use_vllm true \\
        --vllm_mode server \\
        --vllm_server_url http://localhost:8000 \\
        --output_dir output/grpo_server
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    from swift.llm import rlhf_main, RLHFArguments

    result = rlhf_main(RLHFArguments(
        rlhf_type='grpo',
        model='Qwen/Qwen2.5-7B-Instruct',
        dataset=['AI-MO/NuminaMath-TIR#10000'],
        train_type='lora',
        use_vllm=True,
        vllm_mode='server',
        # vllm_server_url='http://localhost:8000',
        num_generations=8,
        torch_dtype='bfloat16',
        learning_rate=5e-6,
        output_dir='output/grpo_server',
    ))


def train_grpo_without_vllm():
    """不使用 vLLM 的 GRPO 训练（适合小模型或单 GPU）。

    不使用 vLLM 时，推理速度较慢，但显存管理更简单。
    适合 < 3B 的小模型或调试阶段。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import rlhf_main, RLHFArguments

    result = rlhf_main(RLHFArguments(
        rlhf_type='grpo',
        model='Qwen/Qwen2.5-1.5B-Instruct',  # 小模型
        dataset=['AI-MO/NuminaMath-TIR#1000'],
        train_type='lora',
        use_vllm=False,                      # 不使用 vLLM
        num_generations=4,                   # 减少候选数量以节省显存
        torch_dtype='bfloat16',
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        output_dir='output/grpo_no_vllm',
    ))


# =============================================================================
# 其他 GRPO 家族算法
# =============================================================================
# ms-swift 内置丰富的 GRPO 家族算法，只需修改 rlhf_type:
#
# DAPO (Decoupled Alignment Policy Optimization):
#   rlhf_type='dapo'
#
# RLOO (REINFORCE Leave-One-Out):
#   rlhf_type='rloo'
#
# Reinforce++:
#   rlhf_type='reinforce_plus_plus'


# === 训练后推理 ===
# swift infer --adapters output/grpo_math/vx-xxx/checkpoint-xxx --stream true


if __name__ == '__main__':
    train_grpo_math()
    # train_grpo_with_server_mode()
    # train_grpo_without_vllm()
