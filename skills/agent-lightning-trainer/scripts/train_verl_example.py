#!/usr/bin/env python3
"""
Agent Lightning VERL（RL 微调）训练示例

使用 GRPO 算法通过强化学习微调 LLM，让 Agent 从交互中学习。

场景示例：数学推理 Agent
- 输入：数学问题
- 输出：逐步推理 + 最终答案
- 奖励：答案是否正确

用法:
    python train_verl_example.py
    python train_verl_example.py --model Qwen/Qwen2.5-3B-Instruct --n_gpus 4
    python train_verl_example.py --dry_run
"""

import argparse
import json
import os
import re
import sys


def create_sample_math_data():
    """创建示例数学训练/验证数据"""
    train_tasks = [
        {"question": "计算 15 + 27 = ?", "answer": "42"},
        {"question": "计算 8 × 7 = ?", "answer": "56"},
        {"question": "一个长方形长5cm宽3cm，面积是多少？", "answer": "15"},
        {"question": "小明有12个苹果，给了小红5个，还剩几个？", "answer": "7"},
        {"question": "计算 100 - 37 = ?", "answer": "63"},
        {"question": "一打铅笔有12支，3打有多少支？", "answer": "36"},
        {"question": "计算 144 ÷ 12 = ?", "answer": "12"},
        {"question": "三角形三边长3、4、5，是直角三角形吗？面积是多少？", "answer": "6"},
    ]

    val_tasks = [
        {"question": "计算 23 + 45 = ?", "answer": "68"},
        {"question": "计算 9 × 8 = ?", "answer": "72"},
        {"question": "小明有20元，买了一本8元的书，还剩多少？", "answer": "12"},
    ]

    return train_tasks, val_tasks


def extract_answer(text):
    """从模型输出中提取答案"""
    # 尝试从 \boxed{} 中提取
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 尝试从"答案是"后提取
    answer_match = re.search(r'答案[是为：:]\s*(\d+)', text)
    if answer_match:
        return answer_match.group(1).strip()

    # 提取最后一个数字
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Agent Lightning VERL 训练示例")

    # 模型配置
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace 模型 ID 或本地路径")

    # 训练超参
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="全局训练 batch size")
    parser.add_argument("--mini_batch_size", type=int, default=32,
                        help="PPO mini batch size")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="学习率")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="GRPO 每组采样数")

    # 序列长度
    parser.add_argument("--max_prompt_length", type=int, default=4096,
                        help="Prompt 最大长度")
    parser.add_argument("--max_response_length", type=int, default=2048,
                        help="回答最大长度")

    # 硬件
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="每节点 GPU 数")
    parser.add_argument("--gpu_memory_util", type=float, default=0.6,
                        help="vLLM GPU 显存利用率")

    # 数据
    parser.add_argument("--data_path", type=str, default=None,
                        help="自定义数据路径 (JSON 格式)")

    # 控制
    parser.add_argument("--dry_run", action="store_true",
                        help="仅 dry-run 验证")
    parser.add_argument("--val_before_train", action="store_true", default=True,
                        help="训练前先验证")

    args = parser.parse_args()

    # 加载数据
    if args.data_path:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        train_tasks = data.get("train", data.get("train_tasks", []))
        val_tasks = data.get("val", data.get("val_tasks", []))
    else:
        train_tasks, val_tasks = create_sample_math_data()

    print("=" * 60)
    print("Agent Lightning VERL (GRPO) Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"GPUs: {args.n_gpus}")
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks: {len(val_tasks)}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"GRPO n: {args.n_samples}")
    print()

    try:
        import agentlightning as agl
        from agentlightning.algorithm import VERL
    except ImportError:
        print("ERROR: agentlightning 未安装")
        print("  pip install agentlightning")
        print("  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128")
        print("  pip install flash-attn --no-build-isolation")
        print("  pip install vllm==0.10.2 verl==0.5.0")
        sys.exit(1)

    # 定义 Agent
    class MathAgent(agl.LitAgent):
        def rollout(self, task, resources):
            """数学推理 Agent"""
            llm = resources["main_llm"]

            # 构建提示
            messages = [
                {"role": "system", "content": "你是一个数学助手。请逐步推理，最后用 \\boxed{} 给出答案。"},
                {"role": "user", "content": task["question"]},
            ]

            # 调用 LLM
            response = llm.chat.completions.create(
                model="main_llm",
                messages=messages,
            )
            output = response.choices[0].message.content

            # 计算奖励
            predicted = extract_answer(output)
            expected = str(task["answer"]).strip()
            reward = 1.0 if predicted == expected else 0.0

            return reward

    # VERL 配置
    verl_config = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": args.train_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
        },
        "actor_rollout_ref": {
            "model": {
                "path": args.model,
            },
            "rollout": {
                "name": "vllm",
                "n": args.n_samples,
                "multi_turn": {"format": "hermes"},
                "gpu_memory_utilization": args.gpu_memory_util,
                "tensor_model_parallel_size": 1,
            },
            "actor": {
                "ppo_mini_batch_size": args.mini_batch_size,
                "lr": args.lr,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
            },
        },
        "trainer": {
            "n_gpus_per_node": args.n_gpus,
            "val_before_train": args.val_before_train,
        },
    }

    print("VERL Config:")
    print(json.dumps(verl_config, indent=2))
    print()

    # 创建 Algorithm 和 Trainer
    algorithm = VERL(config=verl_config)
    trainer = agl.Trainer(
        agent=MathAgent(),
        algorithm=algorithm,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
    )

    if args.dry_run:
        print("Running dry-run (dev mode)...")
        trainer.dev()
        print("\nDry-run completed successfully!")
    else:
        print("Starting VERL GRPO training...")
        trainer.fit()
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
