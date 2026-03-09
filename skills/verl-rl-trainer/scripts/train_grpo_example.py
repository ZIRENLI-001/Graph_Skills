#!/usr/bin/env python3
"""
verl GRPO 训练示例脚本

使用 GRPO（Group Relative Policy Optimization）算法进行 RL 训练。
GRPO 无需 Critic 模型，使用组内相对排序作为基线。

用法:
    # 直接运行（会生成并执行训练命令）
    python train_grpo_example.py

    # 仅打印命令不执行
    python train_grpo_example.py --dry_run

    # 自定义参数
    python train_grpo_example.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train_data ~/data/gsm8k/train.parquet \
        --val_data ~/data/gsm8k/test.parquet \
        --n_gpus 4 \
        --use_lora
"""

import argparse
import os
import subprocess
import sys


def build_grpo_command(args) -> list[str]:
    """构建 GRPO 训练命令"""
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # 算法配置
        "algorithm.adv_estimator=grpo",
        # 数据配置
        f"data.train_files={args.train_data}",
        f"data.val_files={args.val_data}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        # 模型配置
        f"actor_rollout_ref.model.path={args.model}",
        # Actor 配置
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={args.micro_batch_size}",
        f"actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_coef}",
        f"actor_rollout_ref.actor.ppo_epochs={args.ppo_epochs}",
        # Rollout 配置
        f"actor_rollout_ref.rollout.name={args.rollout_engine}",
        f"actor_rollout_ref.rollout.n={args.num_generations}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_util}",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        # Trainer 配置
        f"trainer.n_gpus_per_node={args.n_gpus}",
        f"trainer.nnodes={args.n_nodes}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.val_before_train={args.val_before_train}",
    ]

    # LoRA 配置
    if args.use_lora:
        cmd.extend([
            f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={args.lora_alpha}",
        ])

    # 日志配置
    if args.logger:
        cmd.append(f"trainer.logger=[{','.join(args.logger)}]")
        if 'wandb' in args.logger:
            cmd.append(f"trainer.project_name={args.project_name}")
            cmd.append(f"trainer.experiment_name={args.experiment_name}")

    # 保存频率
    if args.save_freq > 0:
        cmd.append(f"trainer.save_freq={args.save_freq}")
    if args.test_freq > 0:
        cmd.append(f"trainer.test_freq={args.test_freq}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="verl GRPO 训练")

    # 模型与数据
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace 模型 ID 或本地路径")
    parser.add_argument("--train_data", type=str, default="~/data/gsm8k/train.parquet",
                        help="训练数据路径")
    parser.add_argument("--val_data", type=str, default="~/data/gsm8k/test.parquet",
                        help="验证数据路径")

    # 训练超参
    parser.add_argument("--train_batch_size", type=int, default=256,
                        help="全局训练 batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64,
                        help="PPO mini batch size（全局）")
    parser.add_argument("--micro_batch_size", type=int, default=2,
                        help="PPO micro batch size（每 GPU）")
    parser.add_argument("--ppo_epochs", type=int, default=1,
                        help="每次 rollout 更新轮数")
    parser.add_argument("--kl_coef", type=float, default=0.001,
                        help="KL loss 系数")
    parser.add_argument("--total_epochs", type=int, default=1,
                        help="总训练轮数")

    # GRPO 特定参数
    parser.add_argument("--num_generations", type=int, default=5,
                        help="每个 prompt 生成的回答数")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度")

    # 序列长度
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Prompt 最大长度")
    parser.add_argument("--max_response_length", type=int, default=1024,
                        help="生成回答最大长度")

    # Rollout 配置
    parser.add_argument("--rollout_engine", type=str, default="vllm",
                        choices=["vllm", "sglang"],
                        help="推理引擎")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="张量并行度")
    parser.add_argument("--gpu_memory_util", type=float, default=0.4,
                        help="vLLM GPU 显存利用率")

    # 硬件配置
    parser.add_argument("--n_gpus", type=int, default=4,
                        help="每节点 GPU 数")
    parser.add_argument("--n_nodes", type=int, default=1,
                        help="节点数")

    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="使用 LoRA 训练")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")

    # 日志与保存
    parser.add_argument("--logger", nargs="+", default=["console"],
                        help="日志系统: console, wandb, tensorboard")
    parser.add_argument("--project_name", type=str, default="verl-grpo",
                        help="WandB 项目名")
    parser.add_argument("--experiment_name", type=str, default="grpo-experiment",
                        help="实验名")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="保存频率（步数）")
    parser.add_argument("--test_freq", type=int, default=-1,
                        help="评估频率（步数）")
    parser.add_argument("--val_before_train", type=str, default="False",
                        help="训练前先评估")

    # 控制
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印命令不执行")

    args = parser.parse_args()

    # 构建命令
    cmd = build_grpo_command(args)

    print("=" * 60)
    print("verl GRPO Training Command")
    print("=" * 60)
    print()
    print("Command:")
    print(f"  {' '.join(cmd[:3])} \\")
    for c in cmd[3:]:
        print(f"    {c} \\")
    print()

    if args.dry_run:
        print("[DRY RUN] Command not executed.")
        return

    print("Starting training...")
    print("=" * 60)

    result = subprocess.run(cmd, env=os.environ.copy())
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
