#!/usr/bin/env python3
"""
verl PPO 训练示例脚本

使用 PPO（Proximal Policy Optimization）算法进行 RLHF 训练。
PPO 需要 Critic 模型来估计价值函数。

用法:
    # 直接运行
    python train_ppo_example.py

    # 仅打印命令
    python train_ppo_example.py --dry_run

    # 自定义参数
    python train_ppo_example.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --critic_model Qwen/Qwen2.5-7B-Instruct \
        --train_data ~/data/gsm8k/train.parquet \
        --val_data ~/data/gsm8k/test.parquet \
        --n_gpus 8
"""

import argparse
import os
import subprocess
import sys


def build_ppo_command(args) -> list[str]:
    """构建 PPO 训练命令"""
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # 算法配置
        "algorithm.adv_estimator=gae",
        f"algorithm.gamma={args.gamma}",
        f"algorithm.lam={args.lam}",
        f"algorithm.kl_ctrl.type={args.kl_ctrl_type}",
        f"algorithm.kl_ctrl.kl_coef={args.kl_coef}",
        # 数据配置
        f"data.train_files={args.train_data}",
        f"data.val_files={args.val_data}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        # Actor 模型配置
        f"actor_rollout_ref.model.path={args.model}",
        # Actor 训练配置
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={args.micro_batch_size}",
        f"actor_rollout_ref.actor.ppo_epochs={args.ppo_epochs}",
        f"actor_rollout_ref.actor.clip_ratio={args.clip_ratio}",
        # Rollout 配置
        f"actor_rollout_ref.rollout.name={args.rollout_engine}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_util}",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        # Critic 配置（PPO 必需）
        f"critic.model.path={args.critic_model}",
        f"critic.ppo_micro_batch_size={args.critic_micro_batch_size}",
        f"critic.ppo_epochs={args.critic_ppo_epochs}",
        f"critic.cliprange_value={args.cliprange_value}",
        # Trainer 配置
        f"trainer.n_gpus_per_node={args.n_gpus}",
        f"trainer.nnodes={args.n_nodes}",
        f"trainer.total_epochs={args.total_epochs}",
    ]

    # LoRA 配置
    if args.use_lora:
        cmd.extend([
            f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={args.lora_alpha}",
        ])

    # Reward Model
    if args.reward_model:
        cmd.extend([
            "reward_model.enable=True",
            f"reward_model.model.path={args.reward_model}",
            f"reward_model.micro_batch_size={args.rm_micro_batch_size}",
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

    return cmd


def main():
    parser = argparse.ArgumentParser(description="verl PPO 训练")

    # 模型与数据
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Actor 模型路径")
    parser.add_argument("--critic_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Critic 模型路径（PPO 必需）")
    parser.add_argument("--reward_model", type=str, default=None,
                        help="Reward Model 路径（可选，不指定则使用 rule-based reward）")
    parser.add_argument("--train_data", type=str, default="~/data/gsm8k/train.parquet")
    parser.add_argument("--val_data", type=str, default="~/data/gsm8k/test.parquet")

    # 训练超参
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--total_epochs", type=int, default=1)

    # KL 控制
    parser.add_argument("--kl_ctrl_type", type=str, default="fixed",
                        choices=["fixed", "adaptive"])
    parser.add_argument("--kl_coef", type=float, default=0.001)

    # GAE 参数
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)

    # Critic 参数
    parser.add_argument("--critic_micro_batch_size", type=int, default=2)
    parser.add_argument("--critic_ppo_epochs", type=int, default=1)
    parser.add_argument("--cliprange_value", type=float, default=0.5)

    # Reward Model 参数
    parser.add_argument("--rm_micro_batch_size", type=int, default=2)

    # 序列长度
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=1024)

    # Rollout 配置
    parser.add_argument("--rollout_engine", type=str, default="vllm",
                        choices=["vllm", "sglang"])
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_memory_util", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)

    # 硬件
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--n_nodes", type=int, default=1)

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # 日志
    parser.add_argument("--logger", nargs="+", default=["console"])
    parser.add_argument("--project_name", type=str, default="verl-ppo")
    parser.add_argument("--experiment_name", type=str, default="ppo-experiment")
    parser.add_argument("--save_freq", type=int, default=-1)

    # 控制
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    cmd = build_ppo_command(args)

    print("=" * 60)
    print("verl PPO Training Command")
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

    print("Starting PPO training...")
    print("=" * 60)

    result = subprocess.run(cmd, env=os.environ.copy())
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
