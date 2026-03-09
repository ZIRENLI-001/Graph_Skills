#!/usr/bin/env python3
"""
ms-swift SFT (Supervised Fine-Tuning) 训练模板

SFT 适用场景:
  - 有高质量的示例数据（客服对话、代码生成、领域问答等）
  - 想让模型学习特定的行为模式和输出风格
  - 作为 DPO/GRPO 等对齐训练的前置步骤

数据格式 (messages, 推荐):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

CLI 用法:
  # 单卡 LoRA
  CUDA_VISIBLE_DEVICES=0 swift sft \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --dataset AI-ModelScope/alpaca-gpt4-data-en \\
      --train_type lora \\
      --output_dir output/sft_lora

  # 多卡全参数 + DeepSpeed
  CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --dataset AI-ModelScope/alpaca-gpt4-data-en \\
      --train_type full \\
      --deepspeed zero3 \\
      --output_dir output/sft_full

Python API 用法:
  python train_sft_example.py
"""

import os


def train_sft_lora():
    """单卡 LoRA SFT 训练示例。

    推荐用于 3B+ 参数模型，显存需求约 16-24GB。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import sft_main, TrainArguments

    result = sft_main(TrainArguments(
        # === 模型配置 ===
        model='Qwen/Qwen2.5-7B-Instruct',  # 模型ID (ModelScope默认, 加 --use_hf 用HuggingFace)
        # model_type='qwen2_5',             # 通常自动推断，无需指定

        # === 数据集配置 ===
        dataset=[
            'AI-ModelScope/alpaca-gpt4-data-en#1000',  # #N 表示取前N条
            # '/path/to/local/data.jsonl',              # 支持本地文件路径
        ],
        # max_length=2048,                  # 最大序列长度，默认自动

        # === 训练方式 ===
        train_type='lora',                  # 'lora' | 'full' | 'adalora' | 'llamapro' | ...
        # lora_rank=8,                      # LoRA rank，默认8
        # lora_alpha=32,                    # LoRA alpha，默认32
        # lora_dropout=0.05,                # LoRA dropout

        # === 训练超参数 ===
        torch_dtype='bfloat16',             # 'bfloat16' | 'float16' | 'float32'
        num_train_epochs=3,
        learning_rate=1e-4,                 # LoRA 推荐 1e-4，Full 推荐 1e-5
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,     # 有效batch = batch_size × grad_accum × gpu_num
        warmup_ratio=0.05,
        # lr_scheduler_type='cosine',       # 默认 cosine

        # === 评估配置 ===
        eval_steps=100,                     # 每N步评估一次
        # eval_strategy='steps',
        # val_dataset=['/path/to/eval.jsonl'],

        # === 保存配置 ===
        output_dir='output/sft_lora',
        save_steps=100,
        save_total_limit=3,                 # 最多保留N个checkpoint

        # === Hub 推送（可选） ===
        # push_to_hub=True,
        # hub_model_id='your-username/model-name',
        # use_hf=True,                      # 推送到 HuggingFace Hub（默认 ModelScope）

        # === 日志与监控 ===
        logging_steps=5,
        # report_to='tensorboard',          # 'tensorboard' | 'wandb' | 'swanlab'
    ))

    print(f"\nTraining completed!")
    print(f"Best checkpoint: {result.get('best_model_checkpoint', 'N/A')}")


def train_sft_full_multi_gpu():
    """多卡全参数 SFT 训练示例。

    需要 4x A100 80GB 或类似配置。
    使用 DeepSpeed ZeRO-3 进行参数分片。

    CLI 方式 (推荐多卡训练使用CLI):
    CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset AI-ModelScope/alpaca-gpt4-data-en \\
        --train_type full \\
        --torch_dtype bfloat16 \\
        --deepspeed zero3 \\
        --num_train_epochs 3 \\
        --learning_rate 1e-5 \\
        --per_device_train_batch_size 2 \\
        --gradient_accumulation_steps 4 \\
        --output_dir output/sft_full
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    from swift.llm import sft_main, TrainArguments

    result = sft_main(TrainArguments(
        model='Qwen/Qwen2.5-7B-Instruct',
        dataset=['AI-ModelScope/alpaca-gpt4-data-en'],
        train_type='full',
        deepspeed='zero3',
        torch_dtype='bfloat16',
        num_train_epochs=3,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        output_dir='output/sft_full',
    ))

    print(f"\nTraining completed!")


def train_sft_custom_dataset():
    """使用自定义本地数据集的 SFT 训练示例。

    支持的数据集格式:
      - JSONL: 每行一个 JSON 对象
      - JSON: JSON 数组
      - CSV: 带表头的 CSV 文件

    列名映射:
      如果数据集列名不是标准格式，使用 --columns 参数映射:
      --columns '{"question": "query", "answer": "response"}'
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import sft_main, TrainArguments

    result = sft_main(TrainArguments(
        model='Qwen/Qwen2.5-7B-Instruct',
        dataset=[
            '/path/to/train_data.jsonl',
            '/path/to/extra_data.jsonl#500',  # 取前500条
        ],
        val_dataset=['/path/to/eval_data.jsonl'],
        train_type='lora',
        torch_dtype='bfloat16',
        num_train_epochs=3,
        learning_rate=1e-4,
        output_dir='output/sft_custom',
        # 列名映射（如果字段名不标准）
        # columns={'question': 'query', 'answer': 'response'},
    ))


# === 训练后推理 ===
# swift infer --adapters output/sft_lora/vx-xxx/checkpoint-xxx --stream true --temperature 0


if __name__ == '__main__':
    # 选择运行哪个示例
    train_sft_lora()
    # train_sft_full_multi_gpu()
    # train_sft_custom_dataset()
