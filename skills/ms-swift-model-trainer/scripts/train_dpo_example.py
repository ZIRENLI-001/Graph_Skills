#!/usr/bin/env python3
"""
ms-swift DPO (Direct Preference Optimization) 训练模板

DPO 适用场景:
  - 有偏好对数据（chosen/rejected），希望模型输出符合人类偏好
  - 通常在 SFT 之后进行，作为对齐训练的第二阶段
  - 不需要训练单独的奖励模型（相比 PPO 更简单高效）

数据格式要求 (严格):
  {"messages": [..., {"role": "assistant", "content": "chosen_response"}],
   "rejected_response": "rejected_response_text"}

  注意: DPO 对数据格式非常严格，必须包含 rejected_response 字段。
  训练前务必使用 dataset_validator.py 验证数据。

CLI 用法:
  CUDA_VISIBLE_DEVICES=0 swift rlhf \\
      --rlhf_type dpo \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \\
      --train_type lora \\
      --output_dir output/dpo_lora

Python API 用法:
  python train_dpo_example.py
"""

import os


def train_dpo_lora():
    """LoRA DPO 训练示例。

    前置条件:
      1. 建议先进行 SFT 训练获得基线模型
      2. 或直接使用 Instruct 系列模型（已经过 SFT）
      3. 数据集必须包含 rejected_response 字段
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import rlhf_main, RLHFArguments

    result = rlhf_main(RLHFArguments(
        # === RLHF 类型 ===
        rlhf_type='dpo',                   # 'dpo' | 'kto' | 'simpo' | 'orpo' | 'cpo'

        # === 模型配置 ===
        model='Qwen/Qwen2.5-7B-Instruct',
        # 如果使用 SFT 后的模型:
        # model='Qwen/Qwen2.5-7B-Instruct',
        # adapters='output/sft_lora/vx-xxx/checkpoint-xxx',

        # === 数据集配置 ===
        dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji'],
        # 本地数据集:
        # dataset=['/path/to/dpo_data.jsonl'],

        # === 训练方式 ===
        train_type='lora',
        # lora_rank=8,
        # lora_alpha=32,

        # === DPO 超参数 ===
        # beta=0.1,                         # DPO 温度参数，控制偏好强度
                                              # 较大 beta → 更严格的偏好约束
                                              # 较小 beta → 更宽松，倾向于参考模型

        # === 训练超参数 ===
        torch_dtype='bfloat16',
        num_train_epochs=2,                  # DPO 通常 1-3 个 epoch
        learning_rate=5e-6,                  # DPO 建议学习率比 SFT 小 (5e-6 ~ 5e-7)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,

        # === 保存配置 ===
        output_dir='output/dpo_lora',
        save_steps=50,
        save_total_limit=3,
        logging_steps=5,
    ))

    print(f"\nDPO Training completed!")


def train_dpo_from_sft_checkpoint():
    """从 SFT checkpoint 继续 DPO 训练。

    典型两阶段流程:
      1. SFT: swift sft --model Qwen/Qwen2.5-7B-Instruct --dataset sft_data.jsonl ...
      2. DPO: swift rlhf --rlhf_type dpo --model Qwen/Qwen2.5-7B-Instruct \\
              --adapters output/sft_lora/vx-xxx/checkpoint-xxx --dataset dpo_data.jsonl ...
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import rlhf_main, RLHFArguments

    result = rlhf_main(RLHFArguments(
        rlhf_type='dpo',
        model='Qwen/Qwen2.5-7B-Instruct',
        # 加载 SFT 阶段的 LoRA 权重
        adapters='output/sft_lora/vx-xxx/checkpoint-xxx',
        dataset=['/path/to/dpo_data.jsonl'],
        train_type='lora',
        torch_dtype='bfloat16',
        num_train_epochs=2,
        learning_rate=5e-6,
        output_dir='output/dpo_from_sft',
    ))


def train_dpo_with_custom_data():
    """使用自定义 DPO 数据训练。

    DPO 数据集必须为以下格式之一:

    格式 1 (推荐 - messages + rejected_response):
    {"messages": [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "好的回答（chosen）"}
     ],
     "rejected_response": "差的回答（rejected）"}

    格式 2 (prompt + chosen + rejected, ms-swift 自动转换):
    数据集需要配置 columns 映射
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import rlhf_main, RLHFArguments

    result = rlhf_main(RLHFArguments(
        rlhf_type='dpo',
        model='Qwen/Qwen2.5-7B-Instruct',
        dataset=['/path/to/custom_dpo.jsonl'],
        train_type='lora',
        torch_dtype='bfloat16',
        num_train_epochs=2,
        learning_rate=5e-6,
        output_dir='output/dpo_custom',
    ))


# === 其他偏好优化方法 ===
# ms-swift 同时支持以下方法，只需修改 rlhf_type:
#
# KTO (Kahneman-Tversky Optimization):
#   rlhf_type='kto'
#   数据格式: messages + label (True/False)
#
# SimPO (Simple Preference Optimization):
#   rlhf_type='simpo'
#   无需参考模型，更高效
#
# ORPO (Odds Ratio Preference Optimization):
#   rlhf_type='orpo'
#   将 SFT 和偏好对齐合并为一步
#
# CPO (Contrastive Preference Optimization):
#   rlhf_type='cpo'


# === 训练后推理 ===
# swift infer --adapters output/dpo_lora/vx-xxx/checkpoint-xxx --stream true


if __name__ == '__main__':
    train_dpo_lora()
    # train_dpo_from_sft_checkpoint()
    # train_dpo_with_custom_data()
