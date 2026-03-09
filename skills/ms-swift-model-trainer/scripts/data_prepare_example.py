#!/usr/bin/env python3
"""
ms-swift 数据准备模板脚本

演示如何获取、处理和转换数据集，使其符合 ms-swift 训练格式要求。
支持从 HuggingFace / ModelScope 下载数据，以及本地数据格式转换。

ms-swift 支持的标准数据格式:
  1. messages 格式（推荐）
  2. shareGPT/conversations 格式
  3. alpaca 格式
  4. query/response 格式

用法:
  # 直接运行查看示例
  python data_prepare_example.py

  # 转换本地数据
  python data_prepare_example.py --input data.json --output data_converted.jsonl --target_format messages --task sft
"""

import argparse
import json
import os
from pathlib import Path


# =============================================================================
# 1. 从 HuggingFace / ModelScope 下载数据集
# =============================================================================

def download_from_huggingface(dataset_name: str, split: str = 'train', max_samples: int = None) -> list[dict]:
    """从 HuggingFace Hub 下载数据集。

    Args:
        dataset_name: 数据集名称，如 'tatsu-lab/alpaca'
        split: 数据集分割，如 'train', 'test'
        max_samples: 最大样本数，None 表示全部
    """
    from datasets import load_dataset

    print(f"Downloading from HuggingFace: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return [dict(row) for row in ds]


def download_from_modelscope(dataset_name: str, split: str = 'train', max_samples: int = None) -> list[dict]:
    """从 ModelScope Hub 下载数据集。

    Args:
        dataset_name: 数据集名称，如 'AI-ModelScope/alpaca-gpt4-data-en'
        split: 数据集分割
        max_samples: 最大样本数
    """
    from modelscope.msdatasets import MsDataset

    print(f"Downloading from ModelScope: {dataset_name} (split={split})")
    ds = MsDataset.load(dataset_name, split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return [dict(row) for row in ds]


# =============================================================================
# 2. 格式转换函数
# =============================================================================

def alpaca_to_messages(sample: dict) -> dict:
    """将 alpaca 格式转换为 messages 格式。

    alpaca: {"instruction": "...", "input": "...", "output": "..."}
    messages: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    system = sample.get('system', '')

    # 合并 instruction 和 input
    query = f"{instruction}\n{input_text}".strip() if input_text else instruction

    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': query})
    messages.append({'role': 'assistant', 'content': output})

    return {'messages': messages}


def sharegpt_to_messages(sample: dict) -> dict:
    """将 shareGPT/conversations 格式转换为 messages 格式。

    sharegpt: {"conversation": [{"human": "...", "assistant": "..."}]}
    messages: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    conv_key = 'conversation' if 'conversation' in sample else 'conversations'
    conversation = sample.get(conv_key, [])
    system = sample.get('system', '')

    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})

    for turn in conversation:
        if isinstance(turn, dict):
            # 格式 1: {"human": "...", "assistant": "..."}
            if 'human' in turn:
                messages.append({'role': 'user', 'content': turn['human']})
                if 'assistant' in turn:
                    messages.append({'role': 'assistant', 'content': turn['assistant']})
            # 格式 2: {"from": "human/gpt", "value": "..."}
            elif 'from' in turn:
                role_map = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
                role = role_map.get(turn['from'], turn['from'])
                messages.append({'role': role, 'content': turn.get('value', '')})

    return {'messages': messages}


def query_response_to_messages(sample: dict) -> dict:
    """将 query/response 格式转换为 messages 格式。

    query_response: {"query": "...", "response": "...", "history": [["q1", "a1"]]}
    messages: {"messages": [...]}
    """
    query = sample.get('query', '')
    # 尝试多种 response 字段名
    response = (sample.get('response') or sample.get('answer') or
                sample.get('output') or sample.get('text') or '')
    system = sample.get('system', '')
    history = sample.get('history', [])

    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})

    # 添加历史对话
    for h_query, h_response in history:
        messages.append({'role': 'user', 'content': h_query})
        messages.append({'role': 'assistant', 'content': h_response})

    messages.append({'role': 'user', 'content': query})
    messages.append({'role': 'assistant', 'content': response})

    return {'messages': messages}


# =============================================================================
# 3. DPO / GRPO 数据准备
# =============================================================================

def prepare_dpo_data(sample: dict, rejected_key: str = 'rejected_response') -> dict:
    """为 DPO 数据添加 rejected_response 字段。

    DPO 格式要求:
    {"messages": [..., {"role": "assistant", "content": "chosen_response"}],
     "rejected_response": "rejected_response_text"}
    """
    result = dict(sample)

    # 如果已有 rejected_response，直接返回
    if 'rejected_response' in result:
        return result

    # 尝试从其他字段名映射
    for key in ['rejected', 'rejected_answer', 'bad_response', 'lose']:
        if key in result:
            result['rejected_response'] = result.pop(key)
            return result

    # 如果有 chosen/rejected 对（prompt 格式）
    if 'chosen' in result and 'rejected' in result:
        chosen = result.pop('chosen')
        rejected = result.pop('rejected')
        prompt = result.pop('prompt', '')

        messages = [{'role': 'user', 'content': prompt}]
        if isinstance(chosen, str):
            messages.append({'role': 'assistant', 'content': chosen})
        elif isinstance(chosen, list):
            messages.extend(chosen)

        result['messages'] = messages
        result['rejected_response'] = rejected if isinstance(rejected, str) else rejected[-1].get('content', '')

    return result


def prepare_grpo_data(sample: dict, solution_key: str = 'solution') -> dict:
    """为 GRPO 数据添加 solution 字段。

    GRPO 格式: messages + 额外字段（如 solution），额外字段会传递给 ORM。
    {"messages": [{"role": "user", "content": "Solve: 2+2=?"}], "solution": "4"}

    注意: GRPO 的 messages 中通常不包含 assistant 回复，模型会自行生成。
    """
    result = dict(sample)

    # 确保 messages 中没有 assistant 回复（GRPO 自行生成）
    if 'messages' in result:
        result['messages'] = [
            m for m in result['messages']
            if m.get('role') != 'assistant'
        ]

    # 尝试从其他字段映射 solution
    if solution_key not in result:
        for key in ['answer', 'target', 'ground_truth', 'label', 'expected']:
            if key in result:
                result[solution_key] = result[key]
                break

    return result


# =============================================================================
# 4. 数据清洗
# =============================================================================

def clean_data(samples: list[dict], min_length: int = 10, max_length: int = 100000,
               dedup: bool = True) -> list[dict]:
    """数据清洗：去空、去重、截断。

    Args:
        samples: 数据样本列表
        min_length: 最小内容长度
        max_length: 最大内容长度
        dedup: 是否去重
    """
    cleaned = []
    seen = set()

    for sample in samples:
        # 提取内容用于去重和长度检查
        content = json.dumps(sample, ensure_ascii=False)

        # 跳过过短样本
        if len(content) < min_length:
            continue

        # 截断过长样本（仅统计，不修改）
        if len(content) > max_length:
            continue

        # 去重
        if dedup:
            content_hash = hash(content)
            if content_hash in seen:
                continue
            seen.add(content_hash)

        cleaned.append(sample)

    removed = len(samples) - len(cleaned)
    if removed > 0:
        print(f"Cleaned: removed {removed} samples ({removed/len(samples)*100:.1f}%)")

    return cleaned


# =============================================================================
# 5. 保存数据
# =============================================================================

def save_jsonl(samples: list[dict], output_path: str):
    """保存为 JSONL 格式。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved {len(samples)} samples to {output_path}")


# =============================================================================
# 6. 完整示例
# =============================================================================

def example_sft_preparation():
    """示例：准备 SFT 数据。"""
    print("\n" + "=" * 60)
    print("Example: SFT Data Preparation")
    print("=" * 60)

    # 示例 1: 从 alpaca 格式转换
    alpaca_samples = [
        {"instruction": "翻译成英文", "input": "你好世界", "output": "Hello World"},
        {"instruction": "写一首关于春天的诗", "input": "", "output": "春风拂面柳丝长，桃花笑靥映斜阳。"},
    ]

    messages_samples = [alpaca_to_messages(s) for s in alpaca_samples]
    print("\nAlpaca → Messages:")
    print(json.dumps(messages_samples[0], ensure_ascii=False, indent=2))

    # 示例 2: 直接构造 messages 格式
    direct_sample = {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "什么是机器学习？"},
            {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
        ]
    }
    print("\nDirect messages format:")
    print(json.dumps(direct_sample, ensure_ascii=False, indent=2))

    # ms-swift CLI 使用方式
    print("\n--- ms-swift CLI ---")
    print("swift sft \\")
    print("    --model Qwen/Qwen2.5-7B-Instruct \\")
    print("    --dataset /path/to/sft_data.jsonl \\")
    print("    --train_type lora \\")
    print("    --output_dir output/sft")


def example_dpo_preparation():
    """示例：准备 DPO 数据。"""
    print("\n" + "=" * 60)
    print("Example: DPO Data Preparation")
    print("=" * 60)

    # DPO 数据格式
    dpo_sample = {
        "messages": [
            {"role": "user", "content": "解释什么是量子计算"},
            {"role": "assistant", "content": "量子计算是利用量子力学原理进行信息处理的计算方式..."},
        ],
        "rejected_response": "量子计算就是很快的计算机。"
    }
    print("\nDPO format (messages + rejected_response):")
    print(json.dumps(dpo_sample, ensure_ascii=False, indent=2))

    # 从 prompt/chosen/rejected 转换
    raw_dpo = {
        "prompt": "解释什么是量子计算",
        "chosen": "量子计算是利用量子力学原理进行信息处理的计算方式...",
        "rejected": "量子计算就是很快的计算机。"
    }
    converted = prepare_dpo_data(raw_dpo)
    print("\nConverted from prompt/chosen/rejected:")
    print(json.dumps(converted, ensure_ascii=False, indent=2))

    print("\n--- ms-swift CLI ---")
    print("swift rlhf \\")
    print("    --rlhf_type dpo \\")
    print("    --model Qwen/Qwen2.5-7B-Instruct \\")
    print("    --dataset /path/to/dpo_data.jsonl \\")
    print("    --train_type lora \\")
    print("    --output_dir output/dpo")


def example_grpo_preparation():
    """示例：准备 GRPO 数据。"""
    print("\n" + "=" * 60)
    print("Example: GRPO Data Preparation")
    print("=" * 60)

    # GRPO 数据格式（只有用户问题 + 标准答案）
    grpo_sample = {
        "messages": [
            {"role": "user", "content": "计算 15 × 23 的结果"}
        ],
        "solution": "345"
    }
    print("\nGRPO format (messages + solution for ORM):")
    print(json.dumps(grpo_sample, ensure_ascii=False, indent=2))

    print("\n--- ms-swift CLI ---")
    print("swift rlhf \\")
    print("    --rlhf_type grpo \\")
    print("    --model Qwen/Qwen2.5-7B-Instruct \\")
    print("    --dataset /path/to/grpo_data.jsonl \\")
    print("    --train_type lora \\")
    print("    --use_vllm true \\")
    print("    --vllm_mode colocate \\")
    print("    --output_dir output/grpo")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ms-swift 数据准备工具')
    parser.add_argument('--input', help='输入数据文件路径')
    parser.add_argument('--output', help='输出文件路径 (默认: input_converted.jsonl)')
    parser.add_argument('--source_format', choices=['messages', 'sharegpt', 'alpaca', 'query_response', 'auto'],
                        default='auto', help='源数据格式')
    parser.add_argument('--target_format', choices=['messages'], default='messages',
                        help='目标数据格式 (推荐 messages)')
    parser.add_argument('--task', choices=['sft', 'dpo', 'grpo'], default='sft',
                        help='训练任务类型')
    parser.add_argument('--clean', action='store_true', help='启用数据清洗')

    args = parser.parse_args()

    # 如果没有指定输入文件，运行示例
    if not args.input:
        print("No input file specified. Running examples...\n")
        example_sft_preparation()
        example_dpo_preparation()
        example_grpo_preparation()
        return

    # 加载数据
    from dataset_validator import load_dataset, detect_format
    samples = load_dataset(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    # 检测或使用指定格式
    source_fmt = args.source_format
    if source_fmt == 'auto':
        source_fmt = detect_format(samples)
        print(f"Detected format: {source_fmt}")

    # 转换格式
    converters = {
        'alpaca': alpaca_to_messages,
        'sharegpt': sharegpt_to_messages,
        'query_response': query_response_to_messages,
        'messages': lambda x: x,  # 已经是目标格式
    }

    converter = converters.get(source_fmt)
    if not converter:
        print(f"[ERROR] 不支持的源格式: {source_fmt}")
        return

    converted = [converter(s) for s in samples]

    # 根据任务添加特定字段
    if args.task == 'dpo':
        converted = [prepare_dpo_data(s) for s in converted]
    elif args.task == 'grpo':
        converted = [prepare_grpo_data(s) for s in converted]

    # 数据清洗
    if args.clean:
        converted = clean_data(converted)

    # 保存
    output_path = args.output or args.input.rsplit('.', 1)[0] + '_converted.jsonl'
    save_jsonl(converted, output_path)


if __name__ == '__main__':
    main()
