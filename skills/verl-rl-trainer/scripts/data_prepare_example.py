#!/usr/bin/env python3
"""
verl 数据预处理脚本

将常见格式的数据集转换为 verl 所需的 Parquet 格式。
支持 JSONL、JSON、CSV 输入，输出 Parquet 文件。

用法:
    python data_prepare_example.py --input_path data.jsonl --output_dir ~/data/custom --task_type math
    python data_prepare_example.py --input_path data.jsonl --output_dir ~/data/custom --task_type general
    python data_prepare_example.py --input_path data.jsonl --output_dir ~/data/custom --task_type code
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def load_input_data(input_path: str) -> list[dict]:
    """加载输入数据，支持 JSONL、JSON、CSV 格式"""
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == '.jsonl':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    elif suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON 文件应包含一个列表")
    elif suffix == '.csv':
        df = pd.read_csv(path)
        return df.to_dict('records')
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，请使用 .jsonl, .json, .csv")


def convert_to_verl_format(item: dict, task_type: str, system_prompt: str = "") -> dict:
    """
    将单条数据转换为 verl 格式。

    verl 数据格式要求:
    {
        "data_source": str,           # 数据来源标识
        "prompt": [                    # Chat 模板格式
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "ability": str,               # 能力类型标签
        "reward_model": {             # 奖励配置
            "style": "rule",
            "ground_truth": "..."
        },
        "extra_info": {               # 额外信息
            "solution": "...",
            ...
        }
    }
    """
    prompt = []

    # 添加 system prompt
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})

    # 提取用户输入
    user_content = ""
    if "query" in item:
        user_content = item["query"]
    elif "question" in item:
        user_content = item["question"]
    elif "instruction" in item:
        user_content = item["instruction"]
        if item.get("input"):
            user_content += f"\n{item['input']}"
    elif "prompt" in item:
        if isinstance(item["prompt"], str):
            user_content = item["prompt"]
        elif isinstance(item["prompt"], list):
            # 已经是 chat 格式
            return _build_record_from_chat(item, task_type)
    elif "messages" in item:
        return _build_record_from_messages(item, task_type)
    else:
        raise ValueError(f"无法识别用户输入字段，可用字段: {list(item.keys())}")

    prompt.append({"role": "user", "content": user_content})

    # 提取答案/解决方案
    solution = ""
    if "answer" in item:
        solution = str(item["answer"])
    elif "solution" in item:
        solution = str(item["solution"])
    elif "response" in item:
        solution = str(item["response"])
    elif "output" in item:
        solution = str(item["output"])

    # 构建记录
    record = {
        "data_source": item.get("data_source", f"custom_{task_type}"),
        "prompt": prompt,
        "ability": task_type,
        "reward_model": {
            "style": "rule",
            "ground_truth": solution,
        },
        "extra_info": {
            "solution": solution,
        },
    }

    return record


def _build_record_from_chat(item: dict, task_type: str) -> dict:
    """从 chat 格式构建记录"""
    prompt_messages = item["prompt"]
    if isinstance(prompt_messages, str):
        prompt_messages = json.loads(prompt_messages)

    # 只保留到最后一个 user 消息作为 prompt
    user_messages = []
    for msg in prompt_messages:
        if msg["role"] in ("system", "user"):
            user_messages.append(msg)

    solution = item.get("answer", item.get("solution", item.get("response", "")))

    return {
        "data_source": item.get("data_source", f"custom_{task_type}"),
        "prompt": user_messages,
        "ability": task_type,
        "reward_model": {
            "style": "rule",
            "ground_truth": str(solution),
        },
        "extra_info": {
            "solution": str(solution),
        },
    }


def _build_record_from_messages(item: dict, task_type: str) -> dict:
    """从 messages 格式构建记录"""
    messages = item["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)

    # 提取 prompt（system + user 消息）
    prompt = []
    solution = ""
    for msg in messages:
        if msg["role"] in ("system", "user"):
            prompt.append(msg)
        elif msg["role"] == "assistant":
            solution = msg["content"]

    # 优先使用显式的 solution/answer 字段
    if "solution" in item:
        solution = str(item["solution"])
    elif "answer" in item:
        solution = str(item["answer"])

    return {
        "data_source": item.get("data_source", f"custom_{task_type}"),
        "prompt": prompt,
        "ability": task_type,
        "reward_model": {
            "style": "rule",
            "ground_truth": solution,
        },
        "extra_info": {
            "solution": solution,
        },
    }


def get_default_system_prompt(task_type: str) -> str:
    """获取任务类型对应的默认 system prompt"""
    prompts = {
        "math": "Please reason step by step, and put your final answer within \\boxed{}.",
        "code": "You are a helpful coding assistant. Write clean, correct code.",
        "general": "",
    }
    return prompts.get(task_type, "")


def main():
    parser = argparse.ArgumentParser(description="verl 数据预处理工具")
    parser.add_argument("--input_path", type=str, required=True,
                        help="输入数据路径 (.jsonl, .json, .csv)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--task_type", type=str, default="math",
                        choices=["math", "code", "general"],
                        help="任务类型 (default: math)")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="自定义 system prompt（不指定则使用默认值）")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="训练集比例 (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数（用于快速测试）")

    args = parser.parse_args()

    # 加载数据
    print(f"Loading data from: {args.input_path}")
    raw_data = load_input_data(args.input_path)
    print(f"Loaded {len(raw_data)} samples")

    if args.max_samples and len(raw_data) > args.max_samples:
        import random
        random.seed(args.seed)
        raw_data = random.sample(raw_data, args.max_samples)
        print(f"Sampled {args.max_samples} samples")

    # 转换格式
    system_prompt = args.system_prompt or get_default_system_prompt(args.task_type)
    converted = []
    errors = 0
    for i, item in enumerate(raw_data):
        try:
            record = convert_to_verl_format(item, args.task_type, system_prompt)
            converted.append(record)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: Failed to convert sample {i}: {e}")

    print(f"Converted {len(converted)} samples ({errors} errors)")

    if not converted:
        print("ERROR: No samples converted. Check your data format.")
        sys.exit(1)

    # 序列化嵌套字段
    for record in converted:
        record["prompt"] = json.dumps(record["prompt"], ensure_ascii=False)
        record["reward_model"] = json.dumps(record["reward_model"], ensure_ascii=False)
        record["extra_info"] = json.dumps(record["extra_info"], ensure_ascii=False)

    # 分割数据集
    df = pd.DataFrame(converted)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    split_idx = int(len(df) * args.train_ratio)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nOutput:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Test:  {test_path} ({len(test_df)} samples)")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample record:")
    sample = df.iloc[0].to_dict()
    for k, v in sample.items():
        v_str = str(v)
        if len(v_str) > 100:
            v_str = v_str[:100] + "..."
        print(f"  {k}: {v_str}")


if __name__ == "__main__":
    main()
