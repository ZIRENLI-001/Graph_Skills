#!/usr/bin/env python3
"""
ms-swift 数据集格式验证工具

在训练前验证数据集格式是否符合 ms-swift 要求。
超过 50% 的训练失败源于数据格式问题，请务必在训练前运行此脚本。

支持的格式检测:
  - messages 格式（推荐）
  - shareGPT/conversations 格式
  - alpaca 格式
  - query/response 格式

用法:
  python dataset_validator.py --dataset_path /path/to/data.jsonl --task sft
  python dataset_validator.py --dataset_path /path/to/data.jsonl --task dpo
  python dataset_validator.py --dataset_path /path/to/data.jsonl --task grpo
"""

import argparse
import json
import csv
import os
import sys
from collections import Counter
from pathlib import Path


def load_dataset(path: str, max_samples: int = 1000) -> list[dict]:
    """加载数据集，支持 jsonl、json、csv 格式。"""
    path = Path(path)
    suffix = path.suffix.lower()
    samples = []

    if suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    elif suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data[:max_samples]
            else:
                samples = [data]
    elif suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break
                samples.append(dict(row))
    else:
        print(f"[ERROR] 不支持的文件格式: {suffix}")
        print("支持的格式: .jsonl, .json, .csv")
        sys.exit(1)

    return samples


def count_total_samples(path: str) -> int:
    """统计数据集总样本数。"""
    path = Path(path)
    suffix = path.suffix.lower()
    count = 0

    if suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
    elif suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            count = len(data) if isinstance(data, list) else 1
    elif suffix == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)

    return count


def detect_format(samples: list[dict]) -> str:
    """自动检测数据集格式。"""
    if not samples:
        return 'unknown'

    first = samples[0]
    keys = set(first.keys())

    # messages 格式
    if 'messages' in keys:
        return 'messages'

    # shareGPT/conversations 格式
    if 'conversation' in keys or 'conversations' in keys:
        return 'sharegpt'

    # alpaca 格式
    if 'instruction' in keys and 'output' in keys:
        return 'alpaca'

    # query/response 格式
    if 'query' in keys and ('response' in keys or 'answer' in keys or 'output' in keys):
        return 'query_response'

    # prompt/completion 格式
    if 'prompt' in keys and ('completion' in keys or 'chosen' in keys):
        return 'prompt_completion'

    return 'unknown'


def validate_messages_format(samples: list[dict]) -> list[str]:
    """验证 messages 格式。"""
    issues = []
    valid_roles = {'system', 'user', 'assistant', 'tool'}

    for i, sample in enumerate(samples):
        messages = sample.get('messages', [])
        if not isinstance(messages, list):
            issues.append(f"  样本 {i}: 'messages' 字段不是列表")
            continue
        if len(messages) == 0:
            issues.append(f"  样本 {i}: 'messages' 列表为空")
            continue

        for j, msg in enumerate(messages):
            if not isinstance(msg, dict):
                issues.append(f"  样本 {i}, 消息 {j}: 不是字典类型")
                continue
            if 'role' not in msg:
                issues.append(f"  样本 {i}, 消息 {j}: 缺少 'role' 字段")
            elif msg['role'] not in valid_roles:
                issues.append(f"  样本 {i}, 消息 {j}: role '{msg['role']}' 不在 {valid_roles}")
            if 'content' not in msg:
                issues.append(f"  样本 {i}, 消息 {j}: 缺少 'content' 字段")

        # 检查是否有 assistant 回复
        has_assistant = any(m.get('role') == 'assistant' for m in messages if isinstance(m, dict))
        if not has_assistant:
            issues.append(f"  样本 {i}: messages 中没有 assistant 角色的回复")

    return issues


def validate_sharegpt_format(samples: list[dict]) -> list[str]:
    """验证 shareGPT/conversations 格式。"""
    issues = []

    for i, sample in enumerate(samples):
        conv_key = 'conversation' if 'conversation' in sample else 'conversations'
        conv = sample.get(conv_key, [])
        if not isinstance(conv, list):
            issues.append(f"  样本 {i}: '{conv_key}' 字段不是列表")
            continue
        if len(conv) == 0:
            issues.append(f"  样本 {i}: '{conv_key}' 列表为空")
            continue

        for j, turn in enumerate(conv):
            if not isinstance(turn, dict):
                issues.append(f"  样本 {i}, 轮次 {j}: 不是字典类型")
                continue
            if 'human' not in turn and 'from' not in turn:
                issues.append(f"  样本 {i}, 轮次 {j}: 缺少 'human' 或 'from' 字段")
            if 'assistant' not in turn and 'value' not in turn:
                issues.append(f"  样本 {i}, 轮次 {j}: 缺少 'assistant' 或 'value' 字段")

    return issues


def validate_alpaca_format(samples: list[dict]) -> list[str]:
    """验证 alpaca 格式。"""
    issues = []

    for i, sample in enumerate(samples):
        if 'instruction' not in sample:
            issues.append(f"  样本 {i}: 缺少 'instruction' 字段")
        if 'output' not in sample:
            issues.append(f"  样本 {i}: 缺少 'output' 字段")

    return issues


def validate_query_response_format(samples: list[dict]) -> list[str]:
    """验证 query/response 格式。"""
    issues = []
    response_keys = {'response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution', 'text', 'completion', 'content'}

    for i, sample in enumerate(samples):
        if 'query' not in sample:
            issues.append(f"  样本 {i}: 缺少 'query' 字段")
        has_response = bool(response_keys & set(sample.keys()))
        if not has_response:
            issues.append(f"  样本 {i}: 缺少响应字段 (response/answer/output 等)")

    return issues


def check_dpo_compatibility(samples: list[dict], fmt: str) -> tuple[str, list[str]]:
    """检查 DPO 兼容性。"""
    issues = []
    has_rejected = sum(1 for s in samples if 'rejected_response' in s or 'rejected' in s)

    if has_rejected == 0:
        return 'NEEDS MAPPING', [
            "  缺少 'rejected_response' 字段",
            "  DPO 需要: messages + rejected_response",
            "  建议: 为每条数据添加 'rejected_response' 字段，包含被拒绝的回答"
        ]

    if has_rejected < len(samples):
        return 'NEEDS MAPPING', [
            f"  仅 {has_rejected}/{len(samples)} 条样本有 rejected_response 字段",
            "  DPO 要求所有样本都必须包含 'rejected_response'"
        ]

    return 'READY', []


def check_grpo_compatibility(samples: list[dict], fmt: str) -> tuple[str, list[str]]:
    """检查 GRPO 兼容性。"""
    issues = []

    # GRPO 只需要 prompt（user message），额外字段用于 ORM
    has_user_msg = 0
    has_extra_fields = Counter()

    for s in samples:
        if fmt == 'messages':
            messages = s.get('messages', [])
            if any(m.get('role') == 'user' for m in messages if isinstance(m, dict)):
                has_user_msg += 1
        elif fmt == 'query_response':
            if 'query' in s:
                has_user_msg += 1
        elif fmt == 'alpaca':
            if 'instruction' in s:
                has_user_msg += 1

        # 统计额外字段
        extra_keys = set(s.keys()) - {'messages', 'conversation', 'conversations',
                                       'instruction', 'input', 'output', 'system',
                                       'query', 'response', 'history',
                                       'images', 'videos', 'audios',
                                       'rejected_response', 'label', 'tools', 'objects'}
        for k in extra_keys:
            has_extra_fields[k] += 1

    if has_user_msg == 0:
        return 'INCOMPATIBLE', ["  没有检测到用户消息（user/query/instruction）"]

    if has_user_msg < len(samples):
        issues.append(f"  {has_user_msg}/{len(samples)} 条样本有用户消息")

    if has_extra_fields:
        extra_info = ', '.join(f"'{k}'({v}条)" for k, v in has_extra_fields.most_common(5))
        issues.append(f"  检测到额外字段（将传递给 ORM）: {extra_info}")
    else:
        issues.append("  未检测到额外字段（如 'solution'），ORM 将无法获取标准答案")
        issues.append("  建议: 添加 'solution' 等字段用于奖励计算")

    status = 'READY' if has_user_msg == len(samples) else 'NEEDS MAPPING'
    return status, issues


def compute_statistics(samples: list[dict], fmt: str) -> dict:
    """计算数据集统计信息。"""
    stats = {
        'num_samples': len(samples),
        'fields': Counter(),
        'content_lengths': [],
        'num_turns': [],
    }

    for s in samples:
        for k in s.keys():
            stats['fields'][k] += 1

        if fmt == 'messages':
            messages = s.get('messages', [])
            stats['num_turns'].append(len(messages))
            total_len = sum(len(str(m.get('content', ''))) for m in messages if isinstance(m, dict))
            stats['content_lengths'].append(total_len)
        elif fmt == 'alpaca':
            total_len = len(str(s.get('instruction', ''))) + len(str(s.get('input', ''))) + len(str(s.get('output', '')))
            stats['content_lengths'].append(total_len)
        elif fmt == 'query_response':
            total_len = len(str(s.get('query', ''))) + len(str(s.get('response', s.get('answer', ''))))
            stats['content_lengths'].append(total_len)

    return stats


def print_report(dataset_path: str, total_count: int, samples: list[dict],
                 fmt: str, task: str, format_issues: list[str],
                 stats: dict):
    """打印验证报告。"""
    print("=" * 60)
    print("ms-swift Dataset Validator Report")
    print("=" * 60)
    print(f"Dataset:  {dataset_path}")
    print(f"Samples:  {total_count:,}")
    print(f"Checked:  {len(samples):,} (max preview)")
    print(f"Format:   {fmt}")
    print()

    # 字段分布
    print("Fields distribution:")
    for field, count in stats['fields'].most_common():
        pct = count / len(samples) * 100
        print(f"  {field}: {count} ({pct:.0f}%)")
    print()

    # 内容长度统计
    if stats['content_lengths']:
        lengths = sorted(stats['content_lengths'])
        print("Content length (chars):")
        print(f"  Min: {lengths[0]:,}")
        print(f"  Max: {lengths[-1]:,}")
        print(f"  Median: {lengths[len(lengths)//2]:,}")
        print(f"  Mean: {sum(lengths)//len(lengths):,}")
    print()

    # 轮次统计
    if stats['num_turns']:
        turns = stats['num_turns']
        print(f"Turns per sample: min={min(turns)}, max={max(turns)}, avg={sum(turns)/len(turns):.1f}")
        print()

    # 格式验证
    print("Format validation:")
    if format_issues:
        print(f"  ✗ {len(format_issues)} issue(s) found (showing first 10):")
        for issue in format_issues[:10]:
            print(issue)
        if len(format_issues) > 10:
            print(f"  ... and {len(format_issues) - 10} more")
    else:
        print("  ✓ All samples pass format validation")
    print()

    # 任务兼容性
    print("Compatibility:")

    # SFT
    if fmt in ('messages', 'sharegpt', 'alpaca', 'query_response'):
        sft_status = 'READY' if not format_issues else 'NEEDS FIX'
        symbol = '✓' if sft_status == 'READY' else '✗'
        print(f"  {symbol} SFT    — {sft_status}")
    else:
        print(f"  ✗ SFT    — INCOMPATIBLE (unknown format '{fmt}')")

    # DPO
    dpo_status, dpo_issues = check_dpo_compatibility(samples, fmt)
    symbol = '✓' if dpo_status == 'READY' else '✗'
    print(f"  {symbol} DPO    — {dpo_status}")
    if task == 'dpo':
        for issue in dpo_issues:
            print(issue)

    # GRPO
    grpo_status, grpo_issues = check_grpo_compatibility(samples, fmt)
    symbol = '✓' if grpo_status == 'READY' else '✗'
    print(f"  {symbol} GRPO   — {grpo_status}")
    if task == 'grpo':
        for issue in grpo_issues:
            print(issue)

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='ms-swift 数据集格式验证工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dataset_validator.py --dataset_path data.jsonl --task sft
  python dataset_validator.py --dataset_path data.json --task dpo
  python dataset_validator.py --dataset_path data.csv --task grpo --max_samples 5000
        """
    )
    parser.add_argument('--dataset_path', required=True, help='数据集文件路径 (.jsonl/.json/.csv)')
    parser.add_argument('--task', choices=['sft', 'dpo', 'grpo', 'all'], default='all',
                        help='目标训练任务 (default: all)')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='最大检查样本数 (default: 1000)')

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"[ERROR] 文件不存在: {args.dataset_path}")
        sys.exit(1)

    # 加载数据
    print(f"Loading dataset: {args.dataset_path}")
    samples = load_dataset(args.dataset_path, args.max_samples)
    total_count = count_total_samples(args.dataset_path)

    if not samples:
        print("[ERROR] 数据集为空或无法解析")
        sys.exit(1)

    # 检测格式
    fmt = detect_format(samples)

    # 格式验证
    if fmt == 'messages':
        format_issues = validate_messages_format(samples)
    elif fmt == 'sharegpt':
        format_issues = validate_sharegpt_format(samples)
    elif fmt == 'alpaca':
        format_issues = validate_alpaca_format(samples)
    elif fmt == 'query_response':
        format_issues = validate_query_response_format(samples)
    else:
        format_issues = [f"  无法识别数据格式，检测到的字段: {list(samples[0].keys())}"]

    # 统计
    stats = compute_statistics(samples, fmt)

    # 打印报告
    print_report(args.dataset_path, total_count, samples, fmt, args.task, format_issues, stats)


if __name__ == '__main__':
    main()
