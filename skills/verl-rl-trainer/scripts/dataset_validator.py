#!/usr/bin/env python3
"""
verl 数据集验证脚本

验证 Parquet 数据文件是否符合 verl 训练要求。

用法:
    python dataset_validator.py --dataset_path ~/data/gsm8k/train.parquet
    python dataset_validator.py --dataset_path ~/data/custom/train.parquet --verbose
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def validate_dataset(dataset_path: str, verbose: bool = False) -> bool:
    """验证数据集格式"""
    path = Path(dataset_path)

    # 检查文件存在
    if not path.exists():
        print(f"ERROR: File not found: {dataset_path}")
        return False

    # 检查格式
    if path.suffix != '.parquet':
        print(f"WARNING: verl requires Parquet format, got: {path.suffix}")
        print(f"  Use scripts/data_prepare_example.py to convert your data.")
        return False

    # 读取数据
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"ERROR: Failed to read Parquet file: {e}")
        return False

    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print()

    issues = []
    warnings = []

    # 检查必需字段: prompt
    if 'prompt' not in df.columns:
        issues.append("Missing required column: 'prompt'")
    else:
        # 验证 prompt 格式
        sample_prompt = df['prompt'].iloc[0]
        try:
            if isinstance(sample_prompt, str):
                parsed = json.loads(sample_prompt)
            elif isinstance(sample_prompt, list):
                parsed = sample_prompt
            else:
                issues.append(f"Invalid prompt type: {type(sample_prompt)}")
                parsed = None

            if parsed is not None:
                if not isinstance(parsed, list):
                    issues.append("prompt should be a list of message dicts")
                elif len(parsed) == 0:
                    issues.append("prompt list is empty")
                else:
                    # 检查消息格式
                    for msg in parsed:
                        if not isinstance(msg, dict):
                            issues.append(f"prompt message should be dict, got: {type(msg)}")
                            break
                        if 'role' not in msg:
                            issues.append("prompt message missing 'role' field")
                            break
                        if 'content' not in msg:
                            issues.append("prompt message missing 'content' field")
                            break
                        if msg['role'] not in ('system', 'user', 'assistant'):
                            warnings.append(f"Unusual role in prompt: '{msg['role']}'")

                    # 检查是否有 user 消息
                    roles = [msg.get('role') for msg in parsed if isinstance(msg, dict)]
                    if 'user' not in roles:
                        issues.append("prompt must contain at least one 'user' message")

                    if verbose:
                        print(f"Sample prompt (parsed):")
                        for msg in parsed[:3]:
                            content_preview = str(msg.get('content', ''))[:80]
                            print(f"  [{msg.get('role')}]: {content_preview}")
                        print()

        except (json.JSONDecodeError, TypeError) as e:
            issues.append(f"Failed to parse prompt: {e}")

    # 检查推荐字段
    recommended_fields = {
        'data_source': '数据来源标识',
        'ability': '能力类型标签',
        'reward_model': '奖励模型配置',
        'extra_info': '额外信息（含 solution）',
    }

    for field, desc in recommended_fields.items():
        if field not in df.columns:
            warnings.append(f"Missing recommended column: '{field}' ({desc})")

    # 检查 reward_model 字段
    if 'reward_model' in df.columns:
        sample_rm = df['reward_model'].iloc[0]
        try:
            if isinstance(sample_rm, str):
                rm_parsed = json.loads(sample_rm)
            elif isinstance(sample_rm, dict):
                rm_parsed = sample_rm
            else:
                rm_parsed = None
                warnings.append(f"reward_model type unexpected: {type(sample_rm)}")

            if rm_parsed:
                if 'ground_truth' not in rm_parsed and 'style' not in rm_parsed:
                    warnings.append("reward_model missing 'ground_truth' or 'style' field")
                if verbose:
                    print(f"Sample reward_model: {rm_parsed}")
                    print()
        except (json.JSONDecodeError, TypeError):
            warnings.append("Failed to parse reward_model field")

    # 检查 extra_info 中的 solution
    if 'extra_info' in df.columns:
        sample_ei = df['extra_info'].iloc[0]
        try:
            if isinstance(sample_ei, str):
                ei_parsed = json.loads(sample_ei)
            elif isinstance(sample_ei, dict):
                ei_parsed = sample_ei
            else:
                ei_parsed = None

            if ei_parsed and 'solution' not in ei_parsed:
                warnings.append("extra_info missing 'solution' field (needed for verifiable rewards)")
        except (json.JSONDecodeError, TypeError):
            warnings.append("Failed to parse extra_info field")

    # 检查空值
    null_counts = df.isnull().sum()
    for col in df.columns:
        if null_counts[col] > 0:
            pct = null_counts[col] / len(df) * 100
            if pct > 10:
                issues.append(f"Column '{col}' has {pct:.1f}% null values")
            else:
                warnings.append(f"Column '{col}' has {null_counts[col]} null values ({pct:.1f}%)")

    # 检查 prompt 长度分布
    if 'prompt' in df.columns:
        prompt_lengths = df['prompt'].apply(lambda x: len(str(x)))
        if verbose:
            print(f"Prompt length stats (characters):")
            print(f"  Min: {prompt_lengths.min()}")
            print(f"  Max: {prompt_lengths.max()}")
            print(f"  Mean: {prompt_lengths.mean():.0f}")
            print(f"  Median: {prompt_lengths.median():.0f}")
            print()

    # 输出结果
    print("=" * 50)
    if not issues:
        print("RESULT: PASS")
        print()
        print("Compatibility:")
        print("  ✓ GRPO  — READY")
        if 'reward_model' in df.columns or 'extra_info' in df.columns:
            print("  ✓ PPO   — READY (with reward model)")
        else:
            print("  ? PPO   — Needs reward model configuration")
    else:
        print("RESULT: FAIL")
        print()
        print("Issues (must fix):")
        for issue in issues:
            print(f"  ✗ {issue}")

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  ! {warning}")

    print()
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="verl 数据集验证工具")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Parquet 数据文件路径")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细信息")

    args = parser.parse_args()
    success = validate_dataset(args.dataset_path, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
