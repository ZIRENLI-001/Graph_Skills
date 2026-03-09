#!/usr/bin/env python3
"""
Agent Lightning 多 Agent 选择性优化示例

演示如何在多 Agent 系统中只训练特定 Agent，冻结其余 Agent。

场景示例：SQL 查询 Agent 系统
- planner_agent：分析用户问题，制定查询计划（冻结）
- sql_writer_agent：编写 SQL 查询（训练目标）
- validator_agent：验证 SQL 语法（冻结）

用法:
    python train_multiagent_example.py
    python train_multiagent_example.py --agent_match "sql_writer" --dry_run
"""

import argparse
import json
import os
import sys


def create_sample_sql_data():
    """创建示例 SQL 训练数据"""
    train_tasks = [
        {
            "question": "查找所有价格大于100的产品",
            "schema": "products(id, name, price, category)",
            "expected_sql": "SELECT * FROM products WHERE price > 100",
        },
        {
            "question": "统计每个类别的产品数量",
            "schema": "products(id, name, price, category)",
            "expected_sql": "SELECT category, COUNT(*) FROM products GROUP BY category",
        },
        {
            "question": "找出最贵的5个产品",
            "schema": "products(id, name, price, category)",
            "expected_sql": "SELECT * FROM products ORDER BY price DESC LIMIT 5",
        },
        {
            "question": "查找名字包含'手机'的产品",
            "schema": "products(id, name, price, category)",
            "expected_sql": "SELECT * FROM products WHERE name LIKE '%手机%'",
        },
    ]

    val_tasks = [
        {
            "question": "计算所有产品的平均价格",
            "schema": "products(id, name, price, category)",
            "expected_sql": "SELECT AVG(price) FROM products",
        },
    ]

    return train_tasks, val_tasks


def normalize_sql(sql):
    """简单的 SQL 标准化"""
    import re
    sql = sql.strip().rstrip(';')
    sql = re.sub(r'\s+', ' ', sql)
    return sql.upper()


def main():
    parser = argparse.ArgumentParser(description="Agent Lightning 多 Agent 选择性优化示例")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="模型路径")
    parser.add_argument("--agent_match", type=str, default="sql_writer",
                        help="正则匹配要优化的 Agent 名称")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="GPU 数量")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅 dry-run 验证")

    args = parser.parse_args()

    train_tasks, val_tasks = create_sample_sql_data()

    print("=" * 60)
    print("Agent Lightning Multi-Agent Selective Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Agent match pattern: '{args.agent_match}'")
    print(f"Train tasks: {len(train_tasks)}")
    print()

    try:
        import agentlightning as agl
        from agentlightning.algorithm import VERL
    except ImportError:
        print("ERROR: agentlightning 未安装")
        print("  pip install agentlightning")
        sys.exit(1)

    # 定义多 Agent 系统
    class SQLAgentSystem(agl.LitAgent):
        def rollout(self, task, resources):
            llm = resources["main_llm"]

            # Agent 1: Planner（冻结）
            plan_messages = [
                {"role": "system", "content": "你是一个查询计划制定者。分析用户问题并制定查询计划。"},
                {"role": "user", "content": f"问题：{task['question']}\n数据库表：{task['schema']}"},
            ]
            plan_response = llm.chat.completions.create(
                model="planner",  # 逻辑名
                messages=plan_messages,
            )
            plan = plan_response.choices[0].message.content

            # Agent 2: SQL Writer（训练目标）
            write_messages = [
                {"role": "system", "content": "你是一个 SQL 编写专家。根据查询计划编写 SQL。只输出 SQL 语句。"},
                {"role": "user", "content": f"查询计划：{plan}\n表结构：{task['schema']}"},
            ]
            write_response = llm.chat.completions.create(
                model="sql_writer",  # 这个 Agent 名称会被 agent_match 匹配
                messages=write_messages,
            )
            sql = write_response.choices[0].message.content

            # Agent 3: Validator（冻结）
            validate_messages = [
                {"role": "system", "content": "验证 SQL 语法是否正确。如果正确回复 VALID，否则回复 INVALID。"},
                {"role": "user", "content": f"SQL: {sql}\n表结构：{task['schema']}"},
            ]
            validate_response = llm.chat.completions.create(
                model="validator",  # 逻辑名
                messages=validate_messages,
            )

            # 计算奖励
            predicted_sql = normalize_sql(sql)
            expected_sql = normalize_sql(task["expected_sql"])
            reward = 1.0 if predicted_sql == expected_sql else 0.0

            return reward

    # VERL 配置 — 注意 agent_match
    verl_config = {
        "algorithm": {
            "adv_estimator": "grpo",
        },
        "data": {
            "train_batch_size": 16,
            "max_prompt_length": 4096,
            "max_response_length": 1024,
        },
        "actor_rollout_ref": {
            "model": {"path": args.model},
            "rollout": {
                "name": "vllm",
                "n": 4,
                "multi_turn": {"format": "hermes"},
                "gpu_memory_utilization": 0.6,
                "agent_match": args.agent_match,  # 关键：选择性优化
            },
            "actor": {
                "ppo_mini_batch_size": 16,
                "lr": 1e-6,
            },
        },
        "trainer": {
            "n_gpus_per_node": args.n_gpus,
        },
    }

    print(f"Only optimizing agents matching: '{args.agent_match}'")
    print("  planner     → FROZEN")
    print("  sql_writer  → TRAINING")
    print("  validator   → FROZEN")
    print()

    algorithm = VERL(config=verl_config)
    trainer = agl.Trainer(
        agent=SQLAgentSystem(),
        algorithm=algorithm,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
    )

    if args.dry_run:
        print("Running dry-run...")
        trainer.dev()
        print("\nDry-run completed!")
    else:
        print("Starting training...")
        trainer.fit()
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
