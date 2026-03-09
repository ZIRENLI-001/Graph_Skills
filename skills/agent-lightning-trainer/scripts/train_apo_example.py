#!/usr/bin/env python3
"""
Agent Lightning APO（自动提示优化）训练示例

通过文本梯度迭代优化 Agent 的提示词模板，无需 GPU。

场景示例：会议室预订 Agent
- 输入：会议需求（人数、设备、时间）
- 输出：推荐的会议室
- 奖励：推荐是否正确

用法:
    python train_apo_example.py
    python train_apo_example.py --api_base https://api.openai.com/v1 --api_key sk-xxx
    python train_apo_example.py --n_rounds 5 --n_runners 4
"""

import argparse
import json
import os
import sys


def create_sample_data():
    """创建示例训练/验证数据"""
    train_tasks = [
        {
            "request": "需要一个能容纳10人的会议室，需要投影仪，下午2点到4点",
            "expected_room": "大会议室A",
        },
        {
            "request": "2人小会，只需要白板，上午10点半小时",
            "expected_room": "小会议室C",
        },
        {
            "request": "20人培训，需要投影仪和音响，全天",
            "expected_room": "培训室",
        },
        {
            "request": "4人头脑风暴，需要白板和便利贴，下午3点1小时",
            "expected_room": "创意空间B",
        },
        {
            "request": "视频会议，3人参加，需要摄像头和麦克风",
            "expected_room": "视频会议室D",
        },
    ]

    val_tasks = [
        {
            "request": "8人项目评审，需要投影仪，上午9点到11点",
            "expected_room": "大会议室A",
        },
        {
            "request": "1对1面谈，30分钟",
            "expected_room": "小会议室C",
        },
    ]

    return train_tasks, val_tasks


def main():
    parser = argparse.ArgumentParser(description="Agent Lightning APO 训练示例")
    parser.add_argument("--api_base", type=str,
                        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                        help="LLM API base URL")
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""),
                        help="LLM API key")
    parser.add_argument("--n_rounds", type=int, default=3,
                        help="优化轮数 (default: 3)")
    parser.add_argument("--n_runners", type=int, default=4,
                        help="并发 Runner 数 (default: 4)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="自定义数据路径 (JSON 格式)")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅 dry-run 验证")

    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: 需要设置 API key")
        print("  方式一: export OPENAI_API_KEY=sk-xxx")
        print("  方式二: --api_key sk-xxx")
        sys.exit(1)

    # 加载数据
    if args.data_path:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        train_tasks = data.get("train", data.get("train_tasks", []))
        val_tasks = data.get("val", data.get("val_tasks", []))
    else:
        train_tasks, val_tasks = create_sample_data()

    print("=" * 60)
    print("Agent Lightning APO Training")
    print("=" * 60)
    print(f"API Base: {args.api_base}")
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks: {len(val_tasks)}")
    print(f"Optimization rounds: {args.n_rounds}")
    print()

    try:
        import agentlightning as agl
        from agentlightning.algorithm import APO
    except ImportError:
        print("ERROR: agentlightning 未安装")
        print("  pip install agentlightning")
        sys.exit(1)

    # 定义 Agent
    @agl.rollout
    def booking_agent(task, prompt_template, llm):
        """会议室预订 Agent"""
        prompt = prompt_template.format(request=task["request"])
        response = llm.chat.completions.create(
            model="main_llm",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        # 奖励：推荐的房间是否正确
        reward = 1.0 if task["expected_room"] in answer else 0.0
        return reward

    # 初始提示模板
    initial_prompt = (
        "你是一个会议室预订助手。根据以下会议需求，推荐最合适的会议室。\n\n"
        "可用会议室：\n"
        "- 大会议室A：可容纳20人，有投影仪\n"
        "- 小会议室C：可容纳4人，有白板\n"
        "- 培训室：可容纳30人，有投影仪和音响\n"
        "- 创意空间B：可容纳8人，有白板和便利贴\n"
        "- 视频会议室D：可容纳6人，有摄像头和麦克风\n\n"
        "会议需求：{request}\n\n"
        "请直接回复推荐的会议室名称。"
    )

    # 配置 APO 算法
    algorithm = APO(
        llm_proxy_base_url=args.api_base,
        llm_proxy_api_key=args.api_key,
        initial_prompt_template=initial_prompt,
        n_optimization_rounds=args.n_rounds,
    )

    # 创建 Trainer
    trainer = agl.Trainer(
        agent=booking_agent,
        algorithm=algorithm,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
    )

    if args.dry_run:
        print("Running dry-run (dev mode)...")
        trainer.dev()
        print("\nDry-run completed successfully!")
    else:
        print("Starting APO training...")
        trainer.fit()
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
