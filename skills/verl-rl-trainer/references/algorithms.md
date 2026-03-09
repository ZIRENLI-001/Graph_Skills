# verl 支持的 RL 算法详解

## 算法总览

| 算法 | `adv_estimator` 值 | 需要 Critic | 核心思想 |
|------|-------------------|-------------|---------|
| PPO | `gae` | 是 | GAE 优势估计 + 裁剪策略更新 |
| GRPO | `grpo` | 否 | 组内相对排序作为基线 |
| REINFORCE++ | `reinforce_plus_plus` | 否 | 改进的 REINFORCE with baseline |
| RLOO | `rloo` / `rloo_vectorized` | 否 | Leave-One-Out 基线估计 |
| DAPO | 基于 GRPO 扩展 | 否 | 动态采样 + 自适应 KL |
| DrGRPO | 基于 GRPO 扩展 | 否 | 带正则化的 GRPO |
| ReMax | `remax` | 否 | 基于最大奖励的基线 |
| PRIME | `prime` | 是 | 过程奖励引导 |

## PPO（Proximal Policy Optimization）

经典 RLHF 算法，使用 Critic 模型估计价值函数，通过 GAE 计算优势。

### 核心参数

```yaml
algorithm:
  adv_estimator: gae        # 使用 GAE 优势估计
  gamma: 1.0                # 折扣因子
  lam: 1.0                  # GAE lambda
  kl_ctrl:
    type: fixed              # fixed 或 adaptive
    kl_coef: 0.001           # KL 惩罚系数

actor_rollout_ref.actor:
  clip_ratio: 0.2            # PPO 裁剪范围
  ppo_epochs: 1              # 每次 rollout 的更新轮数
  use_kl_loss: False         # PPO 通常不在 loss 中加 KL

critic:
  model.path: ...            # Critic 模型路径（必须）
  ppo_epochs: 1
  cliprange_value: 0.5       # 值函数裁剪范围
```

### 适用场景
- 有训练好的 Reward Model
- 通用对齐任务（helpful、harmless、honest）
- 需要精细控制策略更新幅度

### 显存考虑
PPO 需要加载 4 个模型：Actor、Critic、Reference Model、Reward Model。显存需求约为 GRPO 的 2 倍。

## GRPO（Group Relative Policy Optimization）

DeepSeek 提出，对每个 prompt 生成多个回答，用组内相对排名替代 Critic。

### 核心参数

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  rollout:
    n: 5                     # 每个 prompt 生成的回答数（关键参数）
    temperature: 1.0          # 采样温度
  actor:
    use_kl_loss: True         # GRPO 建议开启 KL loss
    kl_loss_coef: 0.001       # KL 系数
    ppo_epochs: 1
```

### 适用场景
- 数学推理（GSM8K、MATH）
- 代码生成（有测试用例）
- 任何有可验证奖励的任务
- 资源有限（无需 Critic 模型）

### 关键设计
- `n` 越大，基线估计越准，但计算开销越大。推荐 5-16
- 组内奖励标准化：advantage = (reward - mean) / std

## REINFORCE++

改进的 REINFORCE 算法，使用组内均值作为基线，但不做标准化。

```yaml
algorithm:
  adv_estimator: reinforce_plus_plus
```

### 与 GRPO 区别
- GRPO: advantage = (r - mean) / std（标准化）
- REINFORCE++: advantage = r - mean（仅减均值）

## RLOO（REINFORCE Leave-One-Out）

使用 leave-one-out 策略估计基线：每个样本的基线 = 同组其他样本的平均奖励。

```yaml
algorithm:
  adv_estimator: rloo            # 标准版
  # 或
  adv_estimator: rloo_vectorized  # 向量化加速版
```

## DAPO（Dynamic Allocation Policy Optimization）

GRPO 的改进版本，核心改进：
- **动态采样**：根据奖励信号动态调整每个 prompt 的采样数量
- **自适应 KL**：根据训练阶段自动调整 KL 系数
- **过滤机制**：过滤掉全对或全错的 prompt 组，提升训练效率

## 算法选择决策树

```
需要 RL 训练
├─ 有 Reward Model？
│   ├─ 是 → PPO（经典、稳定）
│   └─ 否 → 有可验证奖励？
│       ├─ 是 → GRPO（推荐）或 DAPO（探索更好）
│       └─ 否 → 需要先训练 RM，再用 PPO
├─ 资源受限？
│   ├─ 是 → GRPO（无需 Critic，省显存）
│   └─ 否 → PPO（更通用）
└─ 需要强探索能力？
    ├─ 是 → DAPO 或增大 GRPO 的 n
    └─ 否 → GRPO 或 REINFORCE++
```

## 奖励类型

### 1. 基于函数的可验证奖励（Rule-based）

适用于有明确答案的任务（数学、代码）：

```python
def math_reward(solution_str, ground_truth):
    """数学题奖励函数"""
    predicted = extract_answer(solution_str)
    return 1.0 if predicted == ground_truth else 0.0
```

### 2. 基于模型的奖励（Reward Model）

适用于主观评估任务（对话质量、文风）：

```yaml
reward_model:
  enable: True
  model.path: reward-model-path
  micro_batch_size: 2
```

### 3. 混合奖励

同时使用规则奖励和模型奖励：

```python
final_reward = alpha * rule_reward + (1 - alpha) * model_reward
```
