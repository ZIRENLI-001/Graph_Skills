# 方案迭代与实验管理

## 1. 迭代优先级金字塔

```
收益从高到低(先做上层, 再做下层):

 ████████████████████████████████████████████████████
 █  Level 1: 可信CV                                  █
 █  GM共识: 没有可信CV一切白费                         █
 ████████████████████████████████████████████████████
 █  Level 2: 特征工程                                █
 █  ICLR 2025 TabReD: 特征 > 模型选择               █
 ██████████████████████████████████████████████████
 █  Level 3: 数据清洗/增强/伪标签              █
 █  BirdCLEF/Jigsaw验证有效                  █
 ████████████████████████████████████████████
 █  Level 4: 单模型超参优化              █
 █  Optuna 50→200 trials              █
 ██████████████████████████████████████
 █  Level 5: 多模型训练            █
 █  seed avg + 不同模型          █
 ████████████████████████████████
 █  Level 6: 后处理          █
 █  阈值/裁剪/排序         █
 ██████████████████████████
 █  Level 7: Ensemble  █
 █  Stacking/Blend   █
 ████████████████████
```

### 每层预期收益估计

| Level | 预期CV提升 | 时间投入 | 风险 |
|-------|-----------|---------|------|
| CV建立 | 基础(无此一切无意义) | 2-4小时 | 低 |
| 特征工程 | +2-5% | 数天 | 低 |
| 数据清洗/增强 | +1-3% | 1-2天 | 低 |
| 超参优化 | +0.5-2% | 数小时(Optuna) | 低 |
| 多模型 | +0.5-1% | 数小时 | 低 |
| 后处理 | +0.1-1% | 数小时 | 中(可能过拟合) |
| Ensemble | +0.5-2% | 1-2天 | 中(复杂度增加) |

---

## 2. 实验管理

### 方案对比

| 工具 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| **W&B** | 可视化强, 团队协作, 免费额度 | 需联网, 学习成本 | 团队赛, 深度学习 |
| **MLflow** | 开源, 自托管, 模型管理 | 搭建麻烦 | 本地实验 |
| **CSV日志** | 最简单, 无依赖 | 无可视化 | 快速原型, Notebook赛 |
| **Optuna Dashboard** | 超参搜索可视化 | 仅超参 | 配合Optuna使用 |

### CSV 实验日志模板

```
experiment_id, datetime, method, cv_score, lb_score, notes
001, 2025-01-15, lgb_baseline_5fold, 0.8523, 0.8501, "baseline"
002, 2025-01-15, lgb+groupby_feats, 0.8567, 0.8545, "+groupby聚合特征"
003, 2025-01-16, lgb+xgb_ensemble, 0.8589, 0.8560, "+XGBoost ensemble"
004, 2025-01-16, lgb+xgb+cat_stack, 0.8612, 0.8582, "+CatBoost stacking"
```

---

## 3. CV 策略完整指南

### 策略选择决策树

```
数据是否有时间维度?
├── 是 → 数据是否有分组?
│   ├── 是 → GroupTimeSeriesSplit
│   └── 否 → TimeSeriesSplit (加gap)
└── 否 → 数据是否有自然分组?
    ├── 是 → GroupKFold
    └── 否 → 是分类任务?
        ├── 是 → StratifiedKFold(5)
        └── 否 → KFold(5) 或 分箱后StratifiedKFold
```

### 各策略实现

#### StratifiedKFold（最常用）

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

#### GroupKFold（有分组时）

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    # 同组数据不会跨fold
    pass
```

#### TimeSeriesSplit（时序数据）

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=30)  # gap=30天防止lag泄漏
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    pass
```

#### LLM 赛特殊验证集

```python
# AIMO系列: 使用固定验证集而非CV
val_datasets = {
    'AMC': load_amc_problems(),     # AMC 10/12
    'AIME': load_aime_problems(),   # AIME
    'MATH': load_math_problems(),   # MATH benchmark
}

for name, val_data in val_datasets.items():
    accuracy = evaluate(model, val_data)
    print(f'{name}: {accuracy:.2%}')
```

### CV 关键原则

1. **所有模型使用相同的fold划分** — 保证OOF可比较
2. **CV策略必须匹配test分布** — 如test是未来数据, 用时序CV
3. **时序CV必须加gap** — 防止lag特征泄漏
4. **GroupKFold确保同组不跨fold** — 如同一用户/患者/站点
5. **信任CV而非LB** — GM核心原则

---

## 4. 超参调优

### Optuna LightGBM 模板

```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
print(f'Best CV: {study.best_value:.5f}')
print(f'Best params: {study.best_params}')
```

### 调优策略

```
Phase 1: 粗搜 (50 trials)
  └─ 大范围搜索空间
  └─ 找到大致最优区间

Phase 2: 精搜 (200 trials)
  └─ 缩小到Phase 1最优区间附近
  └─ 更细粒度搜索

Phase 3: 种子平均
  └─ 用最优超参, 多个seed训练
  └─ 平均预测 → 更稳定
```

---

## 5. Ensemble 迭代策略

### Phase 1: Seed Averaging（最简单）

```python
# 同模型, 不同seed, 平均预测
predictions = []
for seed in [42, 123, 456, 789, 2024]:
    model = train_model(params, seed=seed)
    pred = model.predict(test)
    predictions.append(pred)
final_pred = np.mean(predictions, axis=0)
```

### Phase 2: 多模型训练

```python
# 不同类型模型
models = {
    'lgb': train_lgb(X, y),
    'xgb': train_xgb(X, y),
    'cat': train_cat(X, y),
    'nn': train_nn(X, y),
    'tabpfn': train_tabpfn(X, y),
}
```

### Phase 3: Hill Climbing 选子集

```python
def hill_climbing_ensemble(oof_preds_dict, y_true, metric_func):
    """贪心搜索最优模型子集和权重"""
    model_names = list(oof_preds_dict.keys())
    best_score = -np.inf
    selected = []
    weights = []

    while True:
        improved = False
        for name in model_names:
            if name in selected:
                continue
            # 尝试加入这个模型
            trial = selected + [name]
            trial_pred = np.mean([oof_preds_dict[n] for n in trial], axis=0)
            score = metric_func(y_true, trial_pred)
            if score > best_score:
                best_score = score
                best_name = name
                improved = True
        if not improved:
            break
        selected.append(best_name)
        print(f'Added {best_name}, score={best_score:.5f}')

    return selected
```

### Phase 4: 权重优化

```python
from scipy.optimize import minimize

def optimize_weights(oof_preds_list, y_true, metric_func):
    """用scipy优化ensemble权重"""
    n_models = len(oof_preds_list)

    def objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化
        pred = sum(w * p for w, p in zip(weights, oof_preds_list))
        return -metric_func(y_true, pred)  # 最小化负分数

    initial_weights = np.ones(n_models) / n_models
    bounds = [(0, 1)] * n_models
    result = minimize(objective, initial_weights, method='Nelder-Mead', bounds=bounds)

    optimal_weights = result.x / result.x.sum()
    return optimal_weights
```

---

## 6. 伪标签策略

### 基本流程

```python
def pseudo_labeling(model, train, test, target, threshold=0.95):
    """软伪标签策略"""
    # Step 1: 训练初始模型
    model.fit(train[features], train[target])

    # Step 2: 预测test
    test_probs = model.predict_proba(test[features])

    # Step 3: 选择高置信度样本
    confident_mask = (test_probs.max(axis=1) > threshold)
    pseudo_data = test[confident_mask].copy()
    pseudo_data[target] = test_probs[confident_mask].argmax(axis=1)

    # Step 4: 合并训练
    combined = pd.concat([train, pseudo_data])
    model.fit(combined[features], combined[target])

    return model
```

### 注意事项

- 使用**软标签**(概率) > 硬标签(类别) — 更多信息, 更少噪声
- K-fold伪标签: 对每个fold分别生成, 避免泄漏
- 迭代伪标签: 多轮, 每轮提高threshold
- BirdCLEF 2024 3rd 和 Jigsaw 2025 验证有效

---

## 7. CV-LB 一致性追踪

### 追踪方法

```python
import matplotlib.pyplot as plt

# 记录每次实验
experiments = pd.DataFrame({
    'cv': [0.852, 0.857, 0.859, 0.861, 0.865],
    'lb': [0.850, 0.854, 0.856, 0.858, 0.860],
    'method': ['baseline', '+groupby', '+xgb', '+stack', '+post'],
})

# 画CV vs LB散点图
plt.scatter(experiments['cv'], experiments['lb'])
for i, row in experiments.iterrows():
    plt.annotate(row['method'], (row['cv'], row['lb']))
plt.xlabel('CV Score')
plt.ylabel('LB Score')
plt.title('CV vs LB Consistency')

# 检查: CV和LB应该有较强的线性关系
correlation = experiments[['cv', 'lb']].corr().iloc[0, 1]
print(f'CV-LB correlation: {correlation:.4f}')
# > 0.9 → 一致性好, 可信任CV
# < 0.7 → CV策略可能有问题
```

### 不一致原因排查

| 原因 | 表现 | 解决方案 |
|------|------|---------|
| CV策略不匹配 | CV高LB低 | 改用匹配test分布的CV |
| 过拟合Public LB | LB高CV低 | 停止追LB, 信任CV |
| 数据分布漂移 | 两者都不稳 | Adversarial Validation |
| 样本太少 | CV方差大 | RepeatedKFold |
| 时序泄漏 | CV异常高 | 加gap, 检查lag特征 |
| 分组泄漏 | CV异常高 | 改用GroupKFold |
| 随机种子 | CV波动大 | 多seed平均 |
| 评估指标实现 | 细微差异 | 检查自定义metric是否与官方一致 |
