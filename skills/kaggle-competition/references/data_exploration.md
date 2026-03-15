# 数据梳理与特征工程

## 1. EDA 标准流程

```
数据加载(polars/cuDF)
  │
  ├── 1. 数据概览
  │   ├── shape, dtypes, memory usage
  │   ├── missing率 per column
  │   ├── 唯一值数 per column
  │   └── 数值 vs 类别特征分类
  │
  ├── 2. 目标变量分析
  │   ├── 分布图(直方图/箱线图)
  │   ├── 类别平衡(分类任务)
  │   ├── 与时间的关系(是否有漂移)
  │   └── 异常值检查
  │
  ├── 3. 数值特征分析
  │   ├── 分布(直方图, 偏度/峰度)
  │   ├── 相关性矩阵(热图)
  │   ├── 箱线图(检测异常值)
  │   └── 特征间散点图(关键特征对)
  │
  ├── 4. 类别特征分析
  │   ├── 基数(唯一值数)
  │   ├── 频率分布
  │   ├── 与目标的关系(目标均值按类别)
  │   └── 高基数特征标记
  │
  ├── 5. Train-Test分布对比
  │   ├── 各特征分布对比图
  │   ├── Adversarial Validation
  │   └── 标记分布漂移严重的特征
  │
  └── 6. 特殊分析
      ├── 时序特征: 趋势, 季节性, 滞后
      ├── 地理特征: 空间分布
      └── 文本特征: 长度, 词频, 语言
```

## 2. GM 级特征工程完整手册

### 2.1 Groupby 聚合特征（最高价值 ★★★★★）

```python
def create_groupby_features(df, group_cols, agg_cols):
    """按分类键计算聚合统计"""
    features = pd.DataFrame()
    for group_col in group_cols:
        for agg_col in agg_cols:
            group = df.groupby(group_col)[agg_col]
            features[f'{group_col}_{agg_col}_mean'] = group.transform('mean')
            features[f'{group_col}_{agg_col}_std'] = group.transform('std')
            features[f'{group_col}_{agg_col}_count'] = group.transform('count')
            features[f'{group_col}_{agg_col}_min'] = group.transform('min')
            features[f'{group_col}_{agg_col}_max'] = group.transform('max')
            features[f'{group_col}_{agg_col}_q25'] = group.transform(lambda x: x.quantile(0.25))
            features[f'{group_col}_{agg_col}_q75'] = group.transform(lambda x: x.quantile(0.75))
    return features
```

### 2.2 浮点位提取（匿名数据集 ★★★★）

```python
def extract_float_digits(df, col):
    """从float32中提取隐藏的离散编码"""
    df[f'{col}_int'] = (df[col] * 100).astype(int)
    df[f'{col}_decimal'] = (df[col] * 100).astype(int) % 10
    df[f'{col}_decimal2'] = (df[col] * 1000).astype(int) % 10
    return df
```

### 2.3 类别交叉特征（★★★★）

```python
def create_cross_features(df, cat_cols, max_combinations=3):
    """拼接类别列创建交叉特征"""
    from itertools import combinations
    for r in range(2, max_combinations + 1):
        for cols in combinations(cat_cols, r):
            name = '_x_'.join(cols)
            df[name] = df[cols[0]].astype(str)
            for c in cols[1:]:
                df[name] += '_' + df[c].astype(str)
    return df
```

### 2.4 海量特征生成 + 筛选（★★★★）

```python
def mass_feature_generation(df, num_cols, top_k=500):
    """生成10000+候选特征, 用importance筛选"""
    import itertools
    features = pd.DataFrame()

    # 差值和比值
    for c1, c2 in itertools.combinations(num_cols[:30], 2):
        features[f'{c1}_minus_{c2}'] = df[c1] - df[c2]
        features[f'{c1}_div_{c2}'] = df[c1] / (df[c2] + 1e-8)
        features[f'{c1}_mult_{c2}'] = df[c1] * df[c2]

    # 用LightGBM importance筛选
    # ... 训练模型, 取feature_importances_, 保留Top-K
    return features
```

**GPU加速**: 使用 cuDF-pandas 零代码切换GPU, 万级特征生成从小时→分钟

### 2.5 目标编码（CV内防泄漏 ★★★★）

```python
from sklearn.model_selection import KFold

def target_encode_cv(df, col, target, n_splits=5):
    """CV内目标编码, 防止泄漏"""
    encoded = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(df):
        means = df.iloc[tr_idx].groupby(col)[target].mean()
        encoded.iloc[va_idx] = df.iloc[va_idx][col].map(means)

    # 填充缺失值为全局均值
    global_mean = df[target].mean()
    encoded.fillna(global_mean, inplace=True)
    return encoded
```

### 2.6 聚类特征（★★★）

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_cluster_features(df, num_cols, n_clusters_list=[5, 10, 20]):
    """KMeans聚类标签作为新特征"""
    X = StandardScaler().fit_transform(df[num_cols].fillna(0))
    for n in n_clusters_list:
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        df[f'cluster_{n}'] = kmeans.fit_predict(X)
        df[f'cluster_{n}_dist'] = kmeans.transform(X).min(axis=1)
    return df
```

### 2.7 NaN 指示列 + 缺失统计（★★★）

```python
def create_null_features(df, cols):
    """缺失值相关特征"""
    for col in cols:
        df[f'{col}_is_null'] = df[col].isnull().astype(int)
    # 每行缺失值总数
    df['null_count'] = df[cols].isnull().sum(axis=1)
    df['null_ratio'] = df['null_count'] / len(cols)
    return df
```

### 2.8 时序特征（★★★）

```python
def create_time_features(df, date_col, target_col, group_col=None):
    """时序相关特征"""
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    if group_col and target_col:
        # 滞后特征
        for lag in [1, 7, 14, 30]:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        # 滚动统计
        for window in [7, 14, 30]:
            rolling = df.groupby(group_col)[target_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'{target_col}_rolling_mean_{window}'] = rolling
    return df
```

### 2.9 PCA/UMAP 降维特征

```python
from sklearn.decomposition import PCA

def create_dim_reduction_features(df, num_cols, n_components=10):
    """降维特征"""
    X = StandardScaler().fit_transform(df[num_cols].fillna(0))
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(X)
    for i in range(n_components):
        df[f'pca_{i}'] = pca_features[:, i]
    return df
```

---

## 3. 缺失值处理策略

### 决策树

```
缺失模式分析
├── 完全随机缺失 (MCAR)
│   ├── 缺失率 < 5% → 中位数/均值填充
│   ├── 缺失率 5-30% → KNN填充 或 模型填充
│   └── 缺失率 > 30% → 考虑删除列 或 NaN指示列+填充
├── 非随机缺失 (MNAR)
│   ├── NaN指示列(缺失本身是信息) + 中位数填充
│   └── 或填充为特殊值(-999)
└── GBDT天然处理缺失
    └── LightGBM/XGBoost可直接使用含NaN的数据
```

---

## 4. 异常值处理

### 方法对比

| 方法 | 适用场景 | 代码 |
|------|---------|------|
| IQR | 正态/近正态分布 | `Q1-1.5*IQR < x < Q3+1.5*IQR` |
| Z-score | 正态分布 | `abs(zscore) < 3` |
| Isolation Forest | 高维数据 | `IsolationForest().fit_predict()` |
| 分位数裁剪 | 通用 | `df[col].clip(q01, q99)` |

### GBDT 对异常值鲁棒

LightGBM/XGBoost 基于树的分裂, 对异常值天然鲁棒。通常不需要专门处理异常值, 除非异常值本身是数据错误。

---

## 5. 数据泄漏检测

### 5.1 目标泄漏

**定义**: 特征包含预测时不可用的信息

**检测方法**:
```python
# 单特征AUC检测
from sklearn.metrics import roc_auc_score

for col in features:
    if df[col].nunique() > 1:
        auc = roc_auc_score(df['target'], df[col].fillna(0))
        if auc > 0.9 or auc < 0.1:  # 异常高AUC
            print(f'WARNING: {col} AUC={auc:.4f} - 可能泄漏!')
```

**案例**: "当月实际使用量"用来预测"当月是否超标" — 在预测时不可用

### 5.2 训练-测试污染

**常见错误**:
```python
# 错误: Scaler在全量数据上fit
scaler = StandardScaler().fit(pd.concat([train, test]))

# 正确: Scaler只在训练数据上fit
scaler = StandardScaler().fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
```

### 5.3 外部数据/模型泄漏

**案例**: AIMO2中DeepSeek R1被加入白名单后, LB格局巨变
- 原因: 新模型可能在竞赛类似的数据上训练过
- 启示: 关注Discussion中的规则变化和模型更新

### 5.4 Adversarial Validation

```python
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

# 训练分类器区分train/test
combined = pd.concat([train[features], test[features]])
labels = [0]*len(train) + [1]*len(test)

clf = LGBMClassifier(n_estimators=100)
auc = cross_val_score(clf, combined, labels, cv=5, scoring='roc_auc').mean()
print(f'Adversarial AUC: {auc:.4f}')
# AUC ≈ 0.5 → 分布一致
# AUC >> 0.5 → 分布不一致, 需关注

# 找出差异最大的特征
clf.fit(combined, labels)
importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print(importances.head(10))  # 这些特征在train/test间差异最大
```

---

## 6. GPU 加速数据处理

### 6.1 cuDF-pandas（零代码切换）

```python
# 只需一行, pandas操作自动GPU加速
%load_ext cudf.pandas
import pandas as pd  # 自动使用GPU加速版

# 之后所有pandas操作自动在GPU上执行
# groupby, merge, value_counts等都会GPU加速
```

**效果**: Deotte用此方案, 10000+特征生成从小时级→分钟级

### 6.2 Polars（快速数据处理）

```python
import polars as pl

# 懒加载 — 不立即读入内存
df = pl.scan_parquet('large_data.parquet')

# 链式操作
result = (
    df
    .filter(pl.col('value') > 0)
    .group_by('category')
    .agg([
        pl.col('value').mean().alias('mean_value'),
        pl.col('value').std().alias('std_value'),
        pl.col('value').count().alias('count'),
    ])
    .collect()  # 最终才执行
)
```

**趋势**: 2023→2025获奖方案使用从3次→6次, Optiver/Enefit冠军全部使用Polars

### 6.3 RAPIDS cuML

```python
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.svm import SVR
from cuml.linear_model import Ridge

# API与sklearn完全兼容, 但在GPU上运行
# Deotte的72模型Stacking方案核心加速手段
```
