"""
Kaggle Feature Engineering Template
GM-level feature engineering functions for tabular competitions.
"""
import itertools
from typing import List, Optional

import numpy as np
import pandas as pd


class GroupbyFeatures:
    """Groupby aggregation features (highest value technique)."""

    def __init__(self, group_cols: List[str], agg_cols: List[str],
                 agg_funcs: Optional[List[str]] = None):
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs or ["mean", "std", "count", "min", "max"]
        self.agg_maps_ = {}

    def fit(self, df: pd.DataFrame) -> "GroupbyFeatures":
        self.agg_maps_ = {}
        for g_col in self.group_cols:
            for a_col in self.agg_cols:
                for func in self.agg_funcs:
                    key = f"{g_col}_{a_col}_{func}"
                    self.agg_maps_[key] = (
                        df.groupby(g_col)[a_col].agg(func).to_dict()
                    )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for g_col in self.group_cols:
            for a_col in self.agg_cols:
                for func in self.agg_funcs:
                    key = f"{g_col}_{a_col}_{func}"
                    result[key] = df[g_col].map(self.agg_maps_[key])
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class CrossFeatures:
    """Category cross features and numerical interactions."""

    def __init__(self, cat_cols: Optional[List[str]] = None,
                 num_cols: Optional[List[str]] = None,
                 max_cat_combinations: int = 2,
                 num_operations: Optional[List[str]] = None):
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.max_cat_combinations = max_cat_combinations
        self.num_operations = num_operations or ["-", "/", "*"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)

        # Category crosses
        for r in range(2, self.max_cat_combinations + 1):
            for cols in itertools.combinations(self.cat_cols, r):
                name = "_x_".join(cols)
                result[name] = df[cols[0]].astype(str)
                for c in cols[1:]:
                    result[name] += "_" + df[c].astype(str)

        # Numerical interactions
        for c1, c2 in itertools.combinations(self.num_cols[:20], 2):
            if "-" in self.num_operations:
                result[f"{c1}_minus_{c2}"] = df[c1] - df[c2]
            if "/" in self.num_operations:
                result[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-8)
            if "*" in self.num_operations:
                result[f"{c1}_mult_{c2}"] = df[c1] * df[c2]

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)


class TargetEncoder:
    """Target encoding with CV to prevent leakage."""

    def __init__(self, cols: List[str], n_splits: int = 5,
                 smoothing: float = 10.0, random_state: int = 42):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_ = None
        self.encoding_maps_ = {}

    def fit(self, df: pd.DataFrame, target: str) -> "TargetEncoder":
        self.global_mean_ = df[target].mean()
        for col in self.cols:
            group = df.groupby(col)[target]
            means = group.mean()
            counts = group.count()
            smooth_means = (counts * means + self.smoothing * self.global_mean_) / (
                counts + self.smoothing
            )
            self.encoding_maps_[col] = smooth_means.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col in self.cols:
            result[f"{col}_te"] = df[col].map(self.encoding_maps_[col])
            result[f"{col}_te"].fillna(self.global_mean_, inplace=True)
        return result

    def fit_transform_cv(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """CV-safe target encoding for training data."""
        from sklearn.model_selection import KFold

        result = pd.DataFrame(index=df.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in self.cols:
            encoded = pd.Series(index=df.index, dtype=float)
            for tr_idx, va_idx in kf.split(df):
                train_part = df.iloc[tr_idx]
                group = train_part.groupby(col)[target]
                means = group.mean()
                counts = group.count()
                smooth = (counts * means + self.smoothing * df[target].mean()) / (
                    counts + self.smoothing
                )
                encoded.iloc[va_idx] = df.iloc[va_idx][col].map(smooth.to_dict())

            encoded.fillna(df[target].mean(), inplace=True)
            result[f"{col}_te"] = encoded

        # Also fit full maps for test transform
        self.fit(df, target)
        return result


class TimeFeatures:
    """Time series feature engineering."""

    def __init__(self, date_col: str, target_col: Optional[str] = None,
                 group_col: Optional[str] = None,
                 lags: Optional[List[int]] = None,
                 rolling_windows: Optional[List[int]] = None):
        self.date_col = date_col
        self.target_col = target_col
        self.group_col = group_col
        self.lags = lags or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        dt = pd.to_datetime(df[self.date_col])

        # Calendar features
        result["year"] = dt.dt.year
        result["month"] = dt.dt.month
        result["day"] = dt.dt.day
        result["dayofweek"] = dt.dt.dayofweek
        result["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        result["quarter"] = dt.dt.quarter
        result["dayofyear"] = dt.dt.dayofyear

        # Cyclical encoding
        result["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        result["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        result["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        result["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

        # Lag and rolling features
        if self.target_col and self.target_col in df.columns:
            target = df[self.target_col]
            if self.group_col:
                grouped = df.groupby(self.group_col)[self.target_col]
                for lag in self.lags:
                    result[f"lag_{lag}"] = grouped.shift(lag)
                for window in self.rolling_windows:
                    result[f"rolling_mean_{window}"] = grouped.transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                    )
                    result[f"rolling_std_{window}"] = grouped.transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).std()
                    )
            else:
                for lag in self.lags:
                    result[f"lag_{lag}"] = target.shift(lag)
                for window in self.rolling_windows:
                    result[f"rolling_mean_{window}"] = (
                        target.shift(1).rolling(window, min_periods=1).mean()
                    )

        return result


class NullFeatures:
    """Missing value indicator features."""

    def __init__(self, cols: Optional[List[str]] = None, threshold: float = 0.01):
        self.cols = cols
        self.threshold = threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.cols or df.columns.tolist()
        cols_with_nulls = [c for c in cols if df[c].isnull().mean() > self.threshold]

        result = pd.DataFrame(index=df.index)
        for col in cols_with_nulls:
            result[f"{col}_is_null"] = df[col].isnull().astype(int)
        result["null_count"] = df[cols].isnull().sum(axis=1)
        result["null_ratio"] = result["null_count"] / len(cols)
        return result


class FeatureSelector:
    """Feature selection methods."""

    def __init__(self, method: str = "importance", top_k: int = 500):
        self.method = method
        self.top_k = top_k
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        if self.method == "importance":
            self.selected_features_ = self._importance_selection(X, y)
        elif self.method == "null_importance":
            self.selected_features_ = self._null_importance_selection(X, y)
        elif self.method == "correlation":
            self.selected_features_ = self._correlation_selection(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def _importance_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features by LightGBM importance."""
        from lightgbm import LGBMClassifier, LGBMRegressor

        if y.nunique() <= 30:
            model = LGBMClassifier(n_estimators=200, verbose=-1)
        else:
            model = LGBMRegressor(n_estimators=200, verbose=-1)

        model.fit(X.fillna(-999), y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.nlargest(self.top_k).index.tolist()
        return top_features

    def _null_importance_selection(self, X: pd.DataFrame, y: pd.Series,
                                   n_runs: int = 20) -> List[str]:
        """Null importance: compare real vs shuffled target importance."""
        from lightgbm import LGBMClassifier, LGBMRegressor

        ModelClass = LGBMClassifier if y.nunique() <= 30 else LGBMRegressor

        # Real importance
        model = ModelClass(n_estimators=200, verbose=-1)
        model.fit(X.fillna(-999), y)
        real_imp = pd.Series(model.feature_importances_, index=X.columns)

        # Null importance (shuffled target)
        null_imps = pd.DataFrame(index=X.columns)
        for i in range(n_runs):
            y_shuffled = y.sample(frac=1, random_state=i).reset_index(drop=True)
            model.fit(X.fillna(-999), y_shuffled)
            null_imps[i] = model.feature_importances_

        # Select features where real > 95th percentile of null
        threshold = null_imps.quantile(0.95, axis=1)
        selected = real_imp[real_imp > threshold].index.tolist()
        return selected[:self.top_k]

    def _correlation_selection(self, X: pd.DataFrame,
                                threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return [c for c in X.columns if c not in to_drop]


class FloatDigitExtractor:
    """Extract hidden categorical encodings from float32 values."""

    def __init__(self, cols: List[str], multipliers: Optional[List[int]] = None):
        self.cols = cols
        self.multipliers = multipliers or [10, 100, 1000]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col in self.cols:
            for mult in self.multipliers:
                result[f"{col}_digit_{mult}"] = (df[col] * mult).astype(int) % 10
        return result


# ---- Pipeline ----

class FeaturePipeline:
    """Unified feature engineering pipeline."""

    def __init__(self, steps: list):
        """
        steps: list of (name, transformer) tuples
        Example:
            pipeline = FeaturePipeline([
                ('groupby', GroupbyFeatures(['cat1'], ['num1'])),
                ('cross', CrossFeatures(cat_cols=['cat1','cat2'])),
                ('nulls', NullFeatures()),
            ])
        """
        self.steps = steps

    def fit_transform(self, df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        results = [df]
        for name, transformer in self.steps:
            if hasattr(transformer, "fit_transform"):
                if "target" in transformer.__class__.__init__.__code__.co_varnames:
                    result = transformer.fit_transform(df, target)
                else:
                    result = transformer.fit_transform(df)
            else:
                result = transformer.transform(df)
            results.append(result)
            print(f"  [{name}] +{result.shape[1]} features")
        combined = pd.concat(results, axis=1)
        print(f"  Total: {combined.shape[1]} features")
        return combined

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        results = [df]
        for name, transformer in self.steps:
            result = transformer.transform(df)
            results.append(result)
        return pd.concat(results, axis=1)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "cat1": np.random.choice(["A", "B", "C"], n),
        "cat2": np.random.choice(["X", "Y"], n),
        "num1": np.random.randn(n),
        "num2": np.random.randn(n) * 2 + 1,
        "num3": np.random.exponential(1, n),
        "target": np.random.randint(0, 2, n),
    })
    df.loc[np.random.choice(n, 50), "num1"] = np.nan

    print("Feature Engineering Demo")
    print("=" * 40)

    pipeline = FeaturePipeline([
        ("groupby", GroupbyFeatures(["cat1"], ["num1", "num2"])),
        ("cross", CrossFeatures(cat_cols=["cat1", "cat2"], num_cols=["num1", "num2", "num3"])),
        ("nulls", NullFeatures()),
    ])

    result = pipeline.fit_transform(df, target="target")
    print(f"\nFinal shape: {result.shape}")
