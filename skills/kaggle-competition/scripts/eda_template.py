"""
Kaggle EDA (Exploratory Data Analysis) Template
Usage: python eda_template.py --train train.csv --test test.csv --target target_col
"""
import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def load_data(path: str) -> pd.DataFrame:
    """Auto-detect file format and load data."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(path)
    elif p.suffix == ".feather":
        return pd.read_feather(path)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    else:
        return pd.read_csv(path)


def overview(df: pd.DataFrame, name: str = "Dataset"):
    """Print basic statistics."""
    print(f"\n{'='*60}")
    print(f"  {name} Overview")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  Dtypes: {dict(df.dtypes.value_counts())}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        print(f"\n  Missing values ({len(missing_pct)} columns):")
        for col, pct in missing_pct.head(10).items():
            print(f"    {col}: {pct:.1f}%")
        if len(missing_pct) > 10:
            print(f"    ... and {len(missing_pct) - 10} more")
    else:
        print("\n  No missing values!")

    nunique = df.nunique()
    print(f"\n  Unique values (top 10):")
    for col, n in nunique.sort_values(ascending=False).head(10).items():
        print(f"    {col}: {n:,}")


def analyze_target(df: pd.DataFrame, target: str, save_dir: str = "eda_output"):
    """Analyze target variable."""
    print(f"\n{'='*60}")
    print(f"  Target Analysis: {target}")
    print(f"{'='*60}")

    y = df[target]
    print(f"  Type: {y.dtype}")
    print(f"  Unique: {y.nunique()}")
    print(f"  Missing: {y.isnull().sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if y.nunique() <= 30:
        # Classification
        vc = y.value_counts().sort_index()
        print(f"\n  Class distribution:")
        for cls, cnt in vc.items():
            print(f"    {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")
        vc.plot(kind="bar", ax=axes[0], title=f"{target} Distribution")
        axes[0].set_ylabel("Count")
        vc_pct = vc / vc.sum()
        vc_pct.plot(kind="bar", ax=axes[1], title=f"{target} Proportion")
        axes[1].set_ylabel("Proportion")
    else:
        # Regression
        print(f"  Mean: {y.mean():.4f}")
        print(f"  Std:  {y.std():.4f}")
        print(f"  Min:  {y.min():.4f}")
        print(f"  Max:  {y.max():.4f}")
        print(f"  Skew: {y.skew():.4f}")
        y.hist(bins=50, ax=axes[0])
        axes[0].set_title(f"{target} Distribution")
        np.log1p(y.clip(lower=0)).hist(bins=50, ax=axes[1])
        axes[1].set_title(f"log1p({target}) Distribution")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/target_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()


def analyze_numerical(df: pd.DataFrame, target: str, save_dir: str = "eda_output"):
    """Analyze numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target]

    if not num_cols:
        print("\n  No numerical features found.")
        return

    print(f"\n{'='*60}")
    print(f"  Numerical Features: {len(num_cols)}")
    print(f"{'='*60}")

    stats = df[num_cols].describe().T
    stats["missing%"] = df[num_cols].isnull().mean() * 100
    stats["skew"] = df[num_cols].skew()
    print(stats[["mean", "std", "min", "max", "skew", "missing%"]].to_string())

    # Correlation heatmap (top 20 features by correlation with target)
    if target in df.columns and df[target].dtype in [np.float64, np.int64, np.float32, np.int32]:
        corr_with_target = df[num_cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
        top_cols = corr_with_target.head(20).index.tolist()

        if len(top_cols) >= 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = df[top_cols + [target]].corr()
            sns.heatmap(corr_matrix, annot=len(top_cols) <= 15, fmt=".2f",
                        cmap="RdBu_r", center=0, ax=ax)
            ax.set_title("Correlation Matrix (Top 20 features by target correlation)")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=100, bbox_inches="tight")
            plt.close()

            print(f"\n  Top 10 correlated with target:")
            for col, corr in corr_with_target.head(10).items():
                print(f"    {col}: {corr:.4f}")

    # Distribution plots (top 12)
    plot_cols = num_cols[:12]
    n_plots = len(plot_cols)
    if n_plots > 0:
        n_rows = (n_plots + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_plots > 4 else [axes] if n_plots == 1 else axes
        for i, col in enumerate(plot_cols):
            df[col].hist(bins=50, ax=axes[i])
            axes[i].set_title(col, fontsize=10)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/numerical_distributions.png", dpi=100, bbox_inches="tight")
        plt.close()


def analyze_categorical(df: pd.DataFrame, target: str, save_dir: str = "eda_output"):
    """Analyze categorical features."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        print("\n  No categorical features found.")
        return

    print(f"\n{'='*60}")
    print(f"  Categorical Features: {len(cat_cols)}")
    print(f"{'='*60}")

    for col in cat_cols:
        nunique = df[col].nunique()
        missing = df[col].isnull().mean() * 100
        print(f"\n  {col}: {nunique} unique, {missing:.1f}% missing")
        if nunique <= 20:
            vc = df[col].value_counts().head(10)
            for val, cnt in vc.items():
                print(f"    {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")


def train_test_comparison(train: pd.DataFrame, test: pd.DataFrame,
                          target: str, save_dir: str = "eda_output"):
    """Compare train and test distributions."""
    print(f"\n{'='*60}")
    print(f"  Train-Test Distribution Comparison")
    print(f"{'='*60}")

    common_cols = [c for c in train.columns if c in test.columns and c != target]
    num_cols = [c for c in common_cols if train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    if not num_cols:
        return

    # Plot distributions
    plot_cols = num_cols[:12]
    n_plots = len(plot_cols)
    if n_plots > 0:
        n_rows = (n_plots + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_plots > 4 else [axes] if n_plots == 1 else axes
        for i, col in enumerate(plot_cols):
            train[col].hist(bins=50, ax=axes[i], alpha=0.5, label="Train", density=True)
            test[col].hist(bins=50, ax=axes[i], alpha=0.5, label="Test", density=True)
            axes[i].set_title(col, fontsize=10)
            axes[i].legend(fontsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/train_test_comparison.png", dpi=100, bbox_inches="tight")
        plt.close()


def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame,
                           target: str, save_dir: str = "eda_output"):
    """Adversarial validation to detect train-test distribution shift."""
    try:
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("\n  Skipping adversarial validation (lightgbm not installed)")
        return

    print(f"\n{'='*60}")
    print(f"  Adversarial Validation")
    print(f"{'='*60}")

    common_cols = [c for c in train.columns if c in test.columns and c != target]
    num_cols = [c for c in common_cols
                if train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    if len(num_cols) < 2:
        print("  Not enough numerical features for adversarial validation.")
        return

    combined = pd.concat([train[num_cols], test[num_cols]], ignore_index=True)
    labels = np.array([0] * len(train) + [1] * len(test))

    clf = LGBMClassifier(n_estimators=100, verbose=-1)
    scores = cross_val_score(clf, combined.fillna(-999), labels, cv=5, scoring="roc_auc")
    auc = scores.mean()

    print(f"  AUC: {auc:.4f} (±{scores.std():.4f})")
    if auc < 0.55:
        print("  → Train and test distributions are SIMILAR (good!)")
    elif auc < 0.70:
        print("  → MODERATE distribution shift detected")
    else:
        print("  → SIGNIFICANT distribution shift detected!")

    # Feature importance for shift
    clf.fit(combined.fillna(-999), labels)
    importances = pd.Series(clf.feature_importances_, index=num_cols).sort_values(ascending=False)
    print(f"\n  Top features causing shift:")
    for col, imp in importances.head(10).items():
        print(f"    {col}: {imp}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle EDA Template")
    parser.add_argument("--train", required=True, help="Path to train data")
    parser.add_argument("--test", default=None, help="Path to test data")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="eda_output", help="Output directory")
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    train = load_data(args.train)
    test = load_data(args.test) if args.test else None

    # Run analyses
    overview(train, "Train")
    if test is not None:
        overview(test, "Test")

    if args.target in train.columns:
        analyze_target(train, args.target, args.output)

    analyze_numerical(train, args.target, args.output)
    analyze_categorical(train, args.target, args.output)

    if test is not None:
        train_test_comparison(train, test, args.target, args.output)
        adversarial_validation(train, test, args.target, args.output)

    print(f"\n{'='*60}")
    print(f"  EDA complete! Charts saved to {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
