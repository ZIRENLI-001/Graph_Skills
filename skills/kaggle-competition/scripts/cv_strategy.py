"""
Kaggle Cross-Validation Strategy Template
Includes CV factory, OOF management, and CV-LB tracking.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


class CVFactory:
    """Automatically recommend and create CV strategy based on data characteristics."""

    @staticmethod
    def recommend(df: pd.DataFrame, target: str,
                  time_col: Optional[str] = None,
                  group_col: Optional[str] = None,
                  n_splits: int = 5) -> dict:
        """Recommend CV strategy based on data characteristics."""
        info = {
            "n_samples": len(df),
            "target_type": "classification" if df[target].nunique() <= 30 else "regression",
            "target_classes": df[target].nunique(),
            "has_time": time_col is not None,
            "has_groups": group_col is not None,
        }

        if time_col and group_col:
            strategy = "GroupTimeSeriesSplit"
            reason = "Data has both temporal and group structure"
        elif time_col:
            strategy = "TimeSeriesSplit"
            reason = "Data has temporal structure - must respect time ordering"
        elif group_col:
            strategy = "GroupKFold"
            reason = "Data has groups - same group must stay in same fold"
        elif info["target_type"] == "classification":
            if len(df) < 5000:
                strategy = "RepeatedStratifiedKFold"
                reason = "Small classification dataset - use repeated CV for stability"
            else:
                strategy = "StratifiedKFold"
                reason = "Classification with sufficient data"
        else:
            strategy = "KFold"
            reason = "Regression task"

        print(f"Recommended CV: {strategy}")
        print(f"Reason: {reason}")
        print(f"Data: {info}")

        return {"strategy": strategy, "reason": reason, "info": info}

    @staticmethod
    def create(strategy: str, n_splits: int = 5,
               n_repeats: int = 3, gap: int = 0,
               random_state: int = 42):
        """Create CV splitter instance."""
        if strategy == "StratifiedKFold":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        elif strategy == "KFold":
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        elif strategy == "GroupKFold":
            return GroupKFold(n_splits=n_splits)
        elif strategy == "TimeSeriesSplit":
            return TimeSeriesSplit(n_splits=n_splits, gap=gap)
        elif strategy == "RepeatedStratifiedKFold":
            return RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
        elif strategy == "GroupTimeSeriesSplit":
            return GroupTimeSeriesSplit(n_splits=n_splits, gap=gap)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class GroupTimeSeriesSplit:
    """Time series split that respects group structure.
    Groups (e.g. stores) are kept in same fold, and time ordering is respected.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None, time=None):
        if groups is None:
            raise ValueError("groups must be provided")

        unique_times = sorted(X.index if time is None else time.unique())
        n_times = len(unique_times)
        test_size = n_times // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = test_size * (i + 1)
            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n_times:
                break

            train_times = set(unique_times[:train_end])
            test_times = set(unique_times[test_start:test_end])

            if time is not None:
                train_idx = np.where(time.isin(train_times))[0]
                test_idx = np.where(time.isin(test_times))[0]
            else:
                train_idx = np.where(X.index.isin(train_times))[0]
                test_idx = np.where(X.index.isin(test_times))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class OOFManager:
    """Manage out-of-fold predictions for stacking and evaluation."""

    def __init__(self, save_dir: str = "oof_predictions"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.predictions = {}

    def save_oof(self, name: str, oof_preds: np.ndarray,
                 test_preds: Optional[np.ndarray] = None,
                 cv_score: Optional[float] = None):
        """Save OOF predictions."""
        data = {
            "oof": oof_preds,
            "cv_score": cv_score,
        }
        if test_preds is not None:
            data["test"] = test_preds

        np.savez(self.save_dir / f"{name}.npz", **{k: v for k, v in data.items() if v is not None})
        self.predictions[name] = data
        print(f"Saved OOF: {name} (CV={cv_score:.5f})" if cv_score else f"Saved OOF: {name}")

    def load_oof(self, name: str) -> dict:
        """Load OOF predictions."""
        data = np.load(self.save_dir / f"{name}.npz", allow_pickle=True)
        result = {k: data[k] for k in data.files}
        self.predictions[name] = result
        return result

    def load_all(self) -> Dict[str, dict]:
        """Load all saved OOF predictions."""
        for path in self.save_dir.glob("*.npz"):
            name = path.stem
            self.load_oof(name)
        return self.predictions

    def get_oof_matrix(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Get stacking-ready OOF matrix."""
        oof_dict = {}
        test_dict = {}
        for name, data in self.predictions.items():
            oof_dict[name] = data["oof"]
            if "test" in data:
                test_dict[name] = data["test"]

        oof_df = pd.DataFrame(oof_dict)
        test_df = pd.DataFrame(test_dict) if test_dict else None
        return oof_df, test_df


class CVLBTracker:
    """Track CV-LB consistency across experiments."""

    def __init__(self, log_path: str = "cv_lb_log.csv"):
        self.log_path = Path(log_path)
        if self.log_path.exists():
            self.log = pd.read_csv(self.log_path)
        else:
            self.log = pd.DataFrame(columns=[
                "experiment_id", "datetime", "method", "cv_score",
                "lb_score", "notes"
            ])

    def add(self, method: str, cv_score: float,
            lb_score: Optional[float] = None, notes: str = ""):
        """Add experiment result."""
        exp_id = len(self.log) + 1
        new_row = pd.DataFrame([{
            "experiment_id": exp_id,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "method": method,
            "cv_score": cv_score,
            "lb_score": lb_score,
            "notes": notes,
        }])
        self.log = pd.concat([self.log, new_row], ignore_index=True)
        self.log.to_csv(self.log_path, index=False)
        print(f"[{exp_id}] {method}: CV={cv_score:.5f}" +
              (f", LB={lb_score:.5f}" if lb_score else ""))

    def plot(self):
        """Plot CV vs LB consistency."""
        import matplotlib.pyplot as plt

        valid = self.log.dropna(subset=["lb_score"])
        if len(valid) < 2:
            print("Need at least 2 experiments with LB scores to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # CV vs LB scatter
        axes[0].scatter(valid["cv_score"], valid["lb_score"])
        for _, row in valid.iterrows():
            axes[0].annotate(row["method"],
                             (row["cv_score"], row["lb_score"]),
                             fontsize=8)
        axes[0].set_xlabel("CV Score")
        axes[0].set_ylabel("LB Score")
        axes[0].set_title("CV vs LB Consistency")

        corr = valid[["cv_score", "lb_score"]].corr().iloc[0, 1]
        axes[0].text(0.05, 0.95, f"Correlation: {corr:.4f}",
                     transform=axes[0].transAxes, fontsize=12,
                     verticalalignment="top")

        # Score over time
        axes[1].plot(valid["experiment_id"], valid["cv_score"], "bo-", label="CV")
        axes[1].plot(valid["experiment_id"], valid["lb_score"], "ro-", label="LB")
        axes[1].set_xlabel("Experiment #")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Score Progression")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("cv_lb_tracking.png", dpi=100, bbox_inches="tight")
        print("Saved cv_lb_tracking.png")
        plt.close()

    def summary(self):
        """Print experiment summary."""
        print(f"\n{'='*70}")
        print(f"  Experiment Summary ({len(self.log)} experiments)")
        print(f"{'='*70}")
        print(self.log.to_string(index=False))

        valid = self.log.dropna(subset=["lb_score"])
        if len(valid) >= 2:
            corr = valid[["cv_score", "lb_score"]].corr().iloc[0, 1]
            print(f"\n  CV-LB Correlation: {corr:.4f}")
            if corr > 0.9:
                print("  → CV is reliable! Trust CV over LB.")
            elif corr > 0.7:
                print("  → Moderate consistency. CV mostly trustworthy.")
            else:
                print("  → WARNING: Low consistency. Review CV strategy!")

        best = self.log.loc[self.log["cv_score"].idxmax()]
        print(f"\n  Best CV: {best['method']} = {best['cv_score']:.5f}")


class AdversarialValidator:
    """Detect train-test distribution shift."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.feature_importances_ = None

    def validate(self, train: pd.DataFrame, test: pd.DataFrame,
                 features: List[str]) -> float:
        """Run adversarial validation and return AUC."""
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score

        combined = pd.concat(
            [train[features], test[features]], ignore_index=True
        ).fillna(-999)
        labels = np.array([0] * len(train) + [1] * len(test))

        clf = LGBMClassifier(n_estimators=100, verbose=-1)
        scores = cross_val_score(
            clf, combined, labels, cv=self.n_splits, scoring="roc_auc"
        )
        auc = scores.mean()

        # Feature importances
        clf.fit(combined, labels)
        self.feature_importances_ = pd.Series(
            clf.feature_importances_, index=features
        ).sort_values(ascending=False)

        print(f"Adversarial AUC: {auc:.4f}")
        if auc < 0.55:
            print("→ Distributions are SIMILAR (good)")
        elif auc < 0.70:
            print("→ MODERATE shift detected")
        else:
            print("→ SIGNIFICANT shift detected!")

        print("\nTop features causing shift:")
        for col, imp in self.feature_importances_.head(10).items():
            print(f"  {col}: {imp}")

        return auc


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "target": np.random.randint(0, 2, n),
    })

    # 1. CV recommendation
    print("=" * 50)
    print("CV Strategy Recommendation")
    print("=" * 50)
    rec = CVFactory.recommend(df, "target")
    cv = CVFactory.create(rec["strategy"])

    # 2. CV-LB tracker
    print("\n" + "=" * 50)
    print("CV-LB Tracker Demo")
    print("=" * 50)
    tracker = CVLBTracker("demo_log.csv")
    tracker.add("lgb_baseline", 0.852, 0.850, "5-fold baseline")
    tracker.add("lgb+groupby", 0.857, 0.854, "+groupby features")
    tracker.add("lgb+xgb_ensemble", 0.862, 0.858, "+XGB ensemble")
    tracker.summary()

    # Cleanup demo file
    Path("demo_log.csv").unlink(missing_ok=True)
