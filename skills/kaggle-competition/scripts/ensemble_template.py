"""
Kaggle Model Ensemble Template
Stacking, blending, weighted average, rank averaging, hill climbing.
"""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata


class WeightedAverage:
    """Weighted average ensemble with weight optimization."""

    def __init__(self, metric_func: Callable, direction: str = "maximize"):
        self.metric_func = metric_func
        self.direction = direction
        self.weights_ = None

    def optimize(self, oof_preds: Dict[str, np.ndarray],
                 y_true: np.ndarray) -> np.ndarray:
        """Find optimal weights using scipy.optimize."""
        names = list(oof_preds.keys())
        preds_list = [oof_preds[n] for n in names]
        n_models = len(preds_list)

        def objective(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            blended = sum(w * p for w, p in zip(weights, preds_list))
            score = self.metric_func(y_true, blended)
            return -score if self.direction == "maximize" else score

        initial = np.ones(n_models) / n_models
        result = minimize(objective, initial, method="Nelder-Mead",
                          options={"maxiter": 10000, "xatol": 1e-8})

        self.weights_ = np.abs(result.x) / np.abs(result.x).sum()

        print("Optimal weights:")
        for name, w in zip(names, self.weights_):
            print(f"  {name}: {w:.4f}")

        blended = sum(w * p for w, p in zip(self.weights_, preds_list))
        score = self.metric_func(y_true, blended)
        print(f"Ensemble score: {score:.5f}")

        return self.weights_

    def predict(self, test_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply optimized weights to test predictions."""
        names = list(test_preds.keys())
        preds_list = [test_preds[n] for n in names]
        return sum(w * p for w, p in zip(self.weights_, preds_list))


class RankAverage:
    """Rank averaging - best for AUC metric (only relative order matters)."""

    @staticmethod
    def blend(preds_dict: Dict[str, np.ndarray],
              weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Convert predictions to ranks, then weighted average."""
        names = list(preds_dict.keys())

        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}

        total_weight = sum(weights.values())
        result = np.zeros(len(next(iter(preds_dict.values()))))

        for name in names:
            ranks = rankdata(preds_dict[name]) / len(preds_dict[name])
            result += weights.get(name, 1.0 / len(names)) * ranks

        return result / total_weight


class HillClimbing:
    """Greedy hill climbing to find optimal model subset."""

    def __init__(self, metric_func: Callable, direction: str = "maximize"):
        self.metric_func = metric_func
        self.direction = direction
        self.selected_ = []

    def search(self, oof_preds: Dict[str, np.ndarray],
               y_true: np.ndarray, max_models: int = 20) -> List[str]:
        """Greedy forward selection of models."""
        names = list(oof_preds.keys())
        selected = []
        selected_preds = []
        best_score = -np.inf if self.direction == "maximize" else np.inf

        def is_better(new, old):
            return new > old if self.direction == "maximize" else new < old

        print("Hill Climbing Ensemble Selection")
        print("=" * 60)

        for step in range(min(max_models, len(names))):
            improved = False
            best_name = None

            for name in names:
                if name in selected:
                    continue

                trial_preds = selected_preds + [oof_preds[name]]
                trial_pred = np.mean(trial_preds, axis=0)
                score = self.metric_func(y_true, trial_pred)

                if is_better(score, best_score):
                    best_score = score
                    best_name = name
                    improved = True

            if not improved:
                break

            selected.append(best_name)
            selected_preds.append(oof_preds[best_name])
            print(f"  Step {step+1}: +{best_name} → score={best_score:.5f}")

        self.selected_ = selected
        print(f"\nSelected {len(selected)} models")
        return selected


class StackingEnsemble:
    """Multi-level stacking framework."""

    def __init__(self, base_models: dict, meta_model,
                 n_splits: int = 5, include_original: bool = False):
        """
        base_models: dict of {name: model_instance}
        meta_model: sklearn-compatible model for level 2
        include_original: whether to include original features in level 2
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.include_original = include_original
        self.fitted_models_ = {}

    def fit_predict(self, X_train: pd.DataFrame, y_train: np.ndarray,
                    X_test: pd.DataFrame,
                    cv=None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit stacking and return OOF + test predictions."""
        from sklearn.model_selection import StratifiedKFold, KFold

        if cv is None:
            if y_train.dtype in [np.int32, np.int64] and len(np.unique(y_train)) <= 30:
                cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        n_train = len(X_train)
        n_test = len(X_test)
        n_models = len(self.base_models)

        # Level 1: Generate OOF predictions
        oof_matrix = np.zeros((n_train, n_models))
        test_matrix = np.zeros((n_test, n_models))

        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Training Level 1: {name}")
            oof_pred = np.zeros(n_train)
            test_pred = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr = X_train.iloc[tr_idx] if isinstance(X_train, pd.DataFrame) else X_train[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = X_train.iloc[va_idx] if isinstance(X_train, pd.DataFrame) else X_train[va_idx]

                import copy
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_tr, y_tr)

                if hasattr(fold_model, "predict_proba"):
                    oof_pred[va_idx] = fold_model.predict_proba(X_va)[:, 1]
                    test_pred += fold_model.predict_proba(X_test)[:, 1] / self.n_splits
                else:
                    oof_pred[va_idx] = fold_model.predict(X_va)
                    test_pred += fold_model.predict(X_test) / self.n_splits

            oof_matrix[:, i] = oof_pred
            test_matrix[:, i] = test_pred

        # Level 2: Train meta model
        print(f"\nTraining Level 2: {type(self.meta_model).__name__}")

        if self.include_original:
            meta_train = np.hstack([oof_matrix,
                                    X_train.values if isinstance(X_train, pd.DataFrame) else X_train])
            meta_test = np.hstack([test_matrix,
                                   X_test.values if isinstance(X_test, pd.DataFrame) else X_test])
        else:
            meta_train = oof_matrix
            meta_test = test_matrix

        # OOF for meta model
        meta_oof = np.zeros(n_train)
        meta_test_pred = np.zeros(n_test)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(meta_train, y_train)):
            import copy
            fold_meta = copy.deepcopy(self.meta_model)
            fold_meta.fit(meta_train[tr_idx], y_train[tr_idx])

            if hasattr(fold_meta, "predict_proba"):
                meta_oof[va_idx] = fold_meta.predict_proba(meta_train[va_idx])[:, 1]
                meta_test_pred += fold_meta.predict_proba(meta_test)[:, 1] / self.n_splits
            else:
                meta_oof[va_idx] = fold_meta.predict(meta_train[va_idx])
                meta_test_pred += fold_meta.predict(meta_test) / self.n_splits

        return meta_oof, meta_test_pred


class PostProcessor:
    """Post-processing utilities for different metrics."""

    @staticmethod
    def clip_predictions(preds: np.ndarray,
                         low: float = 0.01, high: float = 0.99) -> np.ndarray:
        """Clip predictions for LogLoss metric."""
        return np.clip(preds, low, high)

    @staticmethod
    def scale_predictions(preds: np.ndarray, factor: float = 0.99) -> np.ndarray:
        """Scale predictions for LogLoss calibration."""
        return preds * factor

    @staticmethod
    def optimize_threshold(y_true: np.ndarray, y_pred: np.ndarray,
                           metric_func: Callable,
                           n_thresholds: int = 1000) -> Tuple[float, float]:
        """Find optimal binary classification threshold."""
        thresholds = np.linspace(0, 1, n_thresholds)
        best_score = -np.inf
        best_threshold = 0.5

        for t in thresholds:
            pred_labels = (y_pred >= t).astype(int)
            score = metric_func(y_true, pred_labels)
            if score > best_score:
                best_score = score
                best_threshold = t

        print(f"Optimal threshold: {best_threshold:.4f} (score={best_score:.5f})")
        return best_threshold, best_score

    @staticmethod
    def optimize_multiclass_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                                       n_classes: int,
                                       metric_func: Callable) -> np.ndarray:
        """Optimize per-class thresholds for F1-macro."""
        thresholds = np.full(n_classes, 0.5)

        for cls in range(n_classes):
            best_score = -np.inf
            for t in np.linspace(0.1, 0.9, 81):
                trial_thresholds = thresholds.copy()
                trial_thresholds[cls] = t
                pred_labels = np.array([
                    cls if y_pred[i, cls] >= trial_thresholds[cls] else -1
                    for i in range(len(y_pred))
                    for cls in range(n_classes)
                ])
                # Simplified - in practice use per-class binary evaluation
                score = metric_func(y_true, pred_labels)
                if score > best_score:
                    best_score = score
                    thresholds[cls] = t

        return thresholds

    @staticmethod
    def optimized_rounder(y_true: np.ndarray, y_pred: np.ndarray,
                          n_classes: int) -> np.ndarray:
        """Optimize rounding thresholds for QWK metric."""
        from scipy.optimize import minimize

        def qwk_loss(thresholds):
            thresholds = sorted(thresholds)
            pred_labels = pd.cut(y_pred, [-np.inf] + list(thresholds) + [np.inf],
                                 labels=range(n_classes))
            from sklearn.metrics import cohen_kappa_score
            return -cohen_kappa_score(y_true, pred_labels, weights="quadratic")

        initial = np.arange(1, n_classes) / n_classes * y_pred.max()
        result = minimize(qwk_loss, initial, method="Nelder-Mead")
        optimal_thresholds = sorted(result.x)

        print(f"Optimal thresholds: {[f'{t:.4f}' for t in optimal_thresholds]}")
        return np.array(optimal_thresholds)

    @staticmethod
    def majority_voting(predictions_list: List[np.ndarray]) -> np.ndarray:
        """Majority voting for classification or math competitions."""
        from collections import Counter
        n_samples = len(predictions_list[0])
        result = np.zeros(n_samples)

        for i in range(n_samples):
            votes = [preds[i] for preds in predictions_list]
            result[i] = Counter(votes).most_common(1)[0][0]

        return result


def evaluate_ensemble(oof_preds: Dict[str, np.ndarray],
                      y_true: np.ndarray,
                      metric_func: Callable,
                      metric_name: str = "Score"):
    """Evaluate individual models and ensemble."""
    print(f"\n{'='*60}")
    print(f"  Ensemble Evaluation Report")
    print(f"{'='*60}")

    # Individual scores
    scores = {}
    for name, pred in oof_preds.items():
        score = metric_func(y_true, pred)
        scores[name] = score
        print(f"  {name}: {metric_name}={score:.5f}")

    # Simple average
    avg_pred = np.mean(list(oof_preds.values()), axis=0)
    avg_score = metric_func(y_true, avg_pred)
    print(f"\n  Simple Average: {metric_name}={avg_score:.5f}")

    # Rank average
    rank_pred = RankAverage.blend(oof_preds)
    rank_score = metric_func(y_true, rank_pred)
    print(f"  Rank Average:   {metric_name}={rank_score:.5f}")

    # Best single model
    best_name = max(scores, key=scores.get)
    print(f"\n  Best single: {best_name} ({scores[best_name]:.5f})")
    print(f"  Avg improvement: {avg_score - scores[best_name]:+.5f}")
    print(f"  Rank improvement: {rank_score - scores[best_name]:+.5f}")


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    # Demo
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)

    oof_preds = {
        "lgb": y_true * 0.7 + np.random.randn(n) * 0.3,
        "xgb": y_true * 0.65 + np.random.randn(n) * 0.35,
        "cat": y_true * 0.68 + np.random.randn(n) * 0.32,
        "nn": y_true * 0.60 + np.random.randn(n) * 0.4,
    }

    # Clip to [0, 1]
    oof_preds = {k: np.clip(v, 0, 1) for k, v in oof_preds.items()}

    # 1. Evaluate all methods
    evaluate_ensemble(oof_preds, y_true, roc_auc_score, "AUC")

    # 2. Optimize weights
    print("\n" + "=" * 60)
    wa = WeightedAverage(roc_auc_score)
    wa.optimize(oof_preds, y_true)

    # 3. Hill climbing
    print("\n" + "=" * 60)
    hc = HillClimbing(roc_auc_score)
    hc.search(oof_preds, y_true)

    # 4. Post-processing
    print("\n" + "=" * 60)
    print("Post-processing Demo")
    pp = PostProcessor()
    avg_pred = np.mean(list(oof_preds.values()), axis=0)
    clipped = pp.clip_predictions(avg_pred)
    print(f"  Original range: [{avg_pred.min():.4f}, {avg_pred.max():.4f}]")
    print(f"  Clipped range:  [{clipped.min():.4f}, {clipped.max():.4f}]")
