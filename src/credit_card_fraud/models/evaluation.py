"""Evaluation metrics, interpretation, and uncertainty of the model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_card_fraud.config import DEFAULT_THRESHOLD, TARGET_RECALLS


def _as_arrays(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize labels and probabilities for internal calculations."""
    return (
        np.asarray(y_true, dtype="int64"),
        np.asarray(probabilities, dtype="float64"),
    )


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate the plan metrics for an explicit decision threshold.

    Besides the historical metrics, includes FPR, FNR, specificity, and
    prevalence, which describe the operational cost of false alerts and missed
    frauds.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    y_pred = (probabilities >= threshold).astype("int64")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()

    total_negatives = int((y_true == 0).sum())
    total_positives = int((y_true == 1).sum())

    return {
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "false_positive_rate": (
            float(false_positives / total_negatives) if total_negatives else 0.0
        ),
        "false_negative_rate": (
            float(false_negatives / total_positives) if total_positives else 0.0
        ),
        "specificity": (
            float(true_negatives / total_negatives) if total_negatives else 0.0
        ),
        "prevalence": (float(total_positives / len(y_true)) if len(y_true) else 0.0),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "total_negatives": total_negatives,
        "total_positives": total_positives,
    }


def _row_type_label(row_type: str, threshold: float, chosen_threshold: float) -> str:
    """Return the semantic label of a threshold table row."""
    if threshold == float(DEFAULT_THRESHOLD):
        return "default_0.5"
    if threshold == float(chosen_threshold):
        return "chosen"
    return row_type


def find_best_f1_threshold(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """Find threshold and F1 score maximizing F1 along the precision-recall curve."""
    if len(thresholds) == 0:
        return float(DEFAULT_THRESHOLD), 0.0
    f1_curve = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    if not f1_curve.size:
        return float(DEFAULT_THRESHOLD), 0.0
    position = int(np.argmax(f1_curve))
    return float(thresholds[position]), float(f1_curve[position])


def threshold_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    chosen_threshold: float,
    *,
    target_recalls: tuple[float, ...] = TARGET_RECALLS,
) -> list[dict[str, float | int | str]]:
    """Compare the default threshold, the chosen one, and recall-oriented thresholds.

    The recall-target thresholds are obtained from the precision-recall curve
    on the given set (train/holdout depending on the caller), without assuming
    the 0.5 threshold is adequate.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    rows: list[dict[str, float | int | str]] = []
    seen: set[float] = set()

    def add(row_type: str, threshold: float) -> None:
        threshold = float(threshold)
        if threshold in seen:
            return
        seen.add(threshold)
        metrics = classification_metrics(y_true, probabilities, threshold)
        rows.append(
            {
                "type": _row_type_label(row_type, threshold, chosen_threshold),
                **metrics,
            }
        )

    add("default_0.5", DEFAULT_THRESHOLD)
    add("chosen", chosen_threshold)

    best_threshold, _ = find_best_f1_threshold(precision, recall, thresholds)
    add("max_f1", best_threshold)

    for target in target_recalls:
        candidates = [
            float(threshold)
            for threshold, revocation in zip(thresholds, recall[:-1], strict=True)
            if revocation >= target
        ]
        if candidates:
            add(f"recall_{int(target * 100)}", min(candidates))

    return sorted(rows, key=lambda row: float(row["threshold"]))


def value_impact(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    amounts: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Summarize the impact in original units of FP and FN.

    Only the original ``Amount`` from the base is used. No monetary cost is
    assumed; values are presented in the dataset's original unit.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    amounts = np.asarray(amounts, dtype="float64")
    if len(amounts) != len(y_true):
        raise ValueError("The amount of values must match the target.")

    y_pred = (probabilities >= threshold).astype("int64")
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    total_amount = float(amounts.sum())
    fraud_amount = float(amounts[y_true == 1].sum())
    fp_amount = float(amounts[fp_mask].sum())
    fn_amount = float(amounts[fn_mask].sum())

    metrics = classification_metrics(y_true, probabilities, threshold)

    return {
        "threshold": float(threshold),
        "total_amount": total_amount,
        "fraud_amount": fraud_amount,
        "false_positive_amount": fp_amount,
        "false_negative_amount": fn_amount,
        "undetected_fraud_amount_ratio": (
            float(fn_amount / fraud_amount) if fraud_amount else 0.0
        ),
        "false_positive_amount_ratio": (
            float(fp_amount / total_amount) if total_amount else 0.0
        ),
        "false_positives": int(metrics["false_positives"]),
        "false_negatives": int(metrics["false_negatives"]),
        "true_positives": int(metrics["true_positives"]),
        "true_negatives": int(metrics["true_negatives"]),
    }


def feature_importances(model, names: list[str]) -> pd.Series:
    """Importance of the final estimator variables, in decreasing order.

    For trees, uses ``feature_importances_``; for linear models, uses the
    absolute value of the coefficients. Attributes are anonymized by PCA, so
    reading must be done with caution (R-06).
    """
    estimator = model.named_steps["classifier"]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype="float64")
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype="float64")).ravel()
    else:
        raise TypeError(
            "The final estimator does not expose feature_importances_ nor coef_."
        )

    if len(values) != len(names):
        raise ValueError(
            "The number of importances does not match the number of attributes."
        )

    return pd.Series(values, index=names, name="importance").sort_values(
        ascending=False
    )


def permutation_importance(
    model,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    n_repeats: int = 10,
    sample: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance on the holdout, with repeats and optional sample.

    Permutation is more robust than impurity importance, but still measures
    model association, not causality. If ``sample`` is given, a stratified
    sample is used to limit computational cost.
    """
    from sklearn.inspection import permutation_importance as _permutation_importance

    features = features.reset_index(drop=True)
    target = pd.Series(np.asarray(target), name="target").reset_index(drop=True)

    if sample is not None and len(features) > sample:
        indices = stratified_sample_indices(
            original_target=target, size=sample, seed=seed
        )
        features = features.iloc[indices].reset_index(drop=True)
        target = target.iloc[indices].reset_index(drop=True)

    result = _permutation_importance(
        model,
        features,
        target,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="average_precision",
        n_jobs=-1,
    )

    return pd.DataFrame(
        {
            "mean": result.importances_mean,
            "std": result.importances_std,
        },
        index=list(features.columns),
    ).sort_values("mean", ascending=False)


def stratified_sample_indices(
    *, original_target: pd.Series, size: int, seed: int
) -> np.ndarray:
    """Indices of a stable stratified sample."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(original_target))
    selected, _ = train_test_split(
        indices,
        train_size=size,
        stratify=original_target,
        random_state=seed,
    )
    return np.sort(selected)


def error_summary(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    amounts: pd.Series | np.ndarray,
    threshold: float,
    *,
    index: pd.Index | None = None,
    max_cases: int = 20,
) -> dict[str, object]:
    """Summarized inspection of false positives and false negatives.

    Keeps only counts, value statistics, and the largest cases, without
    reproducing the full base in reports.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    amounts = np.asarray(amounts, dtype="float64")
    if len(amounts) != len(y_true):
        raise ValueError("The amount of values must match the target.")
    if index is None:
        index = pd.RangeIndex(len(y_true))

    y_pred = (probabilities >= threshold).astype("int64")
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    def details(mask: np.ndarray) -> list[dict[str, object]]:
        positions = np.flatnonzero(mask)
        if not len(positions):
            return []
        ordered = positions[np.argsort(amounts[positions])[::-1]][:max_cases]
        return [
            {
                "index": str(index[int(position)]),
                "amount": float(amounts[int(position)]),
                "probability": float(probabilities[int(position)]),
            }
            for position in ordered
        ]

    def value_summary(mask: np.ndarray) -> dict[str, float]:
        selected = amounts[mask]
        if not len(selected):
            return {
                "total": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0,
            }
        return {
            "total": float(selected.sum()),
            "mean": float(selected.mean()),
            "median": float(np.median(selected)),
            "max": float(selected.max()),
        }

    return {
        "threshold": float(threshold),
        "false_positives": int(fp_mask.sum()),
        "false_negatives": int(fn_mask.sum()),
        "false_positive_amount": value_summary(fp_mask),
        "false_negative_amount": value_summary(fn_mask),
        "largest_false_positives": details(fp_mask),
        "largest_false_negatives": details(fn_mask),
    }


def confidence_intervals(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    n_samples: int = 1_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap confidence intervals of the metrics.

    Resamples already computed predictions; does not retrain models.
    """
    y_true, probabilities = _as_arrays(y_true, probabilities)
    generator = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {
        "auprc": [],
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "accuracy": [],
    }

    for _ in range(n_samples):
        indices = generator.integers(0, len(y_true), size=len(y_true))
        y_sample = y_true[indices]
        p_sample = probabilities[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        y_pred = (p_sample >= threshold).astype("int64")
        metrics["auprc"].append(float(average_precision_score(y_sample, p_sample)))
        metrics["roc_auc"].append(float(roc_auc_score(y_sample, p_sample)))
        metrics["precision"].append(
            float(precision_score(y_sample, y_pred, zero_division=0))
        )
        metrics["recall"].append(float(recall_score(y_sample, y_pred, zero_division=0)))
        metrics["f1"].append(float(f1_score(y_sample, y_pred, zero_division=0)))
        metrics["accuracy"].append(float(accuracy_score(y_sample, y_pred)))

    return {
        name: {
            "lower": float(np.percentile(values, 2.5)) if values else 0.0,
            "upper": float(np.percentile(values, 97.5)) if values else 0.0,
        }
        for name, values in metrics.items()
    }
