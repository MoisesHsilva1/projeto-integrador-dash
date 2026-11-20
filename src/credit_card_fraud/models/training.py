"""Training, model selection, and decision threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_card_fraud.config import N_FOLDS, SEED
from credit_card_fraud.models.evaluation import (
    classification_metrics,
    find_best_f1_threshold,
)
from credit_card_fraud.models.persistence import (  # noqa: F401
    load_model,
    save_model,
)

VALIDATION_METRICS = {
    "auprc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

MODEL_HYPERPARAMETERS: dict[str, dict[str, object]] = {
    "Logistic Regression (baseline)": {
        "max_iter": 2_000,
        "class_weight": "balanced",
    },
    "Random Forest": {
        "n_estimators": 150,
        "max_depth": 16,
        "min_samples_leaf": 10,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    "Gradient Boosting (histograms)": {
        "max_iter": 200,
        "learning_rate": 0.1,
        "class_weight": "balanced",
    },
}


def create_models(seed: int = SEED) -> dict[str, Pipeline]:
    """Comparable models: linear baseline and tree candidates.

    Hyperparameters are centralized in ``MODEL_HYPERPARAMETERS`` so they can
    be recorded in the artifact and reused in experiments.
    """
    parameters = {name: dict(values) for name, values in MODEL_HYPERPARAMETERS.items()}
    parameters["Logistic Regression (baseline)"]["random_state"] = seed
    parameters["Random Forest"]["random_state"] = seed
    parameters["Gradient Boosting (histograms)"]["random_state"] = seed

    return {
        "Logistic Regression (baseline)": Pipeline(
            [
                ("standardization", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(**parameters["Logistic Regression (baseline)"]),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(**parameters["Random Forest"]),
                ),
            ]
        ),
        "Gradient Boosting (histograms)": Pipeline(
            [
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        **parameters["Gradient Boosting (histograms)"]
                    ),
                ),
            ]
        ),
    }


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    folds: int = N_FOLDS,
    seed: int = SEED,
    return_probabilities: bool = False,
) -> (
    dict[str, dict[str, float]]
    | tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]
):
    """Compare models by stratified cross-validation on training data."""
    validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: dict[str, dict[str, float]] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}

    for name, model in create_models(seed).items():
        scores = cross_validate(
            model,
            features,
            target,
            cv=validation,
            scoring=VALIDATION_METRICS,
            return_estimator=return_probabilities,
            n_jobs=1,
        )
        result: dict[str, float] = {}
        for metric in VALIDATION_METRICS:
            values = scores[f"test_{metric}"]
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))
        result["train_time_seconds"] = float(np.mean(scores["fit_time"]))
        results[name] = result

        if return_probabilities:
            probabilities = np.empty(len(target), dtype="float64")
            for estimator, (_, val_idx) in zip(
                scores["estimator"], validation.split(features, target), strict=True
            ):
                probabilities[val_idx] = estimator.predict_proba(
                    features.iloc[val_idx]
                )[:, 1]
            probabilities_by_model[name] = probabilities

    if return_probabilities:
        return results, probabilities_by_model
    return results


def choose_best_model(results: dict[str, dict[str, float]]) -> str:
    """Choose the model with the highest mean AUPRC (R-05)."""
    if not results:
        raise ValueError("There are no cross-validation results to compare.")
    return max(results, key=lambda name: results[name]["auprc_mean"])


def out_of_fold_probabilities(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    folds: int = N_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """Out-of-fold probabilities on the training set, for the threshold."""
    validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(
        model,
        features,
        target,
        cv=validation,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def choose_threshold(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Choose the threshold that maximizes F1 on the given probabilities.

    The threshold is chosen on out-of-fold training probabilities, never on
    the holdout, to avoid contaminating the final evaluation.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    return find_best_f1_threshold(precision, recall, thresholds)


def train_and_evaluate(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    *,
    folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict[str, object]:
    """Compare models, choose the best, set the threshold, and evaluate on holdout."""
    results, out_of_fold_preds = compare_models(
        features_train, target_train, folds=folds, seed=seed, return_probabilities=True
    )
    best_name = choose_best_model(results)
    best_model = create_models(seed)[best_name]

    train_probabilities = out_of_fold_preds[best_name]
    threshold, train_f1 = choose_threshold(target_train, train_probabilities)

    best_model.fit(features_train, target_train)
    test_probabilities = best_model.predict_proba(features_test)[:, 1]
    test_metrics = classification_metrics(target_test, test_probabilities, threshold)

    # Baseline evaluated on the same holdout, with the threshold chosen on training.
    baseline_name = "Logistic Regression (baseline)"
    baseline_model = create_models(seed)[baseline_name]
    baseline_train_probabilities = out_of_fold_preds[baseline_name]
    baseline_threshold, baseline_f1 = choose_threshold(
        target_train, baseline_train_probabilities
    )
    baseline_model.fit(features_train, target_train)
    baseline_test_probabilities = baseline_model.predict_proba(features_test)[:, 1]
    baseline_metrics = classification_metrics(
        target_test, baseline_test_probabilities, baseline_threshold
    )

    return {
        "cross_validation": results,
        "best_model": best_name,
        "decision_threshold": threshold,
        "f1_out_of_fold_train": train_f1,
        "holdout_metrics": test_metrics,
        "model": best_model,
        "test_probabilities": test_probabilities,
        "baseline_holdout": {
            "name": baseline_name,
            "decision_threshold": baseline_threshold,
            "f1_out_of_fold_train": baseline_f1,
            "holdout_metrics": baseline_metrics,
            "test_probabilities": baseline_test_probabilities,
        },
    }
