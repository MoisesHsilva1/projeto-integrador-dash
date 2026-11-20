"""Robustness, explainability, and sensitivity experiments (offline track)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from credit_card_fraud.config import (
    N_EXPERIMENT_FOLDS,
    N_FOLDS,
    ROBUSTNESS_SEEDS,
    SEED,
)
from credit_card_fraud.models.evaluation import classification_metrics
from credit_card_fraud.models.imbalance import resample_random
from credit_card_fraud.models.training import choose_threshold

DAY_LIMIT = 86_400.0


def temporal_validation(
    features: pd.DataFrame,
    target: pd.Series,
    model,
    *,
    folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict[str, object]:
    """Sensitivity: train on relative day 1 and test on relative day 2.

    It is a secondary validation. It does not replace the main stratified
    split nor take part in model selection.
    """
    times = features["Time"].to_numpy()
    train_mask = times < DAY_LIMIT
    if train_mask.all() or not train_mask.any():
        raise ValueError("The base does not contain the two required relative days.")

    train = features.loc[train_mask].reset_index(drop=True)
    test = features.loc[~train_mask].reset_index(drop=True)
    target_train = target.loc[train_mask].reset_index(drop=True)
    target_test = target.loc[~train_mask].reset_index(drop=True)

    out_of_fold_probabilities = cross_val_predict(
        clone(model),
        train,
        target_train,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, f1_out_of_fold = choose_threshold(
        target_train, out_of_fold_probabilities
    )

    fitted_model = clone(model).fit(train, target_train)
    test_probabilities = fitted_model.predict_proba(test)[:, 1]

    return {
        "criterion": "Time < 86400 s for training; Time >= 86400 s for testing",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_frauds": int(target_train.sum()),
        "test_frauds": int(target_test.sum()),
        "decision_threshold": float(threshold),
        "f1_out_of_fold_train": float(f1_out_of_fold),
        "metrics": classification_metrics(target_test, test_probabilities, threshold),
    }


def seed_stability(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: Callable[[int], Any],
    *,
    seeds: tuple[int, ...] = ROBUSTNESS_SEEDS,
    folds: int = N_EXPERIMENT_FOLDS,
) -> dict[str, object]:
    """AUPRC in CV repeated with different seeds, on training data."""
    results: dict[str, object] = {}
    auprc_by_seed: list[float] = []

    for seed in seeds:
        start = time.perf_counter()
        validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_validate(
            model_factory(seed),
            features,
            target,
            cv=validation,
            scoring={"auprc": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        result = {
            "auprc_mean": float(np.mean(scores["test_auprc"])),
            "auprc_std": float(np.std(scores["test_auprc"])),
            "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "train_time_seconds": float(time.perf_counter() - start),
        }
        results[str(seed)] = result
        auprc_by_seed.append(result["auprc_mean"])

    return {
        "folds": folds,
        "results": results,
        "overall_auprc_mean": float(np.mean(auprc_by_seed)),
        "auprc_amplitude": float(np.max(auprc_by_seed) - np.min(auprc_by_seed)),
    }


def hyperparameter_tuning(
    features: pd.DataFrame,
    target: pd.Series,
    model,
    param_grid: dict[str, list[Any]],
    *,
    n_iter: int = 4,
    folds: int = N_EXPERIMENT_FOLDS,
    seed: int = SEED,
) -> dict[str, object]:
    """Limited hyperparameter search with CV on training data.

    The holdout does not take part in the search. ``n_iter`` is intentionally
    small to keep the academic cost viable.
    """
    validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    start = time.perf_counter()
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="average_precision",
        cv=validation,
        random_state=seed,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(features, target)
    elapsed = time.perf_counter() - start

    candidates = [
        {
            "params": dict(params),
            "auprc_mean": float(score),
            "auprc_std": float(std),
        }
        for params, score, std in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
            strict=True,
        )
    ]
    candidates.sort(key=lambda item: item["auprc_mean"], reverse=True)

    return {
        "n_iter": int(n_iter),
        "folds": int(folds),
        "search_time_seconds": float(elapsed),
        "best_params": dict(search.best_params_),
        "auprc_mean": float(search.best_score_),
        "candidates": candidates,
        "fitted_model": search.best_estimator_,
    }


def out_of_fold_probabilities_resampled(
    features: pd.DataFrame,
    target: pd.Series,
    model,
    *,
    mode: str,
    folds: int = N_EXPERIMENT_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """Out-of-fold probabilities with resampling only in each fold's training."""
    validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.empty(len(target), dtype="float64")

    for train_index, validation_index in validation.split(features, target):
        train_features, train_target = resample_random(
            features.iloc[train_index],
            target.iloc[train_index],
            mode=mode,  # type: ignore[arg-type]
            seed=seed,
        )
        candidate = clone(model).fit(train_features, train_target)
        probabilities[validation_index] = candidate.predict_proba(
            features.iloc[validation_index]
        )[:, 1]

    return probabilities


def evaluate_resampling_holdout(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    model,
    *,
    mode: str,
    folds: int = N_EXPERIMENT_FOLDS,
    seed: int = SEED,
) -> dict[str, object]:
    """Evaluate under/over sampling on the holdout, with threshold chosen on
    training.
    """
    out_of_fold_probabilities = out_of_fold_probabilities_resampled(
        features_train,
        target_train,
        model,
        mode=mode,
        folds=folds,
        seed=seed,
    )
    threshold, f1_out_of_fold = choose_threshold(
        target_train, out_of_fold_probabilities
    )

    train_features, train_target = resample_random(
        features_train,
        target_train,
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
    )
    fitted_model = clone(model).fit(train_features, train_target)
    test_probabilities = fitted_model.predict_proba(features_test)[:, 1]

    return {
        "mode": mode,
        "folds": int(folds),
        "decision_threshold": float(threshold),
        "f1_out_of_fold_train": float(f1_out_of_fold),
        "resampled_train_rows": int(len(train_features)),
        "metrics": classification_metrics(target_test, test_probabilities, threshold),
    }
