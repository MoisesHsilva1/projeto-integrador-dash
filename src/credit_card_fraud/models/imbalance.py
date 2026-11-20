"""Comparison of class imbalance strategies (offline track).

The strategy used in the pipeline is ``class_weight="balanced"`` (D-19), without
resampling. This module allows comparing it with random resampling using the
same stratified cross-validation methodology.

Mandatory rules:
- resampling is applied **only** to the training folds of cross-validation;
- validation folds are never resampled nor used to fit anything;
- all strategies use the same model, seed, and folds.

SMOTE is not used: it would require the ``imbalanced-learn`` dependency and
decision D-19 records that class weights were preferred because they are
explainable and do not introduce an extra dependency.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from credit_card_fraud.config import N_FOLDS, SEED

ResampleMode = Literal["under", "over"]

STRATEGIES = (
    "Class weights (current)",
    "No class weights",
    "Random undersampling",
    "Random oversampling",
)


def resample_random(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    mode: ResampleMode,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Randomly resample to balance the classes.

    ``under`` reduces the majority class to the minority size; ``over``
    replicates the minority until the majority size. Row order is shuffled.
    If any class is absent, returns unchanged copies.
    """
    indices = np.arange(len(target))
    values = np.asarray(target)
    positives = indices[values == 1]
    negatives = indices[values == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return features.copy(), target.copy()

    generator = np.random.default_rng(seed)
    if mode == "under":
        majority = generator.choice(negatives, size=len(positives), replace=False)
        chosen = np.concatenate([positives, majority])
    elif mode == "over":
        reproduction = generator.choice(
            positives, size=len(negatives) - len(positives), replace=True
        )
        chosen = np.concatenate([negatives, positives, reproduction])
    else:
        raise ValueError(f"Unknown resampling mode: {mode!r}.")

    chosen = generator.permutation(chosen)
    resampled_features = features.iloc[chosen].reset_index(drop=True)
    resampled_target = target.iloc[chosen].reset_index(drop=True)
    return resampled_features, resampled_target


def apply_class_weight(model, class_weight: str | None) -> bool:
    """Adjust ``class_weight`` of the final estimator when the pipeline exposes it."""
    try:
        model.set_params(classifier__class_weight=class_weight)
        return True
    except (ValueError, AttributeError, KeyError):
        return False


def compare_strategies(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    *,
    model=None,
    folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    """Compare imbalance strategies by cross-validation.

    Receives only training data. For each strategy, calculates AUPRC on each
    validation fold (never resampled) and returns mean and standard deviation.
    """
    from credit_card_fraud.models.training import create_models

    base_model = model if model is not None else create_models(seed)["Random Forest"]
    validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    configurations: dict[str, tuple[str | None, str | None]] = {
        "Class weights (current)": (None, "balanced"),
        "No class weights": (None, None),
        "Random undersampling": ("under", None),
        "Random oversampling": ("over", None),
    }

    results: dict[str, dict[str, float]] = {}
    for name in STRATEGIES:
        scores: list[float] = []
        times: list[float] = []
        mode, class_weight = configurations[name]

        for train_index, validation_index in validation.split(
            features_train, target_train
        ):
            fold_features = features_train.iloc[train_index]
            fold_target = target_train.iloc[train_index]

            if mode is not None:
                fold_features, fold_target = resample_random(
                    fold_features, fold_target, mode=mode, seed=seed
                )

            candidate = clone(base_model)
            apply_class_weight(candidate, class_weight)

            start = time.perf_counter()
            candidate.fit(fold_features, fold_target)
            times.append(time.perf_counter() - start)

            probabilities = candidate.predict_proba(
                features_train.iloc[validation_index]
            )[:, 1]
            actuals = target_train.iloc[validation_index]
            scores.append(float(average_precision_score(actuals, probabilities)))

        results[name] = {
            "auprc_mean": float(np.mean(scores)),
            "auprc_std": float(np.std(scores)),
            "train_time_seconds": float(np.mean(times)),
        }

    return results
