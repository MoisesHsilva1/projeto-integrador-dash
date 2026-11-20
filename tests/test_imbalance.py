"""Class imbalance strategy comparison tests."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.models.imbalance import (
    STRATEGIES,
    compare_strategies,
    resample_random,
)


def test_undersampling_equals_classes(synthetic_transactions: pd.DataFrame) -> None:
    features = synthetic_transactions.drop(columns=["Class"])
    target = synthetic_transactions["Class"]

    resampled_features, resampled_target = resample_random(
        features, target, mode="under"
    )

    counts = resampled_target.value_counts()
    assert counts[0] == counts[1]
    assert len(resampled_target) == 2 * int(target.sum())


def test_oversampling_equals_classes(synthetic_transactions: pd.DataFrame) -> None:
    features = synthetic_transactions.drop(columns=["Class"])
    target = synthetic_transactions["Class"]

    resampled_features, resampled_target = resample_random(
        features, target, mode="over"
    )

    counts = resampled_target.value_counts()
    assert counts[0] == counts[1]
    assert len(resampled_target) == 2 * int((target == 0).sum())


def test_resampling_reproducible(synthetic_transactions: pd.DataFrame) -> None:
    features = synthetic_transactions.drop(columns=["Class"])
    target = synthetic_transactions["Class"]

    first = resample_random(features, target, mode="under", seed=42)
    second = resample_random(features, target, mode="under", seed=42)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[1], second[1])


def test_invalid_resampling_mode(synthetic_transactions: pd.DataFrame) -> None:
    import pytest

    features = synthetic_transactions.drop(columns=["Class"])
    target = synthetic_transactions["Class"]

    with pytest.raises(ValueError, match="Unknown"):
        resample_random(features, target, mode="invalid")  # type: ignore[arg-type]


def test_compare_strategies(synthetic_transactions: pd.DataFrame) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    features = synthetic_transactions.drop(columns=["Class"]).astype("float64")
    target = synthetic_transactions["Class"].astype("int8")
    model = Pipeline(
        [("classifier", RandomForestClassifier(n_estimators=10, random_state=42))]
    )

    results = compare_strategies(features, target, model=model, folds=2)

    assert tuple(results) == STRATEGIES
    for values in results.values():
        assert 0.0 <= values["auprc_mean"] <= 1.0
        assert values["auprc_std"] >= 0.0
        assert values["train_time_seconds"] >= 0.0
