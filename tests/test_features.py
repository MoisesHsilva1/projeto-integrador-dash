"""Attribute preparation and split tests."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.config import MODEL_COLUMNS, TARGET_COLUMN
from credit_card_fraud.features.dataset import build_matrix, split_train_test


def test_build_matrix(synthetic_transactions: pd.DataFrame) -> None:
    features, target = build_matrix(synthetic_transactions)

    assert list(features.columns) == MODEL_COLUMNS
    assert len(features) == len(target) == 400
    assert target.name == TARGET_COLUMN
    assert set(target.unique()) == {0, 1}
    assert not features.isna().any().any()


def test_stratified_split(synthetic_transactions: pd.DataFrame) -> None:
    features, target = build_matrix(synthetic_transactions)
    features_train, features_test, target_train, target_test = split_train_test(
        features, target, test_size=0.2
    )

    assert len(features_test) == 80
    assert len(features_train) == 320
    assert not set(features_train.index) & set(features_test.index)
    assert target_test.sum() == 2
    assert target_train.sum() == 10


def test_split_reproducible(synthetic_transactions: pd.DataFrame) -> None:
    features, target = build_matrix(synthetic_transactions)
    first = split_train_test(features, target, seed=42)
    second = split_train_test(features, target, seed=42)

    assert first[1].index.equals(second[1].index)
