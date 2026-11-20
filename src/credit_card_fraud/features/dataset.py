"""Feature matrix construction and train/test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_card_fraud.config import (
    MODEL_COLUMNS,
    SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from credit_card_fraud.data.schema import validate_schema


def build_matrix(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split attributes (Time, V1–V28, Amount) and target (Class)."""
    validate_schema(transactions)
    features = transactions[MODEL_COLUMNS].astype("float64")
    target = transactions[TARGET_COLUMN].astype("int8")
    return features, target


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train and test in a stratified way, before any fitting."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=seed,
    )
