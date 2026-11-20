"""Correlation between attributes and the fraud label."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.config import (
    AMOUNT_COLUMN,
    PCA_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)


def correlation_with_target(
    transactions: pd.DataFrame, columns: list[str] | None = None
) -> pd.Series:
    """Pearson correlation of each attribute with ``Class``, by magnitude."""
    if transactions.empty:
        return pd.Series(dtype="float64", name="correlation")

    selected = columns or [*PCA_COLUMNS, AMOUNT_COLUMN, TIME_COLUMN]
    correlations = transactions[selected + [TARGET_COLUMN]].corr()[TARGET_COLUMN]
    correlations = correlations.drop(labels=[TARGET_COLUMN], errors="ignore")
    correlations = correlations.dropna()
    correlations.name = "correlation"
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)
