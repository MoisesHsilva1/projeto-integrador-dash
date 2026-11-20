"""General indicators and target class distribution."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.config import AMOUNT_COLUMN, TARGET_COLUMN


def general_summary(transactions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the general indicators of the received subset.

    Returns zeros when the subset is empty, so the interface can display an
    empty state without breaking.
    """
    total = int(len(transactions))
    if total == 0:
        return {
            "total_transactions": 0,
            "total_frauds": 0,
            "total_legitimate": 0,
            "fraud_rate": 0.0,
            "total_amount": 0.0,
            "fraud_amount": 0.0,
            "legitimate_amount": 0.0,
            "median_amount": 0.0,
            "average_ticket": 0.0,
        }

    frauds = transactions[TARGET_COLUMN] == 1
    total_frauds = int(frauds.sum())
    total_amount = float(transactions[AMOUNT_COLUMN].sum())

    return {
        "total_transactions": total,
        "total_frauds": total_frauds,
        "total_legitimate": total - total_frauds,
        "fraud_rate": total_frauds / total,
        "total_amount": total_amount,
        "fraud_amount": float(transactions.loc[frauds, AMOUNT_COLUMN].sum()),
        "legitimate_amount": float(transactions.loc[~frauds, AMOUNT_COLUMN].sum()),
        "median_amount": float(transactions[AMOUNT_COLUMN].median()),
        "average_ticket": float(transactions[AMOUNT_COLUMN].mean()),
    }


def class_distribution(transactions: pd.DataFrame) -> pd.DataFrame:
    """Count legitimate and fraud transactions, with each class share."""
    total = len(transactions)
    counts = transactions[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)

    return pd.DataFrame(
        {
            "class": [0, 1],
            "transactions": [int(counts[0]), int(counts[1])],
            "participation": [
                float(counts[0] / total) if total else 0.0,
                float(counts[1] / total) if total else 0.0,
            ],
        }
    )
