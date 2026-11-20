"""Analysis of fraud occurrence by value band (OE-2)."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.config import AMOUNT_COLUMN, TARGET_COLUMN

VALUE_BINS = [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, float("inf")]
VALUE_LABELS = [
    "0-10",
    "10-50",
    "50-100",
    "100-250",
    "250-500",
    "500-1,000",
    "1,000+",
]
VALUE_BAND_COLUMN = "value_band"


def add_value_band(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create the categorical value band column without altering the original base."""
    data = transactions.copy()
    data[VALUE_BAND_COLUMN] = pd.cut(
        data[AMOUNT_COLUMN],
        bins=VALUE_BINS,
        labels=VALUE_LABELS,
        right=False,
    )
    return data


def frauds_by_value_band(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions, frauds, fraud rate, and volume by value band."""
    if transactions.empty:
        return pd.DataFrame(
            {
                "value_band": pd.Series(dtype="string"),
                "transactions": pd.Series(dtype="int64"),
                "frauds": pd.Series(dtype="int64"),
                "fraud_rate": pd.Series(dtype="float64"),
                "total_amount": pd.Series(dtype="float64"),
            }
        )

    data = add_value_band(transactions)
    grouped = data.groupby(VALUE_BAND_COLUMN, observed=False).agg(
        transactions=(TARGET_COLUMN, "size"),
        frauds=(TARGET_COLUMN, "sum"),
        total_amount=(AMOUNT_COLUMN, "sum"),
    )
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"].where(
        grouped["transactions"] > 0
    )

    table = grouped.reset_index()
    table[VALUE_BAND_COLUMN] = table[VALUE_BAND_COLUMN].astype("string")
    return table[
        [
            VALUE_BAND_COLUMN,
            "transactions",
            "frauds",
            "fraud_rate",
            "total_amount",
        ]
    ]
