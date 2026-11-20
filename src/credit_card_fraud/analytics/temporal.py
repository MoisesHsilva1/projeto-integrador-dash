"""Temporal analysis (OE-3).

``Time`` is the number of seconds since the first transaction in the base.
There is no absolute date/time nor a documented midnight origin. Therefore, all
analyses use relative blocks: elapsed hour, relative day (1 and 2), and hour
within the relative 24-hour block (D-13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_card_fraud.config import AMOUNT_COLUMN, TARGET_COLUMN, TIME_COLUMN

SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24
ELAPSED_HOUR_COLUMN = "elapsed_hour"
RELATIVE_DAY_COLUMN = "relative_day"
RELATIVE_DAY_HOUR_COLUMN = "relative_day_hour"


def add_relative_time(transactions: pd.DataFrame) -> pd.DataFrame:
    """Derive relative time blocks from ``Time``."""
    data = transactions.copy()
    hours = np.floor(data[TIME_COLUMN] / SECONDS_PER_HOUR).astype("int64")
    data[ELAPSED_HOUR_COLUMN] = hours
    data[RELATIVE_DAY_COLUMN] = (hours // HOURS_PER_DAY + 1).astype("int64")
    data[RELATIVE_DAY_HOUR_COLUMN] = (hours % HOURS_PER_DAY).astype("int64")
    return data


def temporal_evolution(
    transactions: pd.DataFrame, *, hour_granularity: int = 1
) -> pd.DataFrame:
    """Series of transactions, frauds, and fraud rate per hour block."""
    if hour_granularity < 1:
        raise ValueError("Granularity must be at least 1 hour.")

    columns = [ELAPSED_HOUR_COLUMN, "transactions", "frauds", "fraud_rate"]
    if transactions.empty:
        return pd.DataFrame(columns=columns)

    data = add_relative_time(transactions)
    if hour_granularity > 1:
        data[ELAPSED_HOUR_COLUMN] = (
            data[ELAPSED_HOUR_COLUMN] // hour_granularity * hour_granularity
        )

    grouped = data.groupby(ELAPSED_HOUR_COLUMN, observed=True).agg(
        transactions=(TARGET_COLUMN, "size"),
        frauds=(TARGET_COLUMN, "sum"),
        total_amount=(AMOUNT_COLUMN, "sum"),
    )
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"].where(
        grouped["transactions"] > 0
    )

    start = int(grouped.index.min())
    end = int(grouped.index.max())
    index = range(start, end + hour_granularity, hour_granularity)
    grouped = grouped.reindex(index, fill_value=0)
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"].where(
        grouped["transactions"] > 0
    )

    return grouped.reset_index()[
        [
            ELAPSED_HOUR_COLUMN,
            "transactions",
            "frauds",
            "fraud_rate",
            "total_amount",
        ]
    ]


def temporal_heatmap(transactions: pd.DataFrame) -> pd.DataFrame:
    """Relative day x 24-hour block hour matrix with fraud counts."""
    if transactions.empty:
        return pd.DataFrame()

    data = add_relative_time(transactions)
    matrix = (
        data.groupby([RELATIVE_DAY_COLUMN, RELATIVE_DAY_HOUR_COLUMN], observed=True)[
            TARGET_COLUMN
        ]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=range(HOURS_PER_DAY), fill_value=0)
    )
    matrix.index.name = RELATIVE_DAY_COLUMN
    matrix.columns.name = RELATIVE_DAY_HOUR_COLUMN
    return matrix.astype("int64")
