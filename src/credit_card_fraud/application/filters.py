"""Dashboard filters and dashboard orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from credit_card_fraud.analytics.temporal import SECONDS_PER_HOUR
from credit_card_fraud.config import AMOUNT_COLUMN, TARGET_COLUMN, TIME_COLUMN


class TransactionStatus(StrEnum):
    """Transaction fraud status selectable in the dashboard."""

    ALL = "all"
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"


@dataclass(frozen=True)
class DashboardFilters:
    """Filters applicable to the dashboard subset.

    Amount values in the base's original units and time in elapsed hours
    since the first transaction (``Time / 3600``).
    """

    min_amount: float = 0.0
    max_amount: float = float("inf")
    status: TransactionStatus = TransactionStatus.ALL
    start_hour: float = 0.0
    end_hour: float = float("inf")

    def __post_init__(self) -> None:
        if self.min_amount < 0:
            raise ValueError("Minimum amount cannot be negative.")
        if self.max_amount < self.min_amount:
            raise ValueError(
                "Maximum amount must be greater than or equal to the minimum."
            )
        if self.start_hour < 0:
            raise ValueError("Start hour cannot be negative.")
        if self.end_hour < self.start_hour:
            raise ValueError(
                "End hour must be greater than or equal to the start hour."
            )


def apply_filters(
    transactions: pd.DataFrame, filters: DashboardFilters
) -> pd.DataFrame:
    """Return only the transactions that match the given filters."""
    mask = transactions[AMOUNT_COLUMN].between(filters.min_amount, filters.max_amount)
    mask &= transactions[TIME_COLUMN].between(
        filters.start_hour * SECONDS_PER_HOUR,
        filters.end_hour * SECONDS_PER_HOUR,
    )

    if filters.status is TransactionStatus.FRAUD:
        mask &= transactions[TARGET_COLUMN] == 1
    elif filters.status is TransactionStatus.LEGITIMATE:
        mask &= transactions[TARGET_COLUMN] == 0

    return transactions.loc[mask]
