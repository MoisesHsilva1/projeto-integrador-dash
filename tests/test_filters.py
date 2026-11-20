"""Dashboard filter tests."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_card_fraud.application.filters import (
    DashboardFilters,
    TransactionStatus,
    apply_filters,
)


def test_filter_by_amount(synthetic_transactions: pd.DataFrame) -> None:
    filters = DashboardFilters(min_amount=50, max_amount=100)
    subset = apply_filters(synthetic_transactions, filters)

    assert not subset.empty
    assert subset["Amount"].between(50, 100).all()


def test_filter_by_status(synthetic_transactions: pd.DataFrame) -> None:
    fraud_transactions = apply_filters(
        synthetic_transactions, DashboardFilters(status=TransactionStatus.FRAUD)
    )
    legitimate_transactions = apply_filters(
        synthetic_transactions, DashboardFilters(status=TransactionStatus.LEGITIMATE)
    )

    assert len(fraud_transactions) == 12
    assert fraud_transactions["Class"].eq(1).all()
    assert len(legitimate_transactions) == 388
    assert legitimate_transactions["Class"].eq(0).all()


def test_filter_by_time(synthetic_transactions: pd.DataFrame) -> None:
    filters = DashboardFilters(start_hour=0, end_hour=1)
    subset = apply_filters(synthetic_transactions, filters)

    assert subset["Time"].between(0, 3_600).all()
    assert (subset["Time"] > 3_600).sum() == 0


def test_invalid_filters() -> None:
    with pytest.raises(ValueError, match="Maximum amount"):
        DashboardFilters(min_amount=10, max_amount=5)
    with pytest.raises(ValueError, match="negative"):
        DashboardFilters(min_amount=-1)
    with pytest.raises(ValueError, match="End hour"):
        DashboardFilters(start_hour=10, end_hour=5)
