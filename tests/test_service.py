"""Service tests consumed by the dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_card_fraud.application.filters import (
    DashboardFilters,
    TransactionStatus,
)
from credit_card_fraud.application.service import DashboardService, create_service
from credit_card_fraud.data.audit import save_prepared


def test_service_limits(synthetic_transactions: pd.DataFrame) -> None:
    service = DashboardService(synthetic_transactions)
    limits = service.limits

    assert limits["min_amount"] == 0.0
    assert limits["max_amount"] == synthetic_transactions["Amount"].max()
    assert limits["min_hour"] == 0.0
    assert limits["max_hour"] == pytest.approx(
        synthetic_transactions["Time"].max() / 3_600
    )


def test_service_overview(synthetic_transactions: pd.DataFrame) -> None:
    service = DashboardService(synthetic_transactions)
    view = service.overview(DashboardFilters())

    assert view["empty_data"] is False
    assert view["summary"]["total_transactions"] == 400
    assert view["summary"]["total_frauds"] == 12
    assert len(view["distribution"]) == 2


def test_service_filter_and_empty_subset(
    synthetic_transactions: pd.DataFrame,
) -> None:
    service = DashboardService(synthetic_transactions)
    filters = DashboardFilters(status=TransactionStatus.FRAUD)
    view = service.overview(filters)

    assert view["summary"]["total_transactions"] == 12
    assert service.value_bands(filters)["frauds"].sum() == 12

    empty_filters = DashboardFilters(min_amount=1_000_000)
    assert service.overview(empty_filters)["empty_data"] is True
    assert service.random_sample(empty_filters).empty


def test_service_aggregations(synthetic_transactions: pd.DataFrame) -> None:
    service = DashboardService(synthetic_transactions)
    filters = DashboardFilters()

    series = service.evolution_over_time(filters)
    heatmap = service.heatmap(filters)
    correlations = service.correlations(filters)

    assert series["frauds"].sum() == 12
    assert heatmap.to_numpy().sum() == 12
    assert not correlations.empty


def test_service_full_view(synthetic_transactions: pd.DataFrame) -> None:
    service = DashboardService(synthetic_transactions)
    filters = DashboardFilters()
    view = service.full_view(filters)

    assert set(view) >= {
        "data",
        "summary",
        "distribution",
        "bands",
        "series",
        "heatmap",
        "correlations",
        "empty_data",
    }
    assert view["summary"]["total_transactions"] == 400
    assert view["empty_data"] is False
    assert view["series"]["frauds"].sum() == 12


def test_service_model_metrics(synthetic_transactions: pd.DataFrame) -> None:
    service = DashboardService(synthetic_transactions)
    result = service.model_metrics()
    # The file may or may not exist in the environment; the method must not break.
    assert result is None or isinstance(result, dict)


def test_create_service_from_parquet(
    tmp_path: Path, synthetic_transactions: pd.DataFrame
) -> None:
    path = tmp_path / "transactions.parquet"
    save_prepared(synthetic_transactions, path)

    service = create_service(path)
    assert service.limits["max_amount"] == synthetic_transactions["Amount"].max()


def test_create_service_without_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_service(tmp_path / "nonexistent.parquet")
