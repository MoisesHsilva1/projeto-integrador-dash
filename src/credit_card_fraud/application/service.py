"""Services consumed by the presentation layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_card_fraud.analytics.correlation import correlation_with_target
from credit_card_fraud.analytics.summary import class_distribution, general_summary
from credit_card_fraud.analytics.temporal import (
    SECONDS_PER_HOUR,
    temporal_evolution,
    temporal_heatmap,
)
from credit_card_fraud.analytics.value import frauds_by_value_band
from credit_card_fraud.application.filters import DashboardFilters, apply_filters
from credit_card_fraud.config import AMOUNT_COLUMN, SEED, TIME_COLUMN
from credit_card_fraud.data.loader import (
    load_errors,
    load_importances,
    load_metrics,
    load_prepared,
)
from credit_card_fraud.data.schema import validate_schema


class DashboardService:
    """Orchestrates filters and aggregations for the dashboard.

    The presentation only talks to this service; metrics keep being defined in
    the Analytics/Domain layer.
    """

    def __init__(self, transactions: pd.DataFrame) -> None:
        validate_schema(transactions)
        self._transactions = transactions

    @property
    def limits(self) -> dict[str, float]:
        """Limits available for the filters, in original units and hours."""
        if self._transactions.empty:
            return {
                "min_amount": 0.0,
                "max_amount": 0.0,
                "min_hour": 0.0,
                "max_hour": 0.0,
            }
        hours = self._transactions[TIME_COLUMN] / SECONDS_PER_HOUR
        return {
            "min_amount": float(self._transactions[AMOUNT_COLUMN].min()),
            "max_amount": float(self._transactions[AMOUNT_COLUMN].max()),
            "min_hour": 0.0,
            "max_hour": float(hours.max()),
        }

    def subset(self, filters: DashboardFilters) -> pd.DataFrame:
        return apply_filters(self._transactions, filters)

    def overview(self, filters: DashboardFilters) -> dict[str, object]:
        data = self.subset(filters)
        return {
            "summary": general_summary(data),
            "distribution": class_distribution(data),
            "empty_data": data.empty,
        }

    def value_bands(self, filters: DashboardFilters) -> pd.DataFrame:
        return frauds_by_value_band(self.subset(filters))

    def evolution_over_time(
        self, filters: DashboardFilters, *, hour_granularity: int = 1
    ) -> pd.DataFrame:
        return temporal_evolution(
            self.subset(filters), hour_granularity=hour_granularity
        )

    def heatmap(self, filters: DashboardFilters) -> pd.DataFrame:
        return temporal_heatmap(self.subset(filters))

    def correlations(self, filters: DashboardFilters) -> pd.Series:
        return correlation_with_target(self.subset(filters))

    def full_view(self, filters: DashboardFilters) -> dict[str, object]:
        """Calculate all aggregations from a single subset.

        Avoids recomputing the filter over the base for each visualization.
        """
        data = self.subset(filters)
        return {
            "data": data,
            "summary": general_summary(data),
            "distribution": class_distribution(data),
            "bands": frauds_by_value_band(data),
            "series": temporal_evolution(data),
            "heatmap": temporal_heatmap(data),
            "correlations": correlation_with_target(data),
            "empty_data": data.empty,
        }

    def random_sample(self, filters: DashboardFilters, size: int = 50) -> pd.DataFrame:
        """Reproducible sample for verification in the interface."""
        data = self.subset(filters)
        if data.empty:
            return data
        return data.sample(n=min(size, len(data)), random_state=SEED)

    def model_metrics(self) -> dict | None:
        """Offline model metrics, when available."""
        return load_metrics()

    def model_importances(self) -> dict | None:
        """Permutation importances of the saved model, when available."""
        return load_importances()

    def model_errors(self) -> dict | None:
        """False positives and false negatives summary, when available."""
        return load_errors()


def create_service(path: Path | str | None = None) -> DashboardService:
    """Create the service from the prepared base (Data Access)."""
    return DashboardService(load_prepared(path))
