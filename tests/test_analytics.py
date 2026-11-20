"""Analytics metrics and aggregation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_card_fraud.analytics.correlation import correlation_with_target
from credit_card_fraud.analytics.summary import class_distribution, general_summary
from credit_card_fraud.analytics.temporal import (
    add_relative_time,
    temporal_evolution,
    temporal_heatmap,
)
from credit_card_fraud.analytics.value import frauds_by_value_band


def test_general_summary(synthetic_transactions: pd.DataFrame) -> None:
    summary = general_summary(synthetic_transactions)

    assert summary["total_transactions"] == 400
    assert summary["total_frauds"] == 12
    assert summary["total_legitimate"] == 388
    assert summary["fraud_rate"] == 12 / 400
    assert summary["total_amount"] == synthetic_transactions["Amount"].sum()


def test_general_summary_empty() -> None:
    summary = general_summary(pd.DataFrame(columns=["Amount", "Class"]))

    assert summary["total_transactions"] == 0
    assert summary["fraud_rate"] == 0.0


def test_class_distribution(synthetic_transactions: pd.DataFrame) -> None:
    distribution = class_distribution(synthetic_transactions)

    assert list(distribution["class"]) == [0, 1]
    assert list(distribution["transactions"]) == [388, 12]
    assert distribution["participation"].sum() == 1.0


def test_value_bands_respect_limits() -> None:
    transactions = pd.DataFrame(
        {
            "Amount": [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 9_999.0],
            "Class": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    table = frauds_by_value_band(transactions)

    counts = dict(zip(table["value_band"], table["frauds"], strict=True))
    assert counts == {
        "0-10": 1,
        "10-50": 1,
        "50-100": 1,
        "100-250": 1,
        "250-500": 1,
        "500-1,000": 1,
        "1,000+": 2,
    }
    assert table["fraud_rate"].eq(1.0).all()


def test_value_bands_empty() -> None:
    table = frauds_by_value_band(pd.DataFrame(columns=["Amount", "Class"]))

    assert table.empty
    assert "value_band" in table.columns


def test_relative_time() -> None:
    transactions = pd.DataFrame(
        {
            "Time": [0.0, 3_599.0, 3_600.0, 86_399.0, 86_400.0],
            "Amount": [1.0] * 5,
            "Class": [0, 1, 1, 0, 1],
        }
    )
    derived = add_relative_time(transactions)

    assert derived["elapsed_hour"].tolist() == [0, 0, 1, 23, 24]
    assert derived["relative_day"].tolist() == [1, 1, 1, 1, 2]
    assert derived["relative_day_hour"].tolist() == [0, 0, 1, 23, 0]


def test_temporal_evolution_fills_hours_without_transactions() -> None:
    transactions = pd.DataFrame(
        {
            "Time": [0.0, 3_600.0, 7_200.0],
            "Amount": [1.0, 1.0, 1.0],
            "Class": [0, 1, 0],
        }
    )
    series = temporal_evolution(transactions)

    assert series["elapsed_hour"].tolist() == [0, 1, 2]
    assert series["frauds"].tolist() == [0, 1, 0]


def test_temporal_heatmap() -> None:
    transactions = pd.DataFrame(
        {
            "Time": [0.0, 3_599.0, 3_600.0, 86_400.0],
            "Amount": [1.0] * 4,
            "Class": [1, 1, 1, 1],
        }
    )
    matrix = temporal_heatmap(transactions)

    assert matrix.loc[1, 0] == 2
    assert matrix.loc[1, 1] == 1
    assert matrix.loc[2, 0] == 1
    assert matrix.shape == (2, 24)


def test_correlation_ordered_by_magnitude(synthetic_transactions: pd.DataFrame) -> None:
    transactions = synthetic_transactions.copy()
    transactions["V1"] = transactions["Class"].astype(
        "float64"
    ) * 2 + np.random.default_rng(0).normal(0, 0.01, len(transactions))

    correlations = correlation_with_target(transactions)

    assert correlations.index[0] == "V1"
    assert correlations.iloc[0] > 0
    assert correlations.abs().is_monotonic_decreasing
