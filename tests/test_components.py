"""Plotly presentation component tests."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from credit_card_fraud.analytics.correlation import correlation_with_target
from credit_card_fraud.analytics.summary import class_distribution
from credit_card_fraud.analytics.temporal import temporal_evolution, temporal_heatmap
from credit_card_fraud.analytics.value import frauds_by_value_band
from credit_card_fraud.presentation.components import (
    correlation_chart,
    distribution_chart,
    heatmap_chart,
    temporal_chart,
    value_band_chart,
)


def test_distribution_chart(synthetic_transactions: pd.DataFrame) -> None:
    distribution = class_distribution(synthetic_transactions)
    figure = distribution_chart(distribution)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "pie"


def test_distribution_chart_empty() -> None:
    figure = distribution_chart(pd.DataFrame())
    assert isinstance(figure, go.Figure)
    assert len(figure.layout.annotations) > 0


def test_value_band_chart(synthetic_transactions: pd.DataFrame) -> None:
    bands = frauds_by_value_band(synthetic_transactions)
    figure = value_band_chart(bands)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2


def test_value_band_chart_empty() -> None:
    figure = value_band_chart(pd.DataFrame())
    assert isinstance(figure, go.Figure)
    assert len(figure.layout.annotations) > 0


def test_temporal_chart(synthetic_transactions: pd.DataFrame) -> None:
    series = temporal_evolution(synthetic_transactions)
    figure = temporal_chart(series)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2


def test_temporal_chart_empty() -> None:
    figure = temporal_chart(pd.DataFrame())
    assert isinstance(figure, go.Figure)
    assert len(figure.layout.annotations) > 0


def test_heatmap_chart(synthetic_transactions: pd.DataFrame) -> None:
    matrix = temporal_heatmap(synthetic_transactions)
    figure = heatmap_chart(matrix)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"


def test_heatmap_chart_empty() -> None:
    figure = heatmap_chart(pd.DataFrame())
    assert isinstance(figure, go.Figure)
    assert len(figure.layout.annotations) > 0


def test_correlation_chart(synthetic_transactions: pd.DataFrame) -> None:
    correlations = correlation_with_target(synthetic_transactions)
    figure = correlation_chart(correlations)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "bar"


def test_correlation_chart_empty() -> None:
    figure = correlation_chart(pd.Series(dtype="float64"))
    assert isinstance(figure, go.Figure)
    assert len(figure.layout.annotations) > 0
