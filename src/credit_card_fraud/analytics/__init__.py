"""Analytics/Domain: metrics, aggregations, and analytical rules."""

from credit_card_fraud.analytics.correlation import (
    correlation_matrix,
    correlation_with_target,
)
from credit_card_fraud.analytics.summary import class_distribution, general_summary
from credit_card_fraud.analytics.temporal import (
    add_relative_time,
    temporal_evolution,
    temporal_heatmap,
)
from credit_card_fraud.analytics.value import add_value_band, frauds_by_value_band

__all__ = [
    "add_value_band",
    "add_relative_time",
    "class_distribution",
    "correlation_matrix",
    "correlation_with_target",
    "frauds_by_value_band",
    "general_summary",
    "temporal_heatmap",
    "temporal_evolution",
]
