"""Visual theme and semantic tokens of the dashboard."""

from __future__ import annotations

from typing import Any

# Semantic colors consistent across the dashboard.
LEGITIMATE_COLOR = "#2e9e6b"
FRAUD_COLOR = "#d64545"
NEUTRAL_COLOR = "#6c7a89"
ALERT_COLOR = "#d9a441"
BACKGROUND_COLOR = "#ffffff"

SEMANTIC_COLORS = {
    "legitimate": LEGITIMATE_COLOR,
    "fraud": FRAUD_COLOR,
    "neutral": NEUTRAL_COLOR,
    "alert": ALERT_COLOR,
}

DEFAULT_LAYOUT: dict[str, Any] = {
    "template": "plotly_white",
    "font": {"family": "Inter, sans-serif", "size": 12},
    "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "hoverlabel": {"namelength": -1},
}


def apply_theme(figure: Any) -> Any:
    """Apply the default layout and visual tokens to a Plotly figure."""
    figure.update_layout(**DEFAULT_LAYOUT)
    return figure
