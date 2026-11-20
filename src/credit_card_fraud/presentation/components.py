"""Dashboard chart components (Plotly, pt-BR user-facing text)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from credit_card_fraud.formatting import format_integer
from credit_card_fraud.presentation.themes import (
    FRAUD_COLOR,
    LEGITIMATE_COLOR,
    NEUTRAL_COLOR,
    apply_theme,
)


def _empty_figure(
    message: str = "Sem dados para os filtros selecionados.",
) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return apply_theme(figure)


def distribution_chart(distribution: pd.DataFrame) -> go.Figure:
    """Donut with legitimate and fraud participation."""
    if distribution.empty or distribution["transactions"].sum() == 0:
        return _empty_figure()

    counts = distribution["transactions"].to_numpy()
    labels = ["Legítimas", "Fraudes"]
    colors = [LEGITIMATE_COLOR, FRAUD_COLOR]
    total = int(counts.sum())

    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.62,
                marker={"colors": colors},
                textinfo="percent",
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            )
        ]
    )
    figure.add_annotation(
        text=f"{format_integer(total)}<br>transações",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    figure.update_layout(title="Distribuição de fraudes", showlegend=True)
    return apply_theme(figure)


def value_band_chart(table: pd.DataFrame) -> go.Figure:
    """Fraud bars by value band with fraud rate line."""
    if table.empty:
        return _empty_figure()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=table["value_band"],
            y=table["frauds"],
            name="Fraudes",
            marker_color=FRAUD_COLOR,
            hovertemplate="Faixa %{x}<br>Fraudes: %{y}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["value_band"],
            y=table["fraud_rate"] * 100,
            name="Taxa de fraude (%)",
            yaxis="y2",
            mode="lines+markers",
            line={"color": NEUTRAL_COLOR, "width": 2},
            marker={"color": NEUTRAL_COLOR},
            hovertemplate="Taxa: %{y:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title="Fraudes por faixa de valor",
        xaxis_title="Faixa de valor (unidade original)",
        yaxis_title="Fraudes",
        yaxis2={"title": "Taxa de fraude (%)", "overlaying": "y", "side": "right"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return apply_theme(figure)


def temporal_chart(series: pd.DataFrame) -> go.Figure:
    """Frauds and total volume per elapsed hour."""
    if series.empty:
        return _empty_figure()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=series["elapsed_hour"],
            y=series["frauds"],
            name="Fraudes",
            marker_color=FRAUD_COLOR,
            hovertemplate="Hora %{x}<br>Fraudes: %{y}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=series["elapsed_hour"],
            y=series["transactions"],
            name="Transações",
            yaxis="y2",
            mode="lines",
            line={"color": LEGITIMATE_COLOR, "width": 2},
            opacity=0.8,
            hovertemplate="Transações: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Evolução temporal (blocos relativos)",
        xaxis_title="Hora decorrida desde a primeira transação",
        yaxis_title="Fraudes",
        yaxis2={"title": "Transações", "overlaying": "y", "side": "right"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return apply_theme(figure)


def heatmap_chart(matrix: pd.DataFrame) -> go.Figure:
    """Relative day x relative 24-hour block hour heatmap."""
    if matrix.empty:
        return _empty_figure()

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="Reds",
            hovertemplate="Dia %{y} · Hora %{x}<br>Fraudes: %{z}<extra></extra>",
            colorbar={"title": "Fraudes"},
        )
    )
    figure.update_layout(
        title="Mapa de calor de fraudes (blocos relativos)",
        xaxis_title="Hora dentro do bloco relativo de 24 h",
        yaxis_title="Dia relativo",
    )
    return apply_theme(figure)


def correlation_chart(correlations: pd.Series, quantity: int = 10) -> go.Figure:
    """Attributes with the strongest correlation with the fraud label."""
    if correlations.empty:
        return _empty_figure()

    top = correlations.head(quantity).sort_values()
    colors = [FRAUD_COLOR if value > 0 else LEGITIMATE_COLOR for value in top]

    figure = go.Figure(
        data=go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=NEUTRAL_COLOR, line_width=1)
    figure.update_layout(
        title="Correlações mais fortes com fraude",
        xaxis_title="Correlação com o rótulo de fraude",
        yaxis_title="Atributo",
    )
    return apply_theme(figure)
