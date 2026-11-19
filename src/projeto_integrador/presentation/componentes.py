"""Componentes de gráfico do dashboard (Plotly, pt-BR)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from projeto_integrador.presentation.temas import (
    COR_FRAUDE,
    COR_LEGITIMA,
    COR_NEUTRA,
    aplicar_tema,
)


def _figura_vazia(
    mensagem: str = "Sem dados para os filtros selecionados.",
) -> go.Figure:
    figura = go.Figure()
    figura.add_annotation(
        text=mensagem, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
    )
    figura.update_xaxes(visible=False)
    figura.update_yaxes(visible=False)
    return aplicar_tema(figura)


def grafico_distribuicao(distribuicao: pd.DataFrame) -> go.Figure:
    """Rosca com a participação de legítimas e fraudes."""
    if distribuicao.empty or distribuicao["transacoes"].sum() == 0:
        return _figura_vazia()

    contagens = distribuicao["transacoes"].to_numpy()
    rotulos = ["Legítimas", "Fraudes"]
    cores = [COR_LEGITIMA, COR_FRAUDE]
    total = int(contagens.sum())

    figura = go.Figure(
        data=[
            go.Pie(
                labels=rotulos,
                values=contagens,
                hole=0.62,
                marker={"colors": cores},
                textinfo="percent",
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            )
        ]
    )
    figura.add_annotation(
        text=f"{total:,}".replace(",", ".") + "<br>transações".replace(",", "."),
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    figura.update_layout(title="Distribuição de fraudes", showlegend=True)
    return aplicar_tema(figura)


def grafico_faixas(tabela: pd.DataFrame) -> go.Figure:
    """Barras de fraudes por faixa com a taxa de fraude em linha."""
    if tabela.empty:
        return _figura_vazia()

    figura = go.Figure()
    figura.add_trace(
        go.Bar(
            x=tabela["faixa_valor"],
            y=tabela["fraudes"],
            name="Fraudes",
            marker_color=COR_FRAUDE,
            hovertemplate="Faixa %{x}<br>Fraudes: %{y}<extra></extra>",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=tabela["faixa_valor"],
            y=tabela["taxa_fraude"] * 100,
            name="Taxa de fraude (%)",
            yaxis="y2",
            mode="lines+markers",
            line={"color": COR_NEUTRA, "width": 2},
            marker={"color": COR_NEUTRA},
            hovertemplate="Taxa: %{y:.2f}%<extra></extra>",
        )
    )
    figura.update_layout(
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
    return aplicar_tema(figura)


def grafico_temporal(serie: pd.DataFrame) -> go.Figure:
    """Fraudes e volume total por hora decorrida."""
    if serie.empty:
        return _figura_vazia()

    figura = go.Figure()
    figura.add_trace(
        go.Bar(
            x=serie["hora_decorrida"],
            y=serie["fraudes"],
            name="Fraudes",
            marker_color=COR_FRAUDE,
            hovertemplate="Hora %{x}<br>Fraudes: %{y}<extra></extra>",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=serie["hora_decorrida"],
            y=serie["transacoes"],
            name="Transações",
            yaxis="y2",
            mode="lines",
            line={"color": COR_LEGITIMA, "width": 2},
            opacity=0.8,
            hovertemplate="Transações: %{y}<extra></extra>",
        )
    )
    figura.update_layout(
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
    return aplicar_tema(figura)


def grafico_calor(matriz: pd.DataFrame) -> go.Figure:
    """Mapa de calor dia relativo x hora do bloco de 24 h."""
    if matriz.empty:
        return _figura_vazia()

    figura = go.Figure(
        data=go.Heatmap(
            z=matriz.to_numpy(),
            x=list(matriz.columns),
            y=list(matriz.index),
            colorscale="Reds",
            hovertemplate="Dia %{y} · Hora %{x}<br>Fraudes: %{z}<extra></extra>",
            colorbar={"title": "Fraudes"},
        )
    )
    figura.update_layout(
        title="Mapa de calor de fraudes (blocos relativos)",
        xaxis_title="Hora dentro do bloco relativo de 24 h",
        yaxis_title="Dia relativo",
    )
    return aplicar_tema(figura)


def grafico_correlacoes(correlacoes: pd.Series, quantidade: int = 10) -> go.Figure:
    """Atributos com maior correlação com o rótulo de fraude."""
    if correlacoes.empty:
        return _figura_vazia()

    topo = correlacoes.head(quantidade).sort_values()
    cores = [COR_FRAUDE if valor > 0 else COR_LEGITIMA for valor in topo]

    figura = go.Figure(
        data=go.Bar(
            x=topo.values,
            y=topo.index,
            orientation="h",
            marker_color=cores,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    figura.add_vline(x=0, line_color=COR_NEUTRA, line_width=1)
    figura.update_layout(
        title="Correlações mais fortes com fraude",
        xaxis_title="Correlação com o rótulo de fraude",
        yaxis_title="Atributo",
    )
    return aplicar_tema(figura)
