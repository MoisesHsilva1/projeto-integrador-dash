"""Interactive dashboard (Presentation Layer) — user interface in pt-BR."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from credit_card_fraud.application import (
    DashboardFilters,
    DashboardService,
    TransactionStatus,
    create_service,
)
from credit_card_fraud.formatting import (
    format_amount,
    format_integer,
    format_percentage,
)
from credit_card_fraud.presentation import components, state

STATUS_OPTIONS = {
    "Todas as transações": TransactionStatus.ALL,
    "Somente fraudes": TransactionStatus.FRAUD,
    "Somente legítimas": TransactionStatus.LEGITIMATE,
}

TABLE_LABELS = {
    "value_band": "Faixa de valor",
    "transactions": "Transações",
    "frauds": "Fraudes",
    "fraud_rate": "Taxa de fraude",
    "total_amount": "Valor movimentado",
}

EXPLORER_COLUMNS = ["Time", "Amount", "Class", *[f"V{i}" for i in range(1, 29)]]


def _format_integer(value: float | int) -> str:
    return format_integer(value)


def _format_amount(value: float) -> str:
    return format_amount(value)


def _build_filters(service: DashboardService) -> DashboardFilters:
    limits = service.limits
    current_filters = state.initialize_filters(limits)

    with st.sidebar:
        st.header("Filtros")

        amount_band = st.slider(
            "Faixa de valor (unidade original)",
            min_value=float(limits["min_amount"]),
            max_value=float(limits["max_amount"]),
            value=(
                float(current_filters.min_amount),
                float(current_filters.max_amount),
            ),
            step=max((limits["max_amount"] - limits["min_amount"]) / 500, 0.01),
        )
        st.caption(
            "Valores na unidade original da base; não há moeda definida no "
            "dicionário de dados e nenhuma conversão foi aplicada."
        )

        hour_limit = float(math.ceil(limits["max_hour"]))
        hour_band = st.slider(
            "Tempo decorrido (horas)",
            min_value=0.0,
            max_value=hour_limit,
            value=(float(current_filters.start_hour), float(current_filters.end_hour)),
            step=1.0,
        )
        st.caption(
            "Tempo relativo à primeira transação. A base não contém data/hora "
            "absoluta nem hora de relógio confirmada."
        )

        status_label = st.radio(
            "Classificação da transação",
            list(STATUS_OPTIONS),
            index=list(STATUS_OPTIONS).index(
                next(
                    k for k, v in STATUS_OPTIONS.items() if v == current_filters.status
                )
            ),
        )

        st.divider()
        if st.button("Limpar filtros", use_container_width=True):
            state.clear_filters(limits)
            st.rerun()

        st.divider()
        st.markdown(
            "**Base de dados**\n\n"
            "MLG/ULB — *Credit Card Fraud Detection*\n\n"
            "- 2 dias de transações europeias de cartão\n"
            "- 28 atributos anonimizados (PCA)\n"
            "- Rótulo de fraude: `Class`\n\n"
            "O filtro *tipo de transação* do mockup não está disponível, pois a "
            "base não possui comerciante nem categoria."
        )

    filters = DashboardFilters(
        min_amount=float(amount_band[0]),
        max_amount=float(amount_band[1]),
        status=STATUS_OPTIONS[status_label],
        start_hour=float(hour_band[0]),
        end_hour=float(hour_band[1]),
    )
    state.update_filters(filters)
    return filters


def _value_band_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(
            columns=list(TABLE_LABELS.values()) + ["Taxa de fraude (%)"]
        )
    display = table.rename(columns=TABLE_LABELS).copy()
    display["Taxa de fraude (%)"] = (table["fraud_rate"] * 100).round(2)
    display["Valor movimentado"] = table["total_amount"].round(2)
    return display[
        [
            "Faixa de valor",
            "Transações",
            "Fraudes",
            "Taxa de fraude (%)",
            "Valor movimentado",
        ]
    ]


def _explorer_table(data: pd.DataFrame, page_size: int, page: int) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    columns = [c for c in EXPLORER_COLUMNS if c in data.columns]
    subset = data[columns].copy()
    subset.insert(0, "Status", subset["Class"].map({0: "Legítima", 1: "Fraude"}))
    subset = subset.drop(columns=["Class"])
    start = (page - 1) * page_size
    end = start + page_size
    return subset.iloc[start:end]


def _render_model_metrics(service: DashboardService) -> None:
    metrics = service.model_metrics()
    if not metrics:
        st.info(
            "Model metrics unavailable. Run "
            "`uv run credit-card-fraud-pipeline train` to generate them."
        )
        return

    holdout = metrics.get("holdout_metrics", {})
    st.caption(
        "Holdout results (20% of the data), seed 42. Static: not affected by "
        "sidebar filters."
    )
    columns = st.columns(4)
    columns[0].metric("Modelo", str(metrics.get("model_selected", "—")))
    columns[1].metric("AUPRC", format_amount(float(holdout.get("auprc", 0.0)), 4))
    columns[2].metric("ROC-AUC", format_amount(float(holdout.get("roc_auc", 0.0)), 4))
    columns[3].metric("F1", format_amount(float(holdout.get("f1", 0.0)), 4))

    columns = st.columns(4)
    columns[0].metric(
        "Precisão", format_amount(float(holdout.get("precision", 0.0)), 4)
    )
    columns[1].metric("Recall", format_amount(float(holdout.get("recall", 0.0)), 4))
    columns[2].metric(
        "Limite de decisão", format_amount(float(holdout.get("threshold", 0.0)), 4)
    )
    columns[3].metric("Acurácia", format_amount(float(holdout.get("accuracy", 0.0)), 4))

    st.markdown("**Matriz de confusão (holdout)**")
    confusion = pd.DataFrame(
        {
            "": ["Real: legítima", "Real: fraude"],
            "Previsto legítima": [
                int(holdout.get("true_negatives", 0)),
                int(holdout.get("false_negatives", 0)),
            ],
            "Previsto fraude": [
                int(holdout.get("false_positives", 0)),
                int(holdout.get("true_positives", 0)),
            ],
        }
    )
    st.dataframe(confusion, hide_index=True, use_container_width=True)

    st.markdown("**Validação cruzada (3 dobras)**")
    cross_validation = metrics.get("cross_validation", {})
    if cross_validation:
        rows = []
        for name, values in cross_validation.items():
            rows.append(
                {
                    "Modelo": name,
                    "AUPRC (média)": format_amount(
                        float(values.get("auprc_mean", 0.0)), 4
                    ),
                    "AUPRC (desvio)": format_amount(
                        float(values.get("auprc_std", 0.0)), 4
                    ),
                    "F1 (média)": format_amount(float(values.get("f1_mean", 0.0)), 4),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> None:
    """Render the main dashboard page."""
    st.set_page_config(
        page_title="Análise de Fraudes em Cartão de Crédito",
        layout="wide",
    )

    st.title("Análise de Fraudes em Cartão de Crédito")

    try:
        service = create_service()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    filters = _build_filters(service)
    view = service.full_view(filters)
    summary = view["summary"]

    # Applied filters.
    active = state.get_active_filters(filters, service.limits)
    if active:
        st.caption("Filtros ativos: " + " · ".join(active))

    columns = st.columns(3)
    columns[0].metric("Transações", _format_integer(summary["total_transactions"]))
    columns[1].metric(
        "Transações legítimas", _format_integer(summary["total_legitimate"])
    )
    columns[2].metric(
        "Transações fraudulentas", _format_integer(summary["total_frauds"])
    )

    columns = st.columns(3)
    columns[0].metric("Taxa de fraude", format_percentage(summary["fraud_rate"]))
    columns[1].metric(
        "Valor total movimentado",
        _format_amount(summary["total_amount"]),
        help="Unidade original da base; nenhuma conversão aplicada.",
    )
    columns[2].metric(
        "Valor em fraudes",
        _format_amount(summary["fraud_amount"]),
        help="Unidade original da base; nenhuma conversão aplicada.",
    )

    if view["empty_data"]:
        st.warning(
            "Nenhuma transação atende aos filtros selecionados. "
            "Ajuste os filtros de valor, tempo ou classificação para ver resultados."
        )
        st.stop()

    st.divider()

    tab_overview, tab_transactions, tab_frauds, tab_explorer, tab_model = st.tabs(
        [
            "Visão geral",
            "Análise de transações",
            "Análise de fraudes",
            "Explorador de transações",
            "Desempenho do modelo",
        ]
    )

    with tab_overview:
        col_left, col_right = st.columns([1, 1.5])
        with col_left:
            st.subheader("Distribuição de fraudes")
            st.plotly_chart(
                components.distribution_chart(view["distribution"]),
                use_container_width=True,
                key="distribution_chart",
            )
        with col_right:
            st.subheader("Fraudes por faixa de valor")
            st.plotly_chart(
                components.value_band_chart(view["bands"]),
                use_container_width=True,
                key="value_band_chart_overview",
            )

        st.subheader("Evolução temporal")
        granularity = st.selectbox(
            "Granularidade (horas)",
            [1, 2, 4, 6, 12, 24],
            index=0,
            key="temporal_granularity",
        )
        series = service.evolution_over_time(filters, hour_granularity=granularity)
        st.plotly_chart(
            components.temporal_chart(series),
            use_container_width=True,
            key="temporal_chart",
        )

    with tab_transactions:
        st.subheader("Distribuição de valores")
        st.caption(
            "Volume de transações e fraudes por faixa de valor (mesma agregação "
            "da visão geral)."
        )
        st.plotly_chart(
            components.value_band_chart(view["bands"]),
            use_container_width=True,
            key="value_band_chart_transactions",
        )
        st.subheader("Detalhe por faixa de valor")
        st.dataframe(
            _value_band_table(view["bands"]), hide_index=True, use_container_width=True
        )

    with tab_frauds:
        col_heat, col_corr = st.columns([1.6, 1])
        with col_heat:
            st.subheader("Mapa de calor de fraudes")
            st.plotly_chart(
                components.heatmap_chart(view["heatmap"]),
                use_container_width=True,
                key="heatmap_chart",
            )
            st.caption(
                "Dias e horas são blocos relativos. Não são datas de calendário "
                "nem horas de relógio confirmadas."
            )
        with col_corr:
            st.subheader("Correlações com fraude")
            st.plotly_chart(
                components.correlation_chart(view["correlations"]),
                use_container_width=True,
                key="correlation_chart",
            )
            st.caption(
                "Atributos são componentes PCA anonimizados; a correlação é "
                "estatística, não um fator de negócio nomeado."
            )

    with tab_explorer:
        st.subheader("Explorador de transações")
        st.caption(
            "Transações do recorte atual. Não há identificador, comerciante ou "
            "categoria na base; as colunas exibidas são as disponíveis."
        )
        data = view["data"]
        page_size = st.selectbox("Registros por página", [50, 100, 250], index=0)
        total_pages = max(1, math.ceil(len(data) / page_size))
        page = st.number_input(
            "Página", min_value=1, max_value=total_pages, value=1, step=1
        )
        table = _explorer_table(data, page_size, int(page))
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption(f"Página {int(page)} de {total_pages} · {len(data)} transações.")

    with tab_model:
        st.subheader("Desempenho do modelo (offline)")
        _render_model_metrics(service)



if __name__ == "__main__":
    main()
