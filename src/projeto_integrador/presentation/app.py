"""Dashboard interativo (Presentation Layer) — interface em pt-BR."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from projeto_integrador.application import (
    FiltrosDashboard,
    ServicoDashboard,
    StatusTransacao,
    criar_servico,
)
from projeto_integrador.formatacao import (
    formatar_inteiro,
    formatar_percentual,
    formatar_valor,
)
from projeto_integrador.presentation import componentes, estado

OPCOES_STATUS = {
    "Todas as transações": StatusTransacao.TODAS,
    "Somente fraudes": StatusTransacao.FRAUDE,
    "Somente legítimas": StatusTransacao.LEGITIMA,
}

ROTULOS_TABELA = {
    "faixa_valor": "Faixa de valor",
    "transacoes": "Transações",
    "fraudes": "Fraudes",
    "taxa_fraude": "Taxa de fraude",
    "valor_movimentado": "Valor movimentado",
}

COLUNAS_EXPLORADOR = ["Time", "Amount", "Class", *[f"V{i}" for i in range(1, 29)]]


def _inteiro(valor: float | int) -> str:
    return formatar_inteiro(valor)


def _valor(valor: float) -> str:
    return formatar_valor(valor)


def _construir_filtros(servico: ServicoDashboard) -> FiltrosDashboard:
    limites = servico.limites
    filtros_atuais = estado.inicializar_filtros(limites)

    with st.sidebar:
        st.header("Filtros")

        faixa_valor = st.slider(
            "Faixa de valor (unidade original)",
            min_value=float(limites["valor_min"]),
            max_value=float(limites["valor_max"]),
            value=(float(filtros_atuais.valor_min), float(filtros_atuais.valor_max)),
            step=max((limites["valor_max"] - limites["valor_min"]) / 500, 0.01),
        )
        st.caption(
            "Valores na unidade original da base; não há moeda definida no "
            "dicionário de dados e nenhuma conversão foi aplicada."
        )

        limite_horas = float(math.ceil(limites["hora_max"]))
        faixa_horas = st.slider(
            "Tempo decorrido (horas)",
            min_value=0.0,
            max_value=limite_horas,
            value=(float(filtros_atuais.hora_inicio), float(filtros_atuais.hora_fim)),
            step=1.0,
        )
        st.caption(
            "Tempo relativo à primeira transação. A base não contém data/hora "
            "absoluta nem hora de relógio confirmada."
        )

        rotulo_status = st.radio(
            "Classificação da transação",
            list(OPCOES_STATUS),
            index=list(OPCOES_STATUS).index(
                next(k for k, v in OPCOES_STATUS.items() if v == filtros_atuais.status)
            ),
        )

        st.divider()
        if st.button("Limpar filtros", use_container_width=True):
            estado.limpar_filtros(limites)
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

    filtros = FiltrosDashboard(
        valor_min=float(faixa_valor[0]),
        valor_max=float(faixa_valor[1]),
        status=OPCOES_STATUS[rotulo_status],
        hora_inicio=float(faixa_horas[0]),
        hora_fim=float(faixa_horas[1]),
    )
    estado.atualizar_filtros(filtros)
    return filtros


def _tabela_faixas(tabela: pd.DataFrame) -> pd.DataFrame:
    if tabela.empty:
        return pd.DataFrame(
            columns=list(ROTULOS_TABELA.values()) + ["Taxa de fraude (%)"]
        )
    exibicao = tabela.rename(columns=ROTULOS_TABELA).copy()
    exibicao["Taxa de fraude (%)"] = (tabela["taxa_fraude"] * 100).round(2)
    exibicao["Valor movimentado"] = tabela["valor_movimentado"].round(2)
    return exibicao[
        [
            "Faixa de valor",
            "Transações",
            "Fraudes",
            "Taxa de fraude (%)",
            "Valor movimentado",
        ]
    ]


def _tabela_explorador(
    dados: pd.DataFrame, tamanho_pagina: int, pagina: int
) -> pd.DataFrame:
    if dados.empty:
        return pd.DataFrame()
    colunas = [c for c in COLUNAS_EXPLORADOR if c in dados.columns]
    recorte = dados[colunas].copy()
    recorte.insert(0, "Status", recorte["Class"].map({0: "Legítima", 1: "Fraude"}))
    recorte = recorte.drop(columns=["Class"])
    inicio = (pagina - 1) * tamanho_pagina
    fim = inicio + tamanho_pagina
    return recorte.iloc[inicio:fim]


def _renderizar_metricas_modelo(servico: ServicoDashboard) -> None:
    metricas = servico.metricas_modelo()
    if not metricas:
        st.info(
            "Métricas do modelo indisponíveis. Execute "
            "`uv run projeto-integrador-pipeline treinar` para gerá-las."
        )
        return

    holdout = metricas.get("metricas_holdout", {})
    st.caption(
        "Resultados do holdout (20% dos dados), semente 42. Estáticos: não são "
        "afetados pelos filtros da barra lateral."
    )
    colunas = st.columns(4)
    colunas[0].metric("Modelo", str(metricas.get("modelo_selecionado", "—")))
    colunas[1].metric("AUPRC", formatar_valor(float(holdout.get("auprc", 0.0)), 4))
    colunas[2].metric("ROC-AUC", formatar_valor(float(holdout.get("roc_auc", 0.0)), 4))
    colunas[3].metric("F1", formatar_valor(float(holdout.get("f1", 0.0)), 4))

    colunas = st.columns(4)
    colunas[0].metric(
        "Precisão", formatar_valor(float(holdout.get("precisao", 0.0)), 4)
    )
    colunas[1].metric("Recall", formatar_valor(float(holdout.get("recall", 0.0)), 4))
    colunas[2].metric(
        "Limite de decisão", formatar_valor(float(holdout.get("limite", 0.0)), 4)
    )
    colunas[3].metric(
        "Acurácia", formatar_valor(float(holdout.get("acuracia", 0.0)), 4)
    )

    st.markdown("**Matriz de confusão (holdout)**")
    mc = pd.DataFrame(
        {
            "": ["Real: legítima", "Real: fraude"],
            "Previsto legítima": [
                int(holdout.get("verdadeiros_negativos", 0)),
                int(holdout.get("falsos_negativos", 0)),
            ],
            "Previsto fraude": [
                int(holdout.get("falsos_positivos", 0)),
                int(holdout.get("verdadeiros_positivos", 0)),
            ],
        }
    )
    st.dataframe(mc, hide_index=True, use_container_width=True)

    st.markdown("**Validação cruzada (3 dobras)**")
    cv = metricas.get("validacao_cruzada", {})
    if cv:
        linhas = []
        for nome, valores in cv.items():
            linhas.append(
                {
                    "Modelo": nome,
                    "AUPRC (média)": formatar_valor(
                        float(valores.get("auprc_media", 0.0)), 4
                    ),
                    "AUPRC (desvio)": formatar_valor(
                        float(valores.get("auprc_desvio", 0.0)), 4
                    ),
                    "F1 (média)": formatar_valor(
                        float(valores.get("f1_media", 0.0)), 4
                    ),
                }
            )
        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)


def main() -> None:
    """Renderiza a página principal do dashboard."""
    st.set_page_config(
        page_title="Análise de Fraudes em Cartão de Crédito",
        page_icon="💳",
        layout="wide",
    )

    st.title("Análise de Fraudes em Cartão de Crédito")
    st.caption(
        "Base MLG/ULB · dois dias de transações europeias · tempo relativo · "
        "atributos anonimizados por PCA · protótipo acadêmico de apoio à decisão"
    )

    try:
        servico = criar_servico()
    except FileNotFoundError as erro:
        st.error(str(erro))
        st.stop()

    filtros = _construir_filtros(servico)
    visao = servico.visao_completa(filtros)
    resumo = visao["resumo"]

    # Filtros aplicados.
    ativos = estado.obter_filtros_ativos(filtros, servico.limites)
    if ativos:
        st.caption("Filtros ativos: " + " · ".join(ativos))

    colunas = st.columns(3)
    colunas[0].metric("Transações", _inteiro(resumo["total_transacoes"]))
    colunas[1].metric("Transações legítimas", _inteiro(resumo["total_legitimas"]))
    colunas[2].metric("Transações fraudulentas", _inteiro(resumo["total_fraudes"]))

    colunas = st.columns(3)
    colunas[0].metric("Taxa de fraude", formatar_percentual(resumo["taxa_fraude"]))
    colunas[1].metric(
        "Valor total movimentado",
        _valor(resumo["valor_total"]),
        help="Unidade original da base; nenhuma conversão aplicada.",
    )
    colunas[2].metric(
        "Valor em fraudes",
        _valor(resumo["valor_fraude"]),
        help="Unidade original da base; nenhuma conversão aplicada.",
    )

    if visao["dados_vazios"]:
        st.warning(
            "Nenhuma transação atende aos filtros selecionados. "
            "Ajuste os filtros de valor, tempo ou classificação para ver resultados."
        )
        st.stop()

    st.divider()

    aba_visao, aba_transacoes, aba_fraudes, aba_explorador, aba_modelo = st.tabs(
        [
            "Visão geral",
            "Análise de transações",
            "Análise de fraudes",
            "Explorador de transações",
            "Desempenho do modelo",
        ]
    )

    with aba_visao:
        col_esq, col_dir = st.columns([1, 1.5])
        with col_esq:
            st.subheader("Distribuição de fraudes")
            st.plotly_chart(
                componentes.grafico_distribuicao(visao["distribuicao"]),
                use_container_width=True,
                key="grafico_distribuicao",
            )
        with col_dir:
            st.subheader("Fraudes por faixa de valor")
            st.plotly_chart(
                componentes.grafico_faixas(visao["faixas"]),
                use_container_width=True,
                key="grafico_faixas_visao",
            )

        st.subheader("Evolução temporal")
        granularidade = st.selectbox(
            "Granularidade (horas)", [1, 2, 4, 6, 12, 24], index=0, key="gran_temporal"
        )
        serie = servico.evolucao_no_tempo(filtros, granularidade_horas=granularidade)
        st.plotly_chart(
            componentes.grafico_temporal(serie),
            use_container_width=True,
            key="grafico_temporal",
        )

    with aba_transacoes:
        st.subheader("Distribuição de valores")
        st.caption(
            "Volume de transações e fraudes por faixa de valor (mesma agregação "
            "da visão geral)."
        )
        st.plotly_chart(
            componentes.grafico_faixas(visao["faixas"]),
            use_container_width=True,
            key="grafico_faixas_transacoes",
        )
        st.subheader("Detalhe por faixa de valor")
        st.dataframe(
            _tabela_faixas(visao["faixas"]), hide_index=True, use_container_width=True
        )

    with aba_fraudes:
        col_calor, col_corr = st.columns([1.6, 1])
        with col_calor:
            st.subheader("Mapa de calor de fraudes")
            st.plotly_chart(
                componentes.grafico_calor(visao["calor"]),
                use_container_width=True,
                key="grafico_calor",
            )
            st.caption(
                "Dias e horas são blocos relativos. Não são datas de calendário "
                "nem horas de relógio confirmadas."
            )
        with col_corr:
            st.subheader("Correlações com fraude")
            st.plotly_chart(
                componentes.grafico_correlacoes(visao["correlacoes"]),
                use_container_width=True,
                key="grafico_correlacoes",
            )
            st.caption(
                "Atributos são componentes PCA anonimizados; a correlação é "
                "estatística, não um fator de negócio nomeado."
            )

    with aba_explorador:
        st.subheader("Explorador de transações")
        st.caption(
            "Transações do recorte atual. Não há identificador, comerciante ou "
            "categoria na base; as colunas exibidas são as disponíveis."
        )
        dados = visao["dados"]
        tamanho_pagina = st.selectbox("Registros por página", [50, 100, 250], index=0)
        total_paginas = max(1, math.ceil(len(dados) / tamanho_pagina))
        pagina = st.number_input(
            "Página", min_value=1, max_value=total_paginas, value=1, step=1
        )
        tabela = _tabela_explorador(dados, tamanho_pagina, int(pagina))
        st.dataframe(tabela, hide_index=True, use_container_width=True)
        st.caption(
            f"Página {int(pagina)} de {total_paginas} · {len(dados)} transações."
        )

    with aba_modelo:
        st.subheader("Desempenho do modelo (offline)")
        _renderizar_metricas_modelo(servico)

    st.divider()
    st.caption(
        "Valores na unidade original da base; nenhuma conversão de moeda foi "
        "aplicada. Rótulos históricos; o dashboard é ferramenta de apoio à "
        "decisão, não sistema automatizado de bloqueio."
    )


if __name__ == "__main__":
    main()
