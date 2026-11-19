"""Análise temporal (OE-3).

`Time` é o número de segundos desde a primeira transação da base. Não existe
data/hora absoluta nem origem documentada em meia-noite. Por isso, todas as
análises usam blocos relativos: hora decorrida, dia relativo (1 e 2) e hora
dentro do bloco relativo de 24 horas (D-13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projeto_integrador.config import COLUNA_ALVO, COLUNA_TEMPO, COLUNA_VALOR

SEGUNDOS_POR_HORA = 3_600
HORAS_POR_DIA = 24
COLUNA_HORA_DECORRIDA = "hora_decorrida"
COLUNA_DIA_RELATIVO = "dia_relativo"
COLUNA_HORA_DO_DIA_RELATIVO = "hora_do_dia_relativo"


def adicionar_tempo_relativo(transacoes: pd.DataFrame) -> pd.DataFrame:
    """Deriva blocos de tempo relativos a partir de ``Time``."""
    dados = transacoes.copy()
    horas = np.floor(dados[COLUNA_TEMPO] / SEGUNDOS_POR_HORA).astype("int64")
    dados[COLUNA_HORA_DECORRIDA] = horas
    dados[COLUNA_DIA_RELATIVO] = (horas // HORAS_POR_DIA + 1).astype("int64")
    dados[COLUNA_HORA_DO_DIA_RELATIVO] = (horas % HORAS_POR_DIA).astype("int64")
    return dados


def evolucao_temporal(
    transacoes: pd.DataFrame, *, granularidade_horas: int = 1
) -> pd.DataFrame:
    """Série de transações, fraudes e taxa de fraude por bloco de horas."""
    if granularidade_horas < 1:
        raise ValueError("A granularidade deve ser de pelo menos 1 hora.")

    colunas = [COLUNA_HORA_DECORRIDA, "transacoes", "fraudes", "taxa_fraude"]
    if transacoes.empty:
        return pd.DataFrame(columns=colunas)

    dados = adicionar_tempo_relativo(transacoes)
    if granularidade_horas > 1:
        dados[COLUNA_HORA_DECORRIDA] = (
            dados[COLUNA_HORA_DECORRIDA] // granularidade_horas * granularidade_horas
        )

    agrupado = dados.groupby(COLUNA_HORA_DECORRIDA, observed=True).agg(
        transacoes=(COLUNA_ALVO, "size"),
        fraudes=(COLUNA_ALVO, "sum"),
        valor_movimentado=(COLUNA_VALOR, "sum"),
    )
    agrupado["taxa_fraude"] = agrupado["fraudes"] / agrupado["transacoes"].where(
        agrupado["transacoes"] > 0
    )

    inicio = int(agrupado.index.min())
    fim = int(agrupado.index.max())
    indice = range(inicio, fim + granularidade_horas, granularidade_horas)
    agrupado = agrupado.reindex(indice, fill_value=0)
    agrupado["taxa_fraude"] = agrupado["fraudes"] / agrupado["transacoes"].where(
        agrupado["transacoes"] > 0
    )

    return agrupado.reset_index()[
        [
            COLUNA_HORA_DECORRIDA,
            "transacoes",
            "fraudes",
            "taxa_fraude",
            "valor_movimentado",
        ]
    ]


def calor_temporal(transacoes: pd.DataFrame) -> pd.DataFrame:
    """Matriz dia relativo x hora do bloco de 24 h com a contagem de fraudes."""
    if transacoes.empty:
        return pd.DataFrame()

    dados = adicionar_tempo_relativo(transacoes)
    matriz = (
        dados.groupby(
            [COLUNA_DIA_RELATIVO, COLUNA_HORA_DO_DIA_RELATIVO], observed=True
        )[COLUNA_ALVO]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=range(HORAS_POR_DIA), fill_value=0)
    )
    matriz.index.name = COLUNA_DIA_RELATIVO
    matriz.columns.name = COLUNA_HORA_DO_DIA_RELATIVO
    return matriz.astype("int64")
