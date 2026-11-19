"""Análise da ocorrência de fraudes por faixa de valor (OE-2)."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.config import COLUNA_ALVO, COLUNA_VALOR

LIMITES_FAIXAS = [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, float("inf")]
ROTULOS_FAIXAS = [
    "0-10",
    "10-50",
    "50-100",
    "100-250",
    "250-500",
    "500-1,000",
    "1,000+",
]
COLUNA_FAIXA = "faixa_valor"


def adicionar_faixa_valor(transacoes: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna categórica de faixa de valor sem alterar a base original."""
    dados = transacoes.copy()
    dados[COLUNA_FAIXA] = pd.cut(
        dados[COLUNA_VALOR],
        bins=LIMITES_FAIXAS,
        labels=ROTULOS_FAIXAS,
        right=False,
    )
    return dados


def fraudes_por_faixa_valor(transacoes: pd.DataFrame) -> pd.DataFrame:
    """Agrega transações, fraudes, taxa de fraude e volume por faixa de valor."""
    if transacoes.empty:
        return pd.DataFrame(
            {
                "faixa_valor": pd.Series(dtype="string"),
                "transacoes": pd.Series(dtype="int64"),
                "fraudes": pd.Series(dtype="int64"),
                "taxa_fraude": pd.Series(dtype="float64"),
                "valor_movimentado": pd.Series(dtype="float64"),
            }
        )

    dados = adicionar_faixa_valor(transacoes)
    agrupado = dados.groupby(COLUNA_FAIXA, observed=False).agg(
        transacoes=(COLUNA_ALVO, "size"),
        fraudes=(COLUNA_ALVO, "sum"),
        valor_movimentado=(COLUNA_VALOR, "sum"),
    )
    agrupado["taxa_fraude"] = agrupado["fraudes"] / agrupado["transacoes"].where(
        agrupado["transacoes"] > 0
    )

    tabela = agrupado.reset_index()
    tabela[COLUNA_FAIXA] = tabela[COLUNA_FAIXA].astype("string")
    return tabela[
        [
            COLUNA_FAIXA,
            "transacoes",
            "fraudes",
            "taxa_fraude",
            "valor_movimentado",
        ]
    ]
