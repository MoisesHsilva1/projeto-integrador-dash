"""Correlação entre atributos e o rótulo de fraude."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.config import (
    COLUNA_ALVO,
    COLUNA_TEMPO,
    COLUNA_VALOR,
    COLUNAS_PCA,
)


def correlacao_com_alvo(
    transacoes: pd.DataFrame, colunas: list[str] | None = None
) -> pd.Series:
    """Correlação de Pearson de cada atributo com ``Class``, por magnitude."""
    if transacoes.empty:
        return pd.Series(dtype="float64", name="correlacao")

    selecionadas = colunas or [*COLUNAS_PCA, COLUNA_VALOR, COLUNA_TEMPO]
    correlacoes = transacoes[selecionadas + [COLUNA_ALVO]].corr()[COLUNA_ALVO]
    correlacoes = correlacoes.drop(labels=[COLUNA_ALVO], errors="ignore")
    correlacoes = correlacoes.dropna()
    correlacoes.name = "correlacao"
    return correlacoes.reindex(correlacoes.abs().sort_values(ascending=False).index)


def matriz_correlacao(
    transacoes: pd.DataFrame, colunas: list[str] | None = None
) -> pd.DataFrame:
    """Matriz de correlação para um conjunto de atributos."""
    selecionadas = colunas or [*COLUNAS_PCA, COLUNA_VALOR, COLUNA_TEMPO]
    return transacoes[selecionadas].corr()
