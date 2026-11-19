"""Validação do esquema esperado para a base MLG/ULB (creditcard.csv)."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.config import (
    COLUNA_ALVO,
    COLUNA_TEMPO,
    COLUNA_VALOR,
    COLUNAS_PCA,
)

COLUNAS_OBRIGATORIAS = [COLUNA_TEMPO, *COLUNAS_PCA, COLUNA_VALOR, COLUNA_ALVO]


class ErroDeEsquema(ValueError):
    """Erro levantado quando a base não corresponde ao esquema esperado."""


def validar_esquema(df: pd.DataFrame, *, exigir_alvo: bool = True) -> None:
    """Valida colunas, tipos e faixas mínimas da base MLG/ULB.

    A validação não altera o DataFrame. Levanta ``ErroDeEsquema`` com uma
    mensagem explicativa quando a base não corresponde ao esperado.
    """
    if df.empty:
        raise ErroDeEsquema("A base está vazia.")

    faltantes = [coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in df.columns]
    if faltantes:
        raise ErroDeEsquema(
            "Colunas obrigatórias ausentes: " + ", ".join(sorted(faltantes))
        )

    nao_numericas = [
        coluna
        for coluna in COLUNAS_OBRIGATORIAS
        if not pd.api.types.is_numeric_dtype(df[coluna])
    ]
    if nao_numericas:
        raise ErroDeEsquema(
            "Colunas que deveriam ser numéricas: " + ", ".join(sorted(nao_numericas))
        )

    if (df[COLUNA_TEMPO] < 0).any():
        raise ErroDeEsquema("A coluna 'Time' não pode conter valores negativos.")
    if (df[COLUNA_VALOR] < 0).any():
        raise ErroDeEsquema("A coluna 'Amount' não pode conter valores negativos.")

    if exigir_alvo:
        valores_alvo = set(df[COLUNA_ALVO].unique())
        if not valores_alvo <= {0, 1}:
            raise ErroDeEsquema(
                "A coluna 'Class' deve conter apenas 0 (legítima) e 1 (fraude); "
                f"valores encontrados: {sorted(valores_alvo)}."
            )
