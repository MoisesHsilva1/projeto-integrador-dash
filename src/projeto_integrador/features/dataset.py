"""Construção da matriz de atributos e separação treino/teste."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from projeto_integrador.config import (
    COLUNA_ALVO,
    COLUNAS_MODELO,
    PROPORCAO_TESTE,
    SEMENTE,
)
from projeto_integrador.data.schema import validar_esquema


def construir_matriz(
    transacoes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos (Time, V1–V28, Amount) e alvo (Class)."""
    validar_esquema(transacoes)
    atributos = transacoes[COLUNAS_MODELO].astype("float64")
    alvo = transacoes[COLUNA_ALVO].astype("int8")
    return atributos, alvo


def dividir_treino_teste(
    atributos: pd.DataFrame,
    alvo: pd.Series,
    *,
    proporcao_teste: float = PROPORCAO_TESTE,
    semente: int = SEMENTE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste de forma estratificada, antes de qualquer ajuste."""
    return train_test_split(
        atributos,
        alvo,
        test_size=proporcao_teste,
        stratify=alvo,
        random_state=semente,
    )
