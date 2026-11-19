"""Testes da preparação de atributos e do split."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.config import COLUNA_ALVO, COLUNAS_MODELO
from projeto_integrador.features.dataset import construir_matriz, dividir_treino_teste


def test_construir_matriz(transacoes_sinteticas: pd.DataFrame) -> None:
    atributos, alvo = construir_matriz(transacoes_sinteticas)

    assert list(atributos.columns) == COLUNAS_MODELO
    assert len(atributos) == len(alvo) == 400
    assert alvo.name == COLUNA_ALVO
    assert set(alvo.unique()) == {0, 1}
    assert not atributos.isna().any().any()


def test_split_estratificado(transacoes_sinteticas: pd.DataFrame) -> None:
    atributos, alvo = construir_matriz(transacoes_sinteticas)
    atributos_treino, atributos_teste, alvo_treino, alvo_teste = dividir_treino_teste(
        atributos, alvo, proporcao_teste=0.2
    )

    assert len(atributos_teste) == 80
    assert len(atributos_treino) == 320
    assert not set(atributos_treino.index) & set(atributos_teste.index)
    assert alvo_teste.sum() == 2
    assert alvo_treino.sum() == 10


def test_split_reprodutivel(transacoes_sinteticas: pd.DataFrame) -> None:
    atributos, alvo = construir_matriz(transacoes_sinteticas)
    primeiro = dividir_treino_teste(atributos, alvo, semente=42)
    segundo = dividir_treino_teste(atributos, alvo, semente=42)

    assert primeiro[1].index.equals(segundo[1].index)
