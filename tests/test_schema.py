"""Testes da validação de esquema."""

from __future__ import annotations

import pandas as pd
import pytest

from projeto_integrador.data.schema import ErroDeEsquema, validar_esquema


def test_esquema_valido(transacoes_sinteticas: pd.DataFrame) -> None:
    validar_esquema(transacoes_sinteticas)


def test_coluna_ausente(transacoes_sinteticas: pd.DataFrame) -> None:
    with pytest.raises(ErroDeEsquema, match="ausentes"):
        validar_esquema(transacoes_sinteticas.drop(columns=["V5"]))


def test_coluna_nao_numerica(transacoes_sinteticas: pd.DataFrame) -> None:
    invalida = transacoes_sinteticas.copy()
    invalida["V3"] = "texto"
    with pytest.raises(ErroDeEsquema, match="numéricas"):
        validar_esquema(invalida)


def test_classe_invalida(transacoes_sinteticas: pd.DataFrame) -> None:
    invalida = transacoes_sinteticas.copy()
    invalida.loc[0, "Class"] = 2
    with pytest.raises(ErroDeEsquema, match="Class"):
        validar_esquema(invalida)


def test_valor_negativo(transacoes_sinteticas: pd.DataFrame) -> None:
    invalida = transacoes_sinteticas.copy()
    invalida.loc[0, "Amount"] = -1.0
    with pytest.raises(ErroDeEsquema, match="Amount"):
        validar_esquema(invalida)


def test_base_vazia(transacoes_sinteticas: pd.DataFrame) -> None:
    with pytest.raises(ErroDeEsquema, match="vazia"):
        validar_esquema(transacoes_sinteticas.iloc[0:0])
