"""Testes da auditoria e da preparação da base."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.data.auditoria import auditar, preparar_transacoes


def test_auditoria_conta_classes(transacoes_sinteticas: pd.DataFrame) -> None:
    auditoria = auditar(transacoes_sinteticas)

    assert auditoria["linhas"] == 400
    assert auditoria["colunas"] == 31
    assert auditoria["fraudes"] == 12
    assert auditoria["legitimas"] == 388
    assert auditoria["valores_ausentes"] == 0
    assert auditoria["duplicatas_exatas"] == 0
    assert auditoria["taxa_fraude"] == 12 / 400


def test_preparacao_sem_duplicatas(transacoes_sinteticas: pd.DataFrame) -> None:
    preparadas, registro = preparar_transacoes(transacoes_sinteticas)

    assert len(preparadas) == 400
    assert registro["linhas_originais"] == 400
    assert registro["linhas_finais"] == 400
    assert registro["fraudes_finais"] == 12


def test_preparacao_remove_duplicatas_e_registra_impacto(
    transacoes_sinteticas: pd.DataFrame,
) -> None:
    duplicadas = pd.concat([transacoes_sinteticas, transacoes_sinteticas])
    auditoria = auditar(duplicadas)

    assert auditoria["duplicatas_exatas"] == 400
    assert auditoria["duplicatas_fraudes"] == 12

    preparadas, registro = preparar_transacoes(duplicadas)
    assert len(preparadas) == 400
    assert registro["remocoes"][0]["linhas_removidas"] == 400
    assert registro["remocoes"][0]["fraudes_removidas"] == 12
    assert registro["fraudes_finais"] == 12
    assert not preparadas.duplicated().any()
