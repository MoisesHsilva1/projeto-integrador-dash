"""Testes das métricas e agregações da camada Analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projeto_integrador.analytics.correlacao import correlacao_com_alvo
from projeto_integrador.analytics.resumo import distribuicao_classes, resumo_geral
from projeto_integrador.analytics.temporal import (
    adicionar_tempo_relativo,
    calor_temporal,
    evolucao_temporal,
)
from projeto_integrador.analytics.valor import fraudes_por_faixa_valor


def test_resumo_geral(transacoes_sinteticas: pd.DataFrame) -> None:
    resumo = resumo_geral(transacoes_sinteticas)

    assert resumo["total_transacoes"] == 400
    assert resumo["total_fraudes"] == 12
    assert resumo["total_legitimas"] == 388
    assert resumo["taxa_fraude"] == 12 / 400
    assert resumo["valor_total"] == transacoes_sinteticas["Amount"].sum()


def test_resumo_geral_vazio() -> None:
    resumo = resumo_geral(pd.DataFrame(columns=["Amount", "Class"]))

    assert resumo["total_transacoes"] == 0
    assert resumo["taxa_fraude"] == 0.0


def test_distribuicao_classes(transacoes_sinteticas: pd.DataFrame) -> None:
    distribuicao = distribuicao_classes(transacoes_sinteticas)

    assert list(distribuicao["classe"]) == [0, 1]
    assert list(distribuicao["transacoes"]) == [388, 12]
    assert distribuicao["participacao"].sum() == 1.0


def test_faixas_respeitam_limites() -> None:
    transacoes = pd.DataFrame(
        {
            "Amount": [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 9_999.0],
            "Class": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    tabela = fraudes_por_faixa_valor(transacoes)

    contagens = dict(zip(tabela["faixa_valor"], tabela["fraudes"], strict=True))
    assert contagens == {
        "0-10": 1,
        "10-50": 1,
        "50-100": 1,
        "100-250": 1,
        "250-500": 1,
        "500-1,000": 1,
        "1,000+": 2,
    }
    assert tabela["taxa_fraude"].eq(1.0).all()


def test_faixas_vazio() -> None:
    tabela = fraudes_por_faixa_valor(pd.DataFrame(columns=["Amount", "Class"]))

    assert tabela.empty
    assert "faixa_valor" in tabela.columns


def test_tempo_relativo() -> None:
    transacoes = pd.DataFrame(
        {
            "Time": [0.0, 3_599.0, 3_600.0, 86_399.0, 86_400.0],
            "Amount": [1.0] * 5,
            "Class": [0, 1, 1, 0, 1],
        }
    )
    derivado = adicionar_tempo_relativo(transacoes)

    assert derivado["hora_decorrida"].tolist() == [0, 0, 1, 23, 24]
    assert derivado["dia_relativo"].tolist() == [1, 1, 1, 1, 2]
    assert derivado["hora_do_dia_relativo"].tolist() == [0, 0, 1, 23, 0]


def test_evolucao_temporal_preenche_horas_sem_transacao() -> None:
    transacoes = pd.DataFrame(
        {
            "Time": [0.0, 3_600.0, 7_200.0],
            "Amount": [1.0, 1.0, 1.0],
            "Class": [0, 1, 0],
        }
    )
    serie = evolucao_temporal(transacoes)

    assert serie["hora_decorrida"].tolist() == [0, 1, 2]
    assert serie["fraudes"].tolist() == [0, 1, 0]


def test_calor_temporal() -> None:
    transacoes = pd.DataFrame(
        {
            "Time": [0.0, 3_599.0, 3_600.0, 86_400.0],
            "Amount": [1.0] * 4,
            "Class": [1, 1, 1, 1],
        }
    )
    matriz = calor_temporal(transacoes)

    assert matriz.loc[1, 0] == 2
    assert matriz.loc[1, 1] == 1
    assert matriz.loc[2, 0] == 1
    assert matriz.shape == (2, 24)


def test_correlacao_ordenada_por_magnitude(transacoes_sinteticas: pd.DataFrame) -> None:
    transacoes = transacoes_sinteticas.copy()
    transacoes["V1"] = transacoes["Class"].astype(
        "float64"
    ) * 2 + np.random.default_rng(0).normal(0, 0.01, len(transacoes))

    correlacoes = correlacao_com_alvo(transacoes)

    assert correlacoes.index[0] == "V1"
    assert correlacoes.iloc[0] > 0
    assert correlacoes.abs().is_monotonic_decreasing
