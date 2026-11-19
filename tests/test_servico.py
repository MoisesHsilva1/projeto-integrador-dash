"""Testes do serviço consumido pelo dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projeto_integrador.application.filtros import FiltrosDashboard, StatusTransacao
from projeto_integrador.application.servico import ServicoDashboard, criar_servico
from projeto_integrador.data.auditoria import salvar_preparadas


def test_servico_limites(transacoes_sinteticas: pd.DataFrame) -> None:
    servico = ServicoDashboard(transacoes_sinteticas)
    limites = servico.limites

    assert limites["valor_min"] == 0.0
    assert limites["valor_max"] == transacoes_sinteticas["Amount"].max()
    assert limites["hora_min"] == 0.0
    assert limites["hora_max"] == pytest.approx(
        transacoes_sinteticas["Time"].max() / 3_600
    )


def test_servico_visao_geral(transacoes_sinteticas: pd.DataFrame) -> None:
    servico = ServicoDashboard(transacoes_sinteticas)
    visao = servico.visao_geral(FiltrosDashboard())

    assert visao["dados_vazios"] is False
    assert visao["resumo"]["total_transacoes"] == 400
    assert visao["resumo"]["total_fraudes"] == 12
    assert len(visao["distribuicao"]) == 2


def test_servico_com_filtro_e_recorte_vazio(
    transacoes_sinteticas: pd.DataFrame,
) -> None:
    servico = ServicoDashboard(transacoes_sinteticas)
    filtros = FiltrosDashboard(status=StatusTransacao.FRAUDE)
    visao = servico.visao_geral(filtros)

    assert visao["resumo"]["total_transacoes"] == 12
    assert servico.faixas_de_valor(filtros)["fraudes"].sum() == 12

    vazio = FiltrosDashboard(valor_min=1_000_000)
    assert servico.visao_geral(vazio)["dados_vazios"] is True
    assert servico.amostra_aleatoria(vazio).empty


def test_servico_agregacoes(transacoes_sinteticas: pd.DataFrame) -> None:
    servico = ServicoDashboard(transacoes_sinteticas)
    filtros = FiltrosDashboard()

    serie = servico.evolucao_no_tempo(filtros)
    matriz = servico.mapa_de_calor(filtros)
    correlacoes = servico.correlacoes(filtros)

    assert serie["fraudes"].sum() == 12
    assert matriz.to_numpy().sum() == 12
    assert not correlacoes.empty


def test_servico_visao_completa(transacoes_sinteticas: pd.DataFrame) -> None:
    servico = ServicoDashboard(transacoes_sinteticas)
    filtros = FiltrosDashboard()
    visao = servico.visao_completa(filtros)

    assert set(visao) >= {
        "dados",
        "resumo",
        "distribuicao",
        "faixas",
        "serie",
        "calor",
        "correlacoes",
        "dados_vazios",
    }
    assert visao["resumo"]["total_transacoes"] == 400
    assert visao["dados_vazios"] is False
    assert visao["serie"]["fraudes"].sum() == 12


def test_servico_metricas_modelo(transacoes_sinteticas: pd.DataFrame) -> None:
    servico = ServicoDashboard(transacoes_sinteticas)
    resultado = servico.metricas_modelo()
    # O arquivo pode ou não existir no ambiente; o método não deve quebrar.
    assert resultado is None or isinstance(resultado, dict)


def test_criar_servico_a_partir_do_parquet(
    tmp_path: Path, transacoes_sinteticas: pd.DataFrame
) -> None:
    caminho = tmp_path / "transacoes.parquet"
    salvar_preparadas(transacoes_sinteticas, caminho)

    servico = criar_servico(caminho)
    assert servico.limites["valor_max"] == transacoes_sinteticas["Amount"].max()


def test_criar_servico_sem_arquivo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        criar_servico(tmp_path / "inexistente.parquet")
