"""Testes dos filtros de dashboard."""

from __future__ import annotations

import pandas as pd
import pytest

from projeto_integrador.application.filtros import (
    FiltrosDashboard,
    StatusTransacao,
    aplicar_filtros,
)


def test_filtro_por_valor(transacoes_sinteticas: pd.DataFrame) -> None:
    filtros = FiltrosDashboard(valor_min=50, valor_max=100)
    recorte = aplicar_filtros(transacoes_sinteticas, filtros)

    assert not recorte.empty
    assert recorte["Amount"].between(50, 100).all()


def test_filtro_por_status(transacoes_sinteticas: pd.DataFrame) -> None:
    fraude = aplicar_filtros(
        transacoes_sinteticas, FiltrosDashboard(status=StatusTransacao.FRAUDE)
    )
    legitima = aplicar_filtros(
        transacoes_sinteticas, FiltrosDashboard(status=StatusTransacao.LEGITIMA)
    )

    assert len(fraude) == 12
    assert fraude["Class"].eq(1).all()
    assert len(legitima) == 388
    assert legitima["Class"].eq(0).all()


def test_filtro_por_tempo(transacoes_sinteticas: pd.DataFrame) -> None:
    filtros = FiltrosDashboard(hora_inicio=0, hora_fim=1)
    recorte = aplicar_filtros(transacoes_sinteticas, filtros)

    assert recorte["Time"].between(0, 3_600).all()
    assert (recorte["Time"] > 3_600).sum() == 0


def test_filtros_invalidos() -> None:
    with pytest.raises(ValueError, match="valor máximo"):
        FiltrosDashboard(valor_min=10, valor_max=5)
    with pytest.raises(ValueError, match="negativo"):
        FiltrosDashboard(valor_min=-1)
    with pytest.raises(ValueError, match="hora final"):
        FiltrosDashboard(hora_inicio=10, hora_fim=5)
