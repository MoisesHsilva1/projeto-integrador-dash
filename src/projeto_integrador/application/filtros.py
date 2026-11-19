"""Filtros de interface e orquestração do dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from projeto_integrador.analytics.temporal import SEGUNDOS_POR_HORA
from projeto_integrador.config import COLUNA_ALVO, COLUNA_TEMPO, COLUNA_VALOR


class StatusTransacao(StrEnum):
    """Status de fraude selecionável no dashboard."""

    TODAS = "todas"
    FRAUDE = "fraude"
    LEGITIMA = "legitima"


@dataclass(frozen=True)
class FiltrosDashboard:
    """Filtros aplicáveis ao recorte do dashboard.

    Valores de valor em unidades originais da base e de tempo em horas
    decorridas desde a primeira transação (``Time / 3600``).
    """

    valor_min: float = 0.0
    valor_max: float = float("inf")
    status: StatusTransacao = StatusTransacao.TODAS
    hora_inicio: float = 0.0
    hora_fim: float = float("inf")

    def __post_init__(self) -> None:
        if self.valor_min < 0:
            raise ValueError("O valor mínimo não pode ser negativo.")
        if self.valor_max < self.valor_min:
            raise ValueError("O valor máximo deve ser maior ou igual ao mínimo.")
        if self.hora_inicio < 0:
            raise ValueError("A hora inicial não pode ser negativa.")
        if self.hora_fim < self.hora_inicio:
            raise ValueError("A hora final deve ser maior ou igual à inicial.")


def aplicar_filtros(
    transacoes: pd.DataFrame, filtros: FiltrosDashboard
) -> pd.DataFrame:
    """Retorna apenas as transações que atendem aos filtros informados."""
    mascara = transacoes[COLUNA_VALOR].between(filtros.valor_min, filtros.valor_max)
    mascara &= transacoes[COLUNA_TEMPO].between(
        filtros.hora_inicio * SEGUNDOS_POR_HORA,
        filtros.hora_fim * SEGUNDOS_POR_HORA,
    )

    if filtros.status is StatusTransacao.FRAUDE:
        mascara &= transacoes[COLUNA_ALVO] == 1
    elif filtros.status is StatusTransacao.LEGITIMA:
        mascara &= transacoes[COLUNA_ALVO] == 0

    return transacoes.loc[mascara]
