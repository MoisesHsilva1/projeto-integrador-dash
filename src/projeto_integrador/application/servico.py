"""Serviços que a camada de apresentação consome."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projeto_integrador.analytics.correlacao import correlacao_com_alvo
from projeto_integrador.analytics.resumo import distribuicao_classes, resumo_geral
from projeto_integrador.analytics.temporal import (
    SEGUNDOS_POR_HORA,
    calor_temporal,
    evolucao_temporal,
)
from projeto_integrador.analytics.valor import fraudes_por_faixa_valor
from projeto_integrador.application.filtros import FiltrosDashboard, aplicar_filtros
from projeto_integrador.config import COLUNA_TEMPO, COLUNA_VALOR, SEMENTE
from projeto_integrador.data.loader import carregar_metricas, carregar_preparadas
from projeto_integrador.data.schema import validar_esquema


class ServicoDashboard:
    """Orquestra filtros e agregações para o dashboard.

    A apresentação conversa apenas com este serviço; as métricas continuam
    definidas na camada Analytics/Domain.
    """

    def __init__(self, transacoes: pd.DataFrame) -> None:
        validar_esquema(transacoes)
        self._transacoes = transacoes

    @property
    def limites(self) -> dict[str, float]:
        """Limites disponíveis para os filtros, em unidades originais e horas."""
        horas = self._transacoes[COLUNA_TEMPO] / SEGUNDOS_POR_HORA
        return {
            "valor_min": float(self._transacoes[COLUNA_VALOR].min()),
            "valor_max": float(self._transacoes[COLUNA_VALOR].max()),
            "hora_min": 0.0,
            "hora_max": float(horas.max()),
        }

    def recorte(self, filtros: FiltrosDashboard) -> pd.DataFrame:
        return aplicar_filtros(self._transacoes, filtros)

    def visao_geral(self, filtros: FiltrosDashboard) -> dict[str, object]:
        dados = self.recorte(filtros)
        return {
            "resumo": resumo_geral(dados),
            "distribuicao": distribuicao_classes(dados),
            "dados_vazios": dados.empty,
        }

    def faixas_de_valor(self, filtros: FiltrosDashboard) -> pd.DataFrame:
        return fraudes_por_faixa_valor(self.recorte(filtros))

    def evolucao_no_tempo(
        self, filtros: FiltrosDashboard, *, granularidade_horas: int = 1
    ) -> pd.DataFrame:
        return evolucao_temporal(
            self.recorte(filtros), granularidade_horas=granularidade_horas
        )

    def mapa_de_calor(self, filtros: FiltrosDashboard) -> pd.DataFrame:
        return calor_temporal(self.recorte(filtros))

    def correlacoes(self, filtros: FiltrosDashboard) -> pd.Series:
        return correlacao_com_alvo(self.recorte(filtros))

    def visao_completa(self, filtros: FiltrosDashboard) -> dict[str, object]:
        """Calcula todas as agregações a partir de um único recorte.

        Evita recomputar o filtro sobre a base para cada visualização.
        """
        dados = self.recorte(filtros)
        return {
            "dados": dados,
            "resumo": resumo_geral(dados),
            "distribuicao": distribuicao_classes(dados),
            "faixas": fraudes_por_faixa_valor(dados),
            "serie": evolucao_temporal(dados),
            "calor": calor_temporal(dados),
            "correlacoes": correlacao_com_alvo(dados),
            "dados_vazios": dados.empty,
        }

    def amostra_aleatoria(
        self, filtros: FiltrosDashboard, tamanho: int = 50
    ) -> pd.DataFrame:
        """Amostra reprodutível para conferência na interface."""
        dados = self.recorte(filtros)
        if dados.empty:
            return dados
        return dados.sample(n=min(tamanho, len(dados)), random_state=SEMENTE)

    def metricas_modelo(self) -> dict | None:
        """Métricas offline do modelo, quando disponíveis."""
        return carregar_metricas()


def criar_servico(caminho: Path | str | None = None) -> ServicoDashboard:
    """Cria o serviço a partir da base preparada (Data Access)."""
    return ServicoDashboard(carregar_preparadas(caminho))
