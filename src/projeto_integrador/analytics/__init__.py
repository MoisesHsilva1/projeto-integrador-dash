"""Analytics/Domain: métricas, agregações e regras analíticas."""

from projeto_integrador.analytics.correlacao import (
    correlacao_com_alvo,
    matriz_correlacao,
)
from projeto_integrador.analytics.resumo import distribuicao_classes, resumo_geral
from projeto_integrador.analytics.temporal import (
    adicionar_tempo_relativo,
    calor_temporal,
    evolucao_temporal,
)
from projeto_integrador.analytics.valor import (
    adicionar_faixa_valor,
    fraudes_por_faixa_valor,
)

__all__ = [
    "adicionar_faixa_valor",
    "adicionar_tempo_relativo",
    "calor_temporal",
    "correlacao_com_alvo",
    "distribuicao_classes",
    "evolucao_temporal",
    "fraudes_por_faixa_valor",
    "matriz_correlacao",
    "resumo_geral",
]
