"""Geração de relatórios e figuras (entregáveis das etapas E3, E4 e E7)."""

from projeto_integrador.relatorios.eda import gerar_figuras, gerar_relatorio_eda
from projeto_integrador.relatorios.modelo import gerar_relatorio_modelo
from projeto_integrador.relatorios.qualidade import gerar_relatorio_qualidade

__all__ = [
    "gerar_figuras",
    "gerar_relatorio_eda",
    "gerar_relatorio_modelo",
    "gerar_relatorio_qualidade",
]
