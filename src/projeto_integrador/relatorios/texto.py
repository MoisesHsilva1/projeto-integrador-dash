"""Formatação de números para os relatórios em português.

Reexporta a fonte única de formatação pt-BR (``projeto_integrador.formatacao``),
preservando a API histórica utilizada pelos relatórios e pelos testes.
"""

from __future__ import annotations

from projeto_integrador.formatacao import (
    formatar_inteiro,
    formatar_percentual,
)
from projeto_integrador.formatacao import (
    formatar_valor as formatar_numero,
)

__all__ = ["formatar_inteiro", "formatar_numero", "formatar_percentual"]
