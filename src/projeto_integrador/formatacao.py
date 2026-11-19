"""Formatação de valores no padrão brasileiro (pt-BR).

Módulo neutro, sem dependência de camada de apresentação ou de relatório.
É a fonte única de formatação pt-BR do projeto; os relatórios reexportam
estas funções a partir de ``projeto_integrador.relatorios.texto``.
"""


def formatar_inteiro(valor: int | float) -> str:
    """Formata um inteiro com separador de milhar por ponto."""
    return f"{int(round(valor)):,}".replace(",", ".")


def formatar_valor(valor: float, casas: int = 2) -> str:
    """Formata um valor numérico no padrão pt-BR."""
    return f"{valor:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def formatar_percentual(valor: float, casas: int = 2) -> str:
    """Formata uma proporção (0–1) como percentual no padrão pt-BR."""
    return f"{formatar_valor(valor * 100, casas)}%"


def formatar_hora_relativa(valor: float) -> str:
    """Formata uma hora decorrida relativa, sem afirmar hora de relógio."""
    return f"{formatar_valor(valor, 0)} h"
