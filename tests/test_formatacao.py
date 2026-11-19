"""Testes da formatação pt-BR."""

from __future__ import annotations

from projeto_integrador.formatacao import (
    formatar_hora_relativa,
    formatar_inteiro,
    formatar_percentual,
    formatar_valor,
)


def test_formatar_inteiro() -> None:
    assert formatar_inteiro(1234) == "1.234"
    assert formatar_inteiro(0) == "0"
    assert formatar_inteiro(1234567) == "1.234.567"


def test_formatar_valor() -> None:
    assert formatar_valor(1234.5) == "1.234,50"
    assert formatar_valor(0.0) == "0,00"
    assert formatar_valor(25691.16) == "25.691,16"


def test_formatar_percentual() -> None:
    assert formatar_percentual(0.0125) == "1,25%"
    assert formatar_percentual(0.1727) == "17,27%"


def test_formatar_hora_relativa() -> None:
    assert formatar_hora_relativa(44.0) == "44 h"
