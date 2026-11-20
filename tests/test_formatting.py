"""pt-BR formatting tests."""

from __future__ import annotations

from credit_card_fraud.formatting import (
    format_amount,
    format_integer,
    format_percentage,
    format_relative_hour,
)


def test_format_integer() -> None:
    assert format_integer(1234) == "1.234"
    assert format_integer(0) == "0"
    assert format_integer(1234567) == "1.234.567"


def test_format_amount() -> None:
    assert format_amount(1234.5) == "1.234,50"
    assert format_amount(0.0) == "0,00"
    assert format_amount(25691.16) == "25.691,16"


def test_format_percentage() -> None:
    assert format_percentage(0.0125) == "1,25%"
    assert format_percentage(0.1727) == "17,27%"


def test_format_relative_hour() -> None:
    assert format_relative_hour(44.0) == "44 h"
