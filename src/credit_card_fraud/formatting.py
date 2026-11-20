"""Value formatting following the Brazilian Portuguese (pt-BR) pattern.

Neutral module, with no dependency on the presentation or report layer. It is
the single source of pt-BR formatting in the project; reports re-export these
functions from ``credit_card_fraud.reports.text``.
"""


def format_integer(value: int | float) -> str:
    """Format an integer with a dot thousands separator."""
    return f"{int(round(value)):,}".replace(",", ".")


def format_amount(value: float, decimals: int = 2) -> str:
    """Format a numeric value using the Brazilian Portuguese pattern."""
    return (
        f"{value:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    )


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a proportion (0–1) as a percentage in pt-BR pattern."""
    return f"{format_amount(value * 100, decimals)}%"


def format_relative_hour(value: float) -> str:
    """Format a relative elapsed hour, without asserting a clock time."""
    return f"{format_amount(value, 0)} h"
