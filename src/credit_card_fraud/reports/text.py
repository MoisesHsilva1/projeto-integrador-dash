"""Number formatting for reports.

Re-exports the single source of pt-BR formatting (``credit_card_fraud.formatting``),
preserving the historical API used by reports and tests. User-facing report and
dashboard labels remain in Brazilian Portuguese.
"""

from __future__ import annotations

from credit_card_fraud.formatting import (
    format_amount as format_number,
)
from credit_card_fraud.formatting import (
    format_integer,
    format_percentage,
)

__all__ = ["format_integer", "format_number", "format_percentage"]
