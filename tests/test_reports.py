"""Report and figure generation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_card_fraud.data.audit import audit, prepare_transactions
from credit_card_fraud.reports.eda import generate_eda_report
from credit_card_fraud.reports.quality import generate_quality_report


def test_quality_report(tmp_path: Path, synthetic_transactions: pd.DataFrame) -> None:
    audit_result = audit(synthetic_transactions)
    _, record = prepare_transactions(synthetic_transactions)
    path = generate_quality_report(
        audit_result,
        record,
        checksum="abc123",
        path=tmp_path / "quality.md",
    )

    text = path.read_text(encoding="utf-8")
    assert "abc123" in text
    assert "284" not in text
    assert "Fraudes" in text or "fraudes" in text
    assert "Tratamento aplicado" in text


def test_eda_report(tmp_path: Path, synthetic_transactions: pd.DataFrame) -> None:
    path = generate_eda_report(
        synthetic_transactions,
        figures_directory=tmp_path / "figures",
        path=tmp_path / "eda.md",
    )

    text = path.read_text(encoding="utf-8")
    assert "OE-1" in text
    assert "OE-2" in text
    assert "OE-3" in text
    assert "400" in text

    figures = sorted((tmp_path / "figures").glob("eda_*.png"))
    assert len(figures) == 6
    assert all(figure.stat().st_size > 0 for figure in figures)
