"""Report and figure generation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_card_fraud.data.audit import audit, prepare_transactions
from credit_card_fraud.features.dataset import build_matrix, split_train_test
from credit_card_fraud.models.training import train_and_evaluate
from credit_card_fraud.reports.eda import generate_eda_report
from credit_card_fraud.reports.model import generate_model_report
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


def test_model_report(tmp_path: Path, synthetic_transactions: pd.DataFrame) -> None:
    features, target = build_matrix(synthetic_transactions)
    train, test, target_train, target_test = split_train_test(features, target)
    result = train_and_evaluate(train, target_train, test, target_test, folds=2)

    metadata = {
        "seed": 42,
        "folds": 2,
        "test_proportion": 0.2,
    }
    path = generate_model_report(
        result,
        test,
        target_test,
        metadata=metadata,
        figures_directory=tmp_path / "figures",
        path=tmp_path / "model.md",
    )

    text = path.read_text(encoding="utf-8")
    assert "Relatório de modelagem e avaliação" in text
    assert "AUPRC" in text
    figures = sorted((tmp_path / "figures").glob("modelo_*.png"))
    assert len(figures) == 4
    assert all(figure.stat().st_size > 0 for figure in figures)
