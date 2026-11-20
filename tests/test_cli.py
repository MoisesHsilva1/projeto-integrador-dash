"""CLI pipeline command tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import credit_card_fraud.cli as cli_module
from credit_card_fraud.cli import _check_prerequisites, calculate_md5, main


def test_calculate_md5(tmp_path: Path) -> None:
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello credit card fraud", encoding="utf-8")
    md5 = calculate_md5(sample_file)
    assert isinstance(md5, str)
    assert len(md5) == 32


def test_cli_requires_command() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_cli_handles_missing_file_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_fnf() -> None:
        raise FileNotFoundError("Raw base not found")

    monkeypatch.setattr("credit_card_fraud.cli.load_raw", raise_fnf)
    with pytest.raises(SystemExit) as exc_info:
        main(["prepare"])
    assert exc_info.value.code != 0


def test_check_prerequisites_missing_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises FileNotFoundError with actionable message when raw file is absent."""
    monkeypatch.setattr(cli_module, "RAW_PATH", tmp_path / "creditcard.csv")
    with pytest.raises(FileNotFoundError, match="Place 'creditcard.csv'"):
        _check_prerequisites("prepare")


def test_check_prerequisites_missing_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises FileNotFoundError with actionable message when prepared base is absent."""
    raw = tmp_path / "creditcard.csv"
    raw.touch()
    monkeypatch.setattr(cli_module, "RAW_PATH", raw)
    monkeypatch.setattr(cli_module, "PROCESSED_PATH", tmp_path / "transactions.parquet")
    with pytest.raises(FileNotFoundError, match="Run the 'prepare' step first"):
        _check_prerequisites("eda")


def test_check_prerequisites_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises FileNotFoundError when model artifact is absent."""
    raw = tmp_path / "creditcard.csv"
    raw.touch()
    processed = tmp_path / "transactions.parquet"
    processed.touch()
    monkeypatch.setattr(cli_module, "RAW_PATH", raw)
    monkeypatch.setattr(cli_module, "PROCESSED_PATH", processed)
    monkeypatch.setattr(cli_module, "MODEL_PATH", tmp_path / "model_final.joblib")
    with pytest.raises(FileNotFoundError, match="Run the 'train' step first"):
        _check_prerequisites("experiments")


def test_check_prerequisites_no_error_when_all_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Does not raise when all required files exist."""
    raw = tmp_path / "creditcard.csv"
    raw.touch()
    processed = tmp_path / "transactions.parquet"
    processed.touch()
    model = tmp_path / "model_final.joblib"
    model.touch()
    monkeypatch.setattr(cli_module, "RAW_PATH", raw)
    monkeypatch.setattr(cli_module, "PROCESSED_PATH", processed)
    monkeypatch.setattr(cli_module, "MODEL_PATH", model)
    _check_prerequisites("prepare")
    _check_prerequisites("eda")
    _check_prerequisites("train")
    _check_prerequisites("experiments")


def test_check_prerequisites_dashboard_needs_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dashboard command does not require any file — must not raise."""
    monkeypatch.setattr(cli_module, "RAW_PATH", tmp_path / "creditcard.csv")
    monkeypatch.setattr(cli_module, "PROCESSED_PATH", tmp_path / "transactions.parquet")
    monkeypatch.setattr(cli_module, "MODEL_PATH", tmp_path / "model_final.joblib")
    _check_prerequisites("dashboard")
