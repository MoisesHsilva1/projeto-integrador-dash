"""CLI pipeline command tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from credit_card_fraud.cli import calculate_md5, main


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
