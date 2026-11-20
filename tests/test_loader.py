"""Data Access loading tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from credit_card_fraud.data.loader import load_metrics, load_prepared, load_raw
from credit_card_fraud.data.schema import SchemaError


def test_load_raw_from_csv(
    tmp_path: Path, synthetic_transactions: pd.DataFrame
) -> None:
    path = tmp_path / "raw.csv"
    synthetic_transactions.to_csv(path, index=False)

    loaded = load_raw(path)

    assert len(loaded) == len(synthetic_transactions)
    assert list(loaded.columns) == list(synthetic_transactions.columns)


def test_load_raw_requires_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Raw base not found"):
        load_raw(tmp_path / "nonexistent.csv")


def test_load_raw_validates_schema(tmp_path: Path) -> None:
    invalid = pd.DataFrame({"Time": [1.0], "Amount": [10.0]})
    path = tmp_path / "invalid.csv"
    invalid.to_csv(path, index=False)

    with pytest.raises(SchemaError, match="Missing"):
        load_raw(path)


def test_load_prepared_from_parquet(
    tmp_path: Path, synthetic_transactions: pd.DataFrame
) -> None:
    path = tmp_path / "prepared.parquet"
    synthetic_transactions.to_parquet(path, index=False)

    loaded = load_prepared(path)

    assert len(loaded) == len(synthetic_transactions)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        synthetic_transactions.reset_index(drop=True),
    )


def test_load_prepared_requires_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Prepared base not found"):
        load_prepared(tmp_path / "nonexistent.parquet")


def test_load_metrics_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_metrics(tmp_path / "metrics.json") is None


def test_load_metrics_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    content = {"seed": 42, "selected_model": "Random Forest"}
    path.write_text(json.dumps(content), encoding="utf-8")

    loaded = load_metrics(path)

    assert loaded == content
