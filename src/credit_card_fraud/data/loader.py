"""Loading of the raw base, the prepared base, and model artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from credit_card_fraud.config import (
    ERRORS_PATH,
    IMPORTANCES_PATH,
    MODEL_PATH,
    PROCESSED_PATH,
    RAW_PATH,
)
from credit_card_fraud.data.schema import validate_schema

MSG_RAW_MISSING = (
    "Raw base not found at {path}. See data/README.md to place the "
    "creditcard.csv file in data/raw/."
)
MSG_PREPARED_MISSING = (
    "Prepared base not found at {path}. Run the preparation step "
    "(uv run credit-card-fraud-pipeline preparar) before this step."
)


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the raw base, without modifying it."""
    destination = Path(path) if path is not None else RAW_PATH
    if not destination.exists():
        raise FileNotFoundError(MSG_RAW_MISSING.format(path=destination))

    transactions = pd.read_csv(destination)
    validate_schema(transactions)
    return transactions


@lru_cache(maxsize=4)
def _load_parquet_cached(path: str, _modified_at: float) -> pd.DataFrame:
    # The ``_modified_at`` argument invalidates the cache when the file changes.
    transactions = pd.read_parquet(path)
    validate_schema(transactions)
    return transactions


def load_prepared(path: Path | str | None = None) -> pd.DataFrame:
    """Load the prepared Parquet base, with cache.

    The returned DataFrame is shared across calls: treat it as read-only and
    create copies in the layers that need to transform it.
    """
    destination = Path(path) if path is not None else PROCESSED_PATH
    if not destination.exists():
        raise FileNotFoundError(MSG_PREPARED_MISSING.format(path=destination))

    return _load_parquet_cached(str(destination), destination.stat().st_mtime)


@lru_cache(maxsize=8)
def _load_json_cached(path: str, _modified_at: float) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _load_json_if_exists(path: Path | str | None, default: Path) -> dict | None:
    """Load a JSON artifact, returning ``None`` when absent."""
    destination = Path(path) if path is not None else default
    if not destination.exists():
        return None
    return _load_json_cached(str(destination), destination.stat().st_mtime)


@lru_cache(maxsize=2)
def _load_artifact_cached(path: str, _modified_at: float) -> dict:
    return joblib.load(path)


def load_model_artifact(
    path: Path | str | None = None,
) -> dict[str, object] | None:
    """Load the saved model artifact, when available."""
    destination = Path(path) if path is not None else MODEL_PATH
    if not destination.exists():
        return None
    return _load_artifact_cached(str(destination), destination.stat().st_mtime)


def load_metrics(path: Path | str | None = None) -> dict | None:
    """Load the offline model metrics, when available.

    Returns ``None`` if the file does not exist, so the interface displays an
    empty state instead of breaking.
    """
    from credit_card_fraud.config import METRICS_PATH

    return _load_json_if_exists(path, METRICS_PATH)


def load_importances(path: Path | str | None = None) -> dict | None:
    """Load the permutation importances, when available."""
    return _load_json_if_exists(path, IMPORTANCES_PATH)


def load_errors(path: Path | str | None = None) -> dict | None:
    """Load the holdout error summary, when available."""
    return _load_json_if_exists(path, ERRORS_PATH)
