"""Schema validation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_card_fraud.data.schema import SchemaError, validate_schema


def test_valid_schema(synthetic_transactions: pd.DataFrame) -> None:
    validate_schema(synthetic_transactions)


def test_missing_column(synthetic_transactions: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="Missing"):
        validate_schema(synthetic_transactions.drop(columns=["V5"]))


def test_non_numeric_column(synthetic_transactions: pd.DataFrame) -> None:
    invalid = synthetic_transactions.copy()
    invalid["V3"] = "text"
    with pytest.raises(SchemaError, match="numeric"):
        validate_schema(invalid)


def test_invalid_class(synthetic_transactions: pd.DataFrame) -> None:
    invalid = synthetic_transactions.copy()
    invalid.loc[0, "Class"] = 2
    with pytest.raises(SchemaError, match="Class"):
        validate_schema(invalid)


def test_negative_amount(synthetic_transactions: pd.DataFrame) -> None:
    invalid = synthetic_transactions.copy()
    invalid.loc[0, "Amount"] = -1.0
    with pytest.raises(SchemaError, match="Amount"):
        validate_schema(invalid)


def test_empty_base(synthetic_transactions: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="empty"):
        validate_schema(synthetic_transactions.iloc[0:0])
