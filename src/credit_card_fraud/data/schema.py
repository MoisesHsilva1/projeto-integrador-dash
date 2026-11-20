"""Validation of the expected schema for the MLG/ULB base (creditcard.csv)."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.config import (
    AMOUNT_COLUMN,
    PCA_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)

REQUIRED_COLUMNS = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN, TARGET_COLUMN]


class SchemaError(ValueError):
    """Error raised when the base does not match the expected schema."""


def validate_schema(df: pd.DataFrame, *, require_target: bool = True) -> None:
    """Validate columns, types, and minimum ranges of the MLG/ULB base.

    Validation does not alter the DataFrame. Raises ``SchemaError`` with an
    explanatory message when the base does not match the expected schema.
    """
    if df.empty:
        raise SchemaError("The base is empty.")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SchemaError("Missing required columns: " + ", ".join(sorted(missing)))

    non_numeric = [
        column
        for column in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise SchemaError(
            "Columns that should be numeric: " + ", ".join(sorted(non_numeric))
        )

    if (df[TIME_COLUMN] < 0).any():
        raise SchemaError("The 'Time' column cannot contain negative values.")
    if (df[AMOUNT_COLUMN] < 0).any():
        raise SchemaError("The 'Amount' column cannot contain negative values.")

    if require_target:
        target_values = set(df[TARGET_COLUMN].unique())
        if not target_values <= {0, 1}:
            raise SchemaError(
                "The 'Class' column must contain only 0 (legitimate) and 1 "
                f"(fraud); values found: {sorted(target_values)}."
            )
