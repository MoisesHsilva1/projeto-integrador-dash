"""Quality audit and preparation of the MLG/ULB base."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_card_fraud.config import (
    AMOUNT_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from credit_card_fraud.data.schema import validate_schema


def audit(df: pd.DataFrame) -> dict[str, object]:
    """Summarize the quality of the base, without altering it."""
    validate_schema(df)

    frauds = int(df[TARGET_COLUMN].sum())
    duplicated = df.duplicated()
    without_duplicates = df.loc[~duplicated]
    duration_hours = (df[TIME_COLUMN].max() - df[TIME_COLUMN].min()) / 3600

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "exact_duplicates": int(duplicated.sum()),
        "duplicate_frauds": int(df.loc[duplicated, TARGET_COLUMN].sum()),
        "frauds": frauds,
        "legitimate": int(len(df) - frauds),
        "fraud_rate": float(frauds / len(df)),
        "start_seconds": float(df[TIME_COLUMN].min()),
        "end_seconds": float(df[TIME_COLUMN].max()),
        "duration_hours": float(duration_hours),
        "total_amount": float(df[AMOUNT_COLUMN].sum()),
        "median_amount": float(df[AMOUNT_COLUMN].median()),
        "min_amount": float(df[AMOUNT_COLUMN].min()),
        "max_amount": float(df[AMOUNT_COLUMN].max()),
        "zero_amount_transactions": int((df[AMOUNT_COLUMN] == 0).sum()),
        "zero_amount_frauds": int(df.loc[df[AMOUNT_COLUMN] == 0, TARGET_COLUMN].sum()),
        "without_duplicates": {
            "rows": int(len(without_duplicates)),
            "frauds": int(without_duplicates[TARGET_COLUMN].sum()),
            "legitimate": int(
                len(without_duplicates) - without_duplicates[TARGET_COLUMN].sum()
            ),
            "fraud_rate": float(without_duplicates[TARGET_COLUMN].mean()),
        },
    }


def prepare_transactions(
    df: pd.DataFrame, *, remove_duplicates: bool = True
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the documented cleaning rules and record their impact.

    Rules:
    - remove exact duplicates (decision D-11), recording how many frauds
      were affected;
    - do not remove outliers or zero-amount transactions: frauds can be at
      those extremes (Section 4.6 of the plan);
    - do not alter values, columns, or row order.
    """
    validate_schema(df)

    original_rows = len(df)
    original_frauds = int(df[TARGET_COLUMN].sum())
    transactions = df.copy()

    removals: list[dict[str, object]] = []
    if remove_duplicates:
        duplicated = transactions.duplicated()
        rows_removed = int(duplicated.sum())
        frauds_removed = int(transactions.loc[duplicated, TARGET_COLUMN].sum())
        transactions = transactions.loc[~duplicated].reset_index(drop=True)
        removals.append(
            {
                "rule": "Remoção de duplicatas exatas (mantida a primeira ocorrência)",
                "rows_removed": rows_removed,
                "frauds_removed": frauds_removed,
            }
        )

    record: dict[str, object] = {
        "original_rows": original_rows,
        "final_rows": int(len(transactions)),
        "original_frauds": original_frauds,
        "final_frauds": int(transactions[TARGET_COLUMN].sum()),
        "final_legitimate": int(len(transactions) - transactions[TARGET_COLUMN].sum()),
        "final_fraud_rate": float(transactions[TARGET_COLUMN].mean()),
        "removals": removals,
        "kept": [
            "Transações com Amount = 0 (não são valores inválidos na base).",
            "Outliers de Amount: não removidos; fraudes podem estar nos extremos.",
            "Atributos V1–V28: mantidos como componentes PCA anonimizados.",
        ],
    }

    validate_schema(transactions)
    return transactions, record


def save_prepared(df: pd.DataFrame, path: Path | str) -> None:
    """Write the prepared base to Parquet, creating the directory if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
