"""Audit and base preparation tests."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.data.audit import audit, prepare_transactions


def test_audit_counts_classes(synthetic_transactions: pd.DataFrame) -> None:
    audit_result = audit(synthetic_transactions)

    assert audit_result["rows"] == 400
    assert audit_result["columns"] == 31
    assert audit_result["frauds"] == 12
    assert audit_result["legitimate"] == 388
    assert audit_result["missing_values"] == 0
    assert audit_result["exact_duplicates"] == 0
    assert audit_result["fraud_rate"] == 12 / 400


def test_preparation_without_duplicates(synthetic_transactions: pd.DataFrame) -> None:
    prepared, record = prepare_transactions(synthetic_transactions)

    assert len(prepared) == 400
    assert record["original_rows"] == 400
    assert record["final_rows"] == 400
    assert record["final_frauds"] == 12


def test_preparation_removes_duplicates_and_records_impact(
    synthetic_transactions: pd.DataFrame,
) -> None:
    duplicated = pd.concat([synthetic_transactions, synthetic_transactions])
    audit_result = audit(duplicated)

    assert audit_result["exact_duplicates"] == 400
    assert audit_result["duplicate_frauds"] == 12

    prepared, record = prepare_transactions(duplicated)
    assert len(prepared) == 400
    assert record["removals"][0]["rows_removed"] == 400
    assert record["removals"][0]["frauds_removed"] == 12
    assert record["final_frauds"] == 12
    assert not prepared.duplicated().any()
