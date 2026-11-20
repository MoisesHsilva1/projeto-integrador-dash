"""Training, threshold, and evaluation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_card_fraud.features.dataset import build_matrix, split_train_test
from credit_card_fraud.models.evaluation import (
    classification_metrics,
    error_summary,
    feature_importances,
    permutation_importance,
    value_impact,
)
from credit_card_fraud.models.training import (
    choose_threshold,
    compare_models,
    create_models,
    load_model,
    save_model,
    train_and_evaluate,
)


def test_classification_metrics_known() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = classification_metrics(y, probabilities, 0.5)

    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert 0 < metrics["auprc"] <= 1


def test_choose_threshold_maximizes_f1() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, f1 = choose_threshold(y, probabilities)

    assert 0.2 < threshold <= 0.8
    assert f1 == 1.0


def test_train_and_evaluate_holdout(synthetic_transactions: pd.DataFrame) -> None:
    features, target = build_matrix(synthetic_transactions)
    train, test, target_train, target_test = split_train_test(features, target)
    result = train_and_evaluate(train, target_train, test, target_test, folds=2)

    assert set(result["cross_validation"]) == set(create_models())
    assert result["best_model"] in result["cross_validation"]
    assert 0.0 <= result["decision_threshold"] <= 1.0
    assert len(result["test_probabilities"]) == len(test)

    metrics = result["holdout_metrics"]
    assert metrics["true_positives"] + metrics["false_negatives"] == 2
    assert metrics["true_negatives"] + metrics["false_positives"] == 78

    importances = feature_importances(result["model"], list(test.columns))
    assert len(importances) == len(test.columns)
    assert importances.index[0] in test.columns


def test_save_and_load_model(
    tmp_path: Path, synthetic_transactions: pd.DataFrame
) -> None:
    features, target = build_matrix(synthetic_transactions)
    model = create_models()["Logistic Regression (baseline)"]
    model.fit(features, target)

    path = tmp_path / "model.joblib"
    save_model(model, {"decision_threshold": 0.5}, path)
    loaded = load_model(path)

    assert loaded["metadata"]["decision_threshold"] == 0.5
    probabilities = loaded["model"].predict_proba(features)[:, 1]
    assert probabilities.shape == (len(features),)


def test_value_impact_calculation() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])
    amounts = np.array([10.0, 50.0, 100.0, 200.0])
    impact = value_impact(y_true, probabilities, amounts, threshold=0.5)

    assert impact["total_amount"] == 360.0
    assert impact["fraud_amount"] == 300.0
    assert impact["false_positive_amount"] == 50.0
    assert impact["false_negative_amount"] == 100.0
    assert impact["false_positives"] == 1
    assert impact["false_negatives"] == 1


def test_error_summary() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])
    amounts = np.array([10.0, 50.0, 100.0, 200.0])
    summary = error_summary(y_true, probabilities, amounts, threshold=0.5)

    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert len(summary["largest_false_positives"]) == 1
    assert len(summary["largest_false_negatives"]) == 1


def test_permutation_importance(synthetic_transactions: pd.DataFrame) -> None:
    features, target = build_matrix(synthetic_transactions)
    model = create_models()["Logistic Regression (baseline)"]
    model.fit(features, target)

    importances = permutation_importance(
        model, features, target, n_repeats=2, sample=100, seed=42
    )
    assert not importances.empty
    assert "mean" in importances.columns
    assert "std" in importances.columns


def test_compare_models_return_probabilities(
    synthetic_transactions: pd.DataFrame,
) -> None:
    features, target = build_matrix(synthetic_transactions)
    results, probs = compare_models(
        features, target, folds=2, return_probabilities=True
    )
    assert len(results) == len(create_models())
    assert set(probs.keys()) == set(create_models())
    for p in probs.values():
        assert len(p) == len(target)
