"""Model evaluation figures and report (step E7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from credit_card_fraud.models.evaluation import feature_importances  # noqa: E402
from credit_card_fraud.reports.text import (  # noqa: E402
    format_integer,
    format_number,
    format_percentage,
)

FRAUD_COLOR = "#d62728"
LEGITIMATE_COLOR = "#4c78a8"


def _save(figure: plt.Figure, destination: Path) -> Path:
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def _model_comparison_figure(
    cross_validation: dict[str, dict[str, float]], destination: Path
) -> Path:
    names = list(cross_validation)
    means = [cross_validation[name]["auprc_mean"] for name in names]
    stds = [cross_validation[name]["auprc_std"] for name in names]
    positions = np.arange(len(names))

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(positions, means, yerr=stds, capsize=6, color=LEGITIMATE_COLOR)
    axis.set_xticks(positions, names, rotation=12, ha="right")
    axis.set_ylabel("Mean AUPRC (cross-validation)")
    axis.set_ylim(0, 1)
    axis.set_title("Model comparison by AUPRC")
    for position, mean in zip(positions, means, strict=True):
        axis.text(position, mean + 0.02, format_number(mean, 3), ha="center")
    return _save(figure, destination)


def _precision_recall_curve_figure(
    y_test: pd.Series, probabilities: np.ndarray, threshold: float, destination: Path
) -> Path:
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(6.5, 5))
    axis.plot(recall, precision, color=FRAUD_COLOR, linewidth=2)
    axis.axhline(
        float(y_test.mean()),
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Prevalence (reference)",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title(f"Precision-recall curve on holdout (threshold = {threshold:.3f})")
    axis.legend(loc="upper right")
    return _save(figure, destination)


def _confusion_matrix_figure(
    metrics: dict[str, float | int], destination: Path
) -> Path:
    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate (predicted)", "Fraud (predicted)"],
        yticklabels=["Legitimate (actual)", "Fraud (actual)"],
        ax=axis,
    )
    axis.set_title("Confusion matrix on holdout")
    return _save(figure, destination)


def _importance_figure(importances: pd.Series, destination: Path) -> Path:
    top = importances.head(15).sort_values()
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(top.index, top.values, color=LEGITIMATE_COLOR)
    axis.set_xlabel("Importance")
    axis.set_title("15 most important attributes in the final model")
    return _save(figure, destination)


def generate_model_figures(
    result: dict[str, object],
    test_features: pd.DataFrame,
    y_test: pd.Series,
    directory: Path,
) -> dict[str, Path]:
    """Generate the evaluation figures and return paths by logical name."""
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    model = result["model"]
    importances = feature_importances(model, list(test_features.columns))
    probabilities = result["test_probabilities"]

    return {
        "comparison": _model_comparison_figure(
            result["cross_validation"],
            directory / "modelo_01_comparison.png",
        ),
        "precision_recall_curve": _precision_recall_curve_figure(
            y_test,
            probabilities,
            float(result["decision_threshold"]),
            directory / "modelo_02_precision_recall_curve.png",
        ),
        "confusion_matrix": _confusion_matrix_figure(
            result["holdout_metrics"],
            directory / "modelo_03_confusion_matrix.png",
        ),
        "importance": _importance_figure(
            importances, directory / "modelo_04_feature_importances.png"
        ),
    }


def generate_model_report(
    result: dict[str, object],
    test_features: pd.DataFrame,
    y_test: pd.Series,
    *,
    metadata: dict[str, object],
    figures_directory: Path,
    path: Path,
) -> Path:
    """Generate the figures and the evaluation report in Markdown."""
    figures = generate_model_figures(result, test_features, y_test, figures_directory)
    cross_validation = result["cross_validation"]
    metrics = result["holdout_metrics"]
    model = result["model"]
    importances = feature_importances(model, list(test_features.columns))

    cv_rows = "\n".join(
        f"| {name} | {format_number(values['auprc_mean'], 3)} ± "
        f"{format_number(values['auprc_std'], 3)} | "
        f"{format_number(values['roc_auc_mean'], 3)} | "
        f"{format_number(values['precision_mean'], 3)} | "
        f"{format_number(values['recall_mean'], 3)} | "
        f"{format_number(values['f1_mean'], 3)} |"
        for name, values in cross_validation.items()
    )
    importance_rows = "\n".join(
        f"| {attribute} | {format_number(value, 4)} |"
        for attribute, value in importances.head(10).items()
    )
    prevalence = float(y_test.mean())
    total_test = format_integer(metrics["total_negatives"] + metrics["total_positives"])
    positives_test = format_integer(metrics["total_positives"])
    true_negatives = format_integer(metrics["true_negatives"])
    false_positives = format_integer(metrics["false_positives"])
    false_negatives = format_integer(metrics["false_negatives"])
    true_positives = format_integer(metrics["true_positives"])
    test_proportion = format_percentage(float(metadata["test_proportion"]))

    content = f"""# Relatório de modelagem e avaliação

## 1. Configuração

| Item | Valor |
|---|---|
| Base | MLG/ULB preparada (`data/processed/transactions.parquet`) |
| Attributes | `Time`, `V1`–`V28`, `Amount` (30 attributes) |
| Target | `Class` (1 = fraud) |
| Seed | {metadata["seed"]} |
| Cross-validation | {metadata["folds"]} stratified folds on training |
| Selected model | {result["best_model"]} |
| Decision threshold | {format_number(float(result["decision_threshold"]), 4)} |
| Threshold origin | out-of-fold training probabilities, maximizing F1 |
| Imbalance | `class_weight="balanced"` (no resampling) |
| Split | stratified, {test_proportion} for testing |

## 2. Comparison on cross-validation (training)

| Model | AUPRC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
{cv_rows}

Selection uses the highest mean AUPRC (R-05), not accuracy. In the
cross-validation table, precision, recall, and F1 use the default 0.5 threshold
only for comparison; the final model is evaluated with the threshold chosen on
training.

![Model comparison](figures/{figures["comparison"].name})

## 3. Evaluation on holdout

| Metric | Value |
|---|---|
| AUPRC | {format_number(metrics["auprc"], 4)} |
| ROC-AUC | {format_number(metrics["roc_auc"], 4)} |
| Precision | {format_number(metrics["precision"], 4)} |
| Recall | {format_number(metrics["recall"], 4)} |
| F1 | {format_number(metrics["f1"], 4)} |
| Accuracy (context) | {format_number(metrics["accuracy"], 4)} |

- Holdout prevalence (AUPRC reference): {format_percentage(prevalence)}.
- Test transactions: {total_test}
  ({positives_test} frauds).

| Confusion matrix | Predicted legitimate | Predicted fraud |
|---|---|---|
| Actual legitimate | {true_negatives} | {false_positives} |
| Actual fraud | {false_negatives} | {true_positives} |

![Precision-recall curve](figures/{figures["precision_recall_curve"].name})

![Confusion matrix](figures/{figures["confusion_matrix"].name})

## 4. Associated factors

| Attribute | Importance |
|---|---|
{importance_rows}

![Feature importances](figures/{figures["importance"].name})

Attributes are PCA-anonymized components. Importance indicates contribution to
the model on this base, not a nameable business factor. `Time` is relative and
`Amount` is in the base's original unit.

## 5. Limitations

- Two-day base, relative time, and anonymized attributes: causality cannot be
  interpreted nor seasonality asserted.
- Exact duplicates were removed; the number of frauds is lower than in the raw
  file and this is documented in the quality report.
- Few positives make metrics sensitive to the split; cross-validation reduces,
  but does not eliminate, this instability.
- The chosen threshold maximizes F1; a business-oriented threshold may
  prioritize recall (fewer missed frauds) at the cost of more false positives.
- The model is academic and must not be used for automatic transaction blocking.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
