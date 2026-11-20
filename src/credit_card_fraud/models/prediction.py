"""Inference of the saved model, separated from training."""

from __future__ import annotations

import pandas as pd

from credit_card_fraud.config import MODEL_COLUMNS


def validate_attributes(
    attributes: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Guarantee columns, order, and types required by the trained pipeline."""
    columns = list(metadata.get("columns", MODEL_COLUMNS))
    missing = [column for column in columns if column not in attributes.columns]
    if missing:
        raise ValueError(
            "Missing columns for prediction: " + ", ".join(sorted(missing))
        )

    data = attributes[columns].astype("float64")
    if data.isna().any().any():
        raise ValueError("Prediction does not accept missing values.")
    return data


def predict(
    model,
    metadata: dict[str, object],
    attributes: pd.DataFrame,
) -> dict[str, object]:
    """Calculate probabilities and classes for one or more records.

    Uses exactly the pipeline and threshold registered in the artifact,
    avoiding reimplementation of the training transformation.
    """
    if "decision_threshold" not in metadata:
        raise ValueError("The artifact does not contain the decision threshold.")

    data = validate_attributes(attributes, metadata)
    probabilities = model.predict_proba(data)[:, 1]
    threshold = float(metadata["decision_threshold"])
    classes = (probabilities >= threshold).astype("int64")

    return {
        "model": str(metadata.get("model_selected", "model")),
        "decision_threshold": threshold,
        "probabilities": [float(value) for value in probabilities],
        "classes": [int(value) for value in classes],
    }
