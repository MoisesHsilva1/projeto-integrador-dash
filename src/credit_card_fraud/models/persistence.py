"""Persistence of the model artifact and environment metadata."""

from __future__ import annotations

import platform
from pathlib import Path

import joblib


def collect_environment() -> dict[str, str]:
    """Relevant versions to reproduce training."""
    import numpy
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "joblib": joblib.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }


def save_model(model, metadata: dict[str, object], path: Path) -> None:
    """Save the pipeline and its metadata in a single artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def load_model(path: Path) -> dict[str, object]:
    """Load the artifact saved by :func:`save_model`."""
    return joblib.load(path)
