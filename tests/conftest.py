"""Shared fixtures across tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning

from credit_card_fraud.config import PCA_COLUMNS


@pytest.fixture(autouse=True)
def suppress_known_third_party_warnings() -> None:  # type: ignore[return]
    """Suppress known deprecation warnings emitted by third-party libs.

    - ``UndefinedMetricWarning``: expected when tiny synthetic datasets produce
      no positive predictions in some folds.
    - ``MatplotlibDeprecationWarning``: seaborn's internal ``bxp(vert=...)``
      call, suppressed until seaborn completes its own migration. Our code
      already passes ``orient="v"`` explicitly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.filterwarnings(
            "ignore",
            message="vert: bool was deprecated",
            category=DeprecationWarning,
        )
        yield


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """Small base with the same schema as MLG/ULB, for fast tests."""
    generator = np.random.default_rng(42)
    quantity = 400

    data: dict[str, np.ndarray] = {
        "Time": np.sort(generator.uniform(0, 172_800, quantity)),
        "Amount": np.round(generator.lognormal(3, 1.2, quantity), 2),
    }
    for column in PCA_COLUMNS:
        data[column] = generator.normal(size=quantity)

    target = np.zeros(quantity, dtype="int64")
    target[generator.choice(quantity, size=12, replace=False)] = 1
    data["Class"] = target

    transactions = pd.DataFrame(data)
    transactions.loc[0, "Amount"] = 0.0
    transactions.loc[1, "Amount"] = 10.0
    transactions.loc[2, "Amount"] = 100.0
    transactions.loc[3, "Amount"] = 1_000.0
    return transactions
