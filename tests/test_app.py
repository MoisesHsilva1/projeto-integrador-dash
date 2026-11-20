"""Dashboard smoke test (requires the prepared base)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from credit_card_fraud.config import PROCESSED_PATH

DASHBOARD_APP = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "credit_card_fraud"
    / "presentation"
    / "app.py"
)

pytestmark = pytest.mark.skipif(
    not PROCESSED_PATH.exists(),
    reason="Prepared base missing; run the pipeline before this test.",
)


def test_dashboard_renders_without_exceptions() -> None:
    app = AppTest.from_file(str(DASHBOARD_APP), default_timeout=180)
    app.run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert labels[:3] == [
        "Transações",
        "Transações legítimas",
        "Transações fraudulentas",
    ]
    assert "Taxa de fraude" in labels
    assert len(app.subheader) >= 6


def test_dashboard_filters_by_status() -> None:
    app = AppTest.from_file(str(DASHBOARD_APP), default_timeout=180)
    app.run()
    app.radio[0].set_value("Somente fraudes").run()

    assert not app.exception
    assert app.metric[0].value == "473"
    assert app.metric[2].value == "473"
