"""Teste de fumaça do dashboard (requer a base preparada)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from projeto_integrador.config import CAMINHO_PROCESSADO

APLICATIVO = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "projeto_integrador"
    / "presentation"
    / "app.py"
)

pytestmark = pytest.mark.skipif(
    not CAMINHO_PROCESSADO.exists(),
    reason="Base preparada ausente; execute o pipeline antes deste teste.",
)


def test_dashboard_renderiza_sem_excecoes() -> None:
    aplicativo = AppTest.from_file(str(APLICATIVO), default_timeout=180)
    aplicativo.run()

    assert not aplicativo.exception
    rotulos = [metrica.label for metrica in aplicativo.metric]
    assert rotulos[:3] == [
        "Transações",
        "Transações legítimas",
        "Transações fraudulentas",
    ]
    assert "Taxa de fraude" in rotulos
    assert len(aplicativo.subheader) >= 6


def test_dashboard_filtra_por_status() -> None:
    aplicativo = AppTest.from_file(str(APLICATIVO), default_timeout=180)
    aplicativo.run()
    aplicativo.radio[0].set_value("Somente fraudes").run()

    assert not aplicativo.exception
    assert aplicativo.metric[0].value == "473"
    assert aplicativo.metric[2].value == "473"
