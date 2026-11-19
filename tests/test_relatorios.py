"""Testes da geração de relatórios e figuras."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projeto_integrador.data.auditoria import auditar, preparar_transacoes
from projeto_integrador.relatorios.eda import gerar_relatorio_eda
from projeto_integrador.relatorios.qualidade import gerar_relatorio_qualidade


def test_relatorio_de_qualidade(
    tmp_path: Path, transacoes_sinteticas: pd.DataFrame
) -> None:
    auditoria = auditar(transacoes_sinteticas)
    _, registro = preparar_transacoes(transacoes_sinteticas)
    caminho = gerar_relatorio_qualidade(
        auditoria,
        registro,
        checksum="abc123",
        caminho=tmp_path / "qualidade.md",
    )

    texto = caminho.read_text(encoding="utf-8")
    assert "abc123" in texto
    assert "284" not in texto
    assert "Fraudes" in texto or "fraudes" in texto
    assert "Tratamento aplicado" in texto


def test_relatorio_de_eda(tmp_path: Path, transacoes_sinteticas: pd.DataFrame) -> None:
    caminho = gerar_relatorio_eda(
        transacoes_sinteticas,
        diretorio_figuras=tmp_path / "figures",
        caminho=tmp_path / "eda.md",
    )

    texto = caminho.read_text(encoding="utf-8")
    assert "OE-1" in texto
    assert "OE-2" in texto
    assert "OE-3" in texto
    assert "400" in texto

    figuras = sorted((tmp_path / "figures").glob("eda_*.png"))
    assert len(figuras) == 6
    assert all(figura.stat().st_size > 0 for figura in figuras)
