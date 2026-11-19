"""Fixtures compartilhadas pelos testes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projeto_integrador.config import COLUNAS_PCA


@pytest.fixture
def transacoes_sinteticas() -> pd.DataFrame:
    """Base pequena com o mesmo esquema da MLG/ULB, para testes rápidos."""
    gerador = np.random.default_rng(42)
    quantidade = 400

    dados: dict[str, np.ndarray] = {
        "Time": np.sort(gerador.uniform(0, 172_800, quantidade)),
        "Amount": np.round(gerador.lognormal(3, 1.2, quantidade), 2),
    }
    for coluna in COLUNAS_PCA:
        dados[coluna] = gerador.normal(size=quantidade)

    alvo = np.zeros(quantidade, dtype="int64")
    alvo[gerador.choice(quantidade, size=12, replace=False)] = 1
    dados["Class"] = alvo

    transacoes = pd.DataFrame(dados)
    transacoes.loc[0, "Amount"] = 0.0
    transacoes.loc[1, "Amount"] = 10.0
    transacoes.loc[2, "Amount"] = 100.0
    transacoes.loc[3, "Amount"] = 1_000.0
    return transacoes
