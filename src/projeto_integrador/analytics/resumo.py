"""Indicadores gerais e distribuição da classe alvo."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.config import COLUNA_ALVO, COLUNA_VALOR


def resumo_geral(transacoes: pd.DataFrame) -> dict[str, float | int]:
    """Calcula os indicadores gerais do recorte recebido.

    Devolve zeros quando o recorte está vazio, para que a interface possa
    exibir um estado vazio sem quebrar.
    """
    total = int(len(transacoes))
    if total == 0:
        return {
            "total_transacoes": 0,
            "total_fraudes": 0,
            "total_legitimas": 0,
            "taxa_fraude": 0.0,
            "valor_total": 0.0,
            "valor_fraude": 0.0,
            "valor_legitimo": 0.0,
            "valor_mediano": 0.0,
            "ticket_medio": 0.0,
        }

    fraudes = transacoes[COLUNA_ALVO] == 1
    total_fraudes = int(fraudes.sum())
    valor_total = float(transacoes[COLUNA_VALOR].sum())

    return {
        "total_transacoes": total,
        "total_fraudes": total_fraudes,
        "total_legitimas": total - total_fraudes,
        "taxa_fraude": total_fraudes / total,
        "valor_total": valor_total,
        "valor_fraude": float(transacoes.loc[fraudes, COLUNA_VALOR].sum()),
        "valor_legitimo": float(transacoes.loc[~fraudes, COLUNA_VALOR].sum()),
        "valor_mediano": float(transacoes[COLUNA_VALOR].median()),
        "ticket_medio": float(transacoes[COLUNA_VALOR].mean()),
    }


def distribuicao_classes(transacoes: pd.DataFrame) -> pd.DataFrame:
    """Conta legítimas e fraudes, com a participação de cada classe."""
    total = len(transacoes)
    contagens = transacoes[COLUNA_ALVO].value_counts().reindex([0, 1], fill_value=0)

    return pd.DataFrame(
        {
            "classe": [0, 1],
            "transacoes": [int(contagens[0]), int(contagens[1])],
            "participacao": [
                float(contagens[0] / total) if total else 0.0,
                float(contagens[1] / total) if total else 0.0,
            ],
        }
    )
