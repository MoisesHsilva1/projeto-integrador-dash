"""Auditoria de qualidade e preparação da base MLG/ULB."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projeto_integrador.config import COLUNA_ALVO, COLUNA_TEMPO, COLUNA_VALOR
from projeto_integrador.data.schema import validar_esquema


def auditar(df: pd.DataFrame) -> dict[str, object]:
    """Resume a qualidade da base, sem alterá-la."""
    validar_esquema(df)

    fraudes = int(df[COLUNA_ALVO].sum())
    duplicadas = df.duplicated()
    sem_duplicatas = df.loc[~duplicadas]
    duracao_horas = (df[COLUNA_TEMPO].max() - df[COLUNA_TEMPO].min()) / 3600

    return {
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "valores_ausentes": int(df.isna().sum().sum()),
        "duplicatas_exatas": int(duplicadas.sum()),
        "duplicatas_fraudes": int(df.loc[duplicadas, COLUNA_ALVO].sum()),
        "fraudes": fraudes,
        "legitimas": int(len(df) - fraudes),
        "taxa_fraude": float(fraudes / len(df)),
        "inicio_segundos": float(df[COLUNA_TEMPO].min()),
        "fim_segundos": float(df[COLUNA_TEMPO].max()),
        "duracao_horas": float(duracao_horas),
        "valor_total": float(df[COLUNA_VALOR].sum()),
        "valor_mediano": float(df[COLUNA_VALOR].median()),
        "valor_minimo": float(df[COLUNA_VALOR].min()),
        "valor_maximo": float(df[COLUNA_VALOR].max()),
        "valor_zero_transacoes": int((df[COLUNA_VALOR] == 0).sum()),
        "valor_zero_fraudes": int(df.loc[df[COLUNA_VALOR] == 0, COLUNA_ALVO].sum()),
        "sem_duplicatas": {
            "linhas": int(len(sem_duplicatas)),
            "fraudes": int(sem_duplicatas[COLUNA_ALVO].sum()),
            "legitimas": int(len(sem_duplicatas) - sem_duplicatas[COLUNA_ALVO].sum()),
            "taxa_fraude": float(sem_duplicatas[COLUNA_ALVO].mean()),
        },
    }


def preparar_transacoes(
    df: pd.DataFrame, *, remover_duplicatas: bool = True
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aplica as regras de limpeza documentadas e registra o impacto.

    Regras:
    - remove duplicatas exatas (decisão D-11), registrando quantas fraudes
      foram afetadas;
    - não remove outliers nem transações de valor zero: fraudes podem estar
      nesses extremos (Seção 4.6 do plano);
    - não altera valores, colunas ou a ordem das linhas.
    """
    validar_esquema(df)

    linhas_originais = len(df)
    fraudes_originais = int(df[COLUNA_ALVO].sum())
    transacoes = df.copy()

    remocoes: list[dict[str, object]] = []
    if remover_duplicatas:
        duplicadas = transacoes.duplicated()
        linhas_removidas = int(duplicadas.sum())
        fraudes_removidas = int(transacoes.loc[duplicadas, COLUNA_ALVO].sum())
        transacoes = transacoes.loc[~duplicadas].reset_index(drop=True)
        remocoes.append(
            {
                "regra": "Remoção de duplicatas exatas (mantida a primeira ocorrência)",
                "linhas_removidas": linhas_removidas,
                "fraudes_removidas": fraudes_removidas,
            }
        )

    registro: dict[str, object] = {
        "linhas_originais": linhas_originais,
        "linhas_finais": int(len(transacoes)),
        "fraudes_originais": fraudes_originais,
        "fraudes_finais": int(transacoes[COLUNA_ALVO].sum()),
        "legitimas_finais": int(len(transacoes) - transacoes[COLUNA_ALVO].sum()),
        "taxa_fraude_final": float(transacoes[COLUNA_ALVO].mean()),
        "remocoes": remocoes,
        "mantidos": [
            "Transações com Amount = 0 (não são valores inválidos na base).",
            "Outliers de Amount: não removidos; fraudes podem estar nos extremos.",
            "Atributos V1–V28: mantidos como componentes PCA anonimizados.",
        ],
    }

    validar_esquema(transacoes)
    return transacoes, registro


def salvar_preparadas(df: pd.DataFrame, caminho: Path | str) -> None:
    """Grava a base tratada em Parquet, criando o diretório se necessário."""
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
