"""Carga da base bruta e da base preparada, com cache de leitura."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from projeto_integrador.config import CAMINHO_BRUTO, CAMINHO_PROCESSADO
from projeto_integrador.data.schema import validar_esquema

MENSAGEM_BRUTO_AUSENTE = (
    "Base bruta não encontrada em {caminho}. Consulte data/README.md para "
    "colocar o arquivo creditcard.csv em data/raw/."
)
MENSAGEM_PROCESSADO_AUSENTE = (
    "Base preparada não encontrada em {caminho}. Execute a etapa de preparação "
    "(uv run projeto-integrador-pipeline preparar) antes desta etapa."
)


def carregar_brutas(caminho: Path | str | None = None) -> pd.DataFrame:
    """Carrega e valida a base bruta, sem modificá-la."""
    destino = Path(caminho) if caminho is not None else CAMINHO_BRUTO
    if not destino.exists():
        raise FileNotFoundError(MENSAGEM_BRUTO_AUSENTE.format(caminho=destino))

    transacoes = pd.read_csv(destino)
    validar_esquema(transacoes)
    return transacoes


@lru_cache(maxsize=4)
def _carregar_em_cache(caminho: str, _modificado_em: float) -> pd.DataFrame:
    # O argumento ``_modificado_em`` invalida o cache quando o arquivo muda.
    transacoes = pd.read_parquet(caminho)
    validar_esquema(transacoes)
    return transacoes


def carregar_preparadas(caminho: Path | str | None = None) -> pd.DataFrame:
    """Carrega a base tratada em Parquet, com cache.

    O DataFrame retornado é compartilhado entre chamadas: trate-o como
    somente leitura e crie cópias nas camadas que precisarem transformá-lo.
    """
    destino = Path(caminho) if caminho is not None else CAMINHO_PROCESSADO
    if not destino.exists():
        raise FileNotFoundError(MENSAGEM_PROCESSADO_AUSENTE.format(caminho=destino))

    return _carregar_em_cache(str(destino), destino.stat().st_mtime)


@lru_cache(maxsize=4)
def _carregar_metricas_em_cache(caminho: str, _modificado_em: float) -> dict:
    with open(caminho, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def carregar_metricas(caminho: Path | str | None = None) -> dict | None:
    """Carrega as métricas offline do modelo, quando disponíveis.

    Retorna ``None`` se o arquivo não existir, para que a interface exiba um
    estado vazio em vez de quebrar.
    """
    from projeto_integrador.config import CAMINHO_METRICAS

    destino = Path(caminho) if caminho is not None else CAMINHO_METRICAS
    if not destino.exists():
        return None
    return _carregar_metricas_em_cache(str(destino), destino.stat().st_mtime)
