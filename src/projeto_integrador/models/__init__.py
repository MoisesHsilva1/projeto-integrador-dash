"""Trilha offline de ML: treino e avaliação."""

from projeto_integrador.models.avaliacao import (
    importancia_variaveis,
    metricas_classificacao,
)
from projeto_integrador.models.treino import (
    carregar_modelo,
    comparar_modelos,
    criar_modelos,
    escolher_limite,
    escolher_melhor_modelo,
    salvar_modelo,
    treinar_e_avaliar,
)

__all__ = [
    "carregar_modelo",
    "comparar_modelos",
    "criar_modelos",
    "escolher_limite",
    "escolher_melhor_modelo",
    "importancia_variaveis",
    "metricas_classificacao",
    "salvar_modelo",
    "treinar_e_avaliar",
]
