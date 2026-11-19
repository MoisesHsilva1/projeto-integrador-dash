"""Testes de treino, limite e avaliação."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from projeto_integrador.features.dataset import construir_matriz, dividir_treino_teste
from projeto_integrador.models.avaliacao import (
    importancia_variaveis,
    metricas_classificacao,
)
from projeto_integrador.models.treino import (
    carregar_modelo,
    criar_modelos,
    escolher_limite,
    salvar_modelo,
    treinar_e_avaliar,
)


def test_metricas_classificacao_conhecidas() -> None:
    y = np.array([0, 0, 1, 1])
    probabilidades = np.array([0.1, 0.6, 0.4, 0.9])
    metricas = metricas_classificacao(y, probabilidades, 0.5)

    assert metricas["verdadeiros_negativos"] == 1
    assert metricas["falsos_positivos"] == 1
    assert metricas["falsos_negativos"] == 1
    assert metricas["verdadeiros_positivos"] == 1
    assert metricas["precisao"] == 0.5
    assert metricas["recall"] == 0.5
    assert metricas["f1"] == 0.5
    assert metricas["acuracia"] == 0.5
    assert metricas["roc_auc"] == 0.75
    assert 0 < metricas["auprc"] <= 1


def test_escolher_limite_maximiza_f1() -> None:
    y = np.array([0, 0, 1, 1])
    probabilidades = np.array([0.1, 0.2, 0.8, 0.9])

    limite, f1 = escolher_limite(y, probabilidades)

    assert 0.2 < limite <= 0.8
    assert f1 == 1.0


def test_treinar_e_avaliar_holdout(transacoes_sinteticas: pd.DataFrame) -> None:
    atributos, alvo = construir_matriz(transacoes_sinteticas)
    treino, teste, alvo_treino, alvo_teste = dividir_treino_teste(atributos, alvo)
    resultado = treinar_e_avaliar(treino, alvo_treino, teste, alvo_teste, folds=2)

    assert set(resultado["validacao_cruzada"]) == set(criar_modelos())
    assert resultado["melhor_modelo"] in resultado["validacao_cruzada"]
    assert 0.0 <= resultado["limite_decisao"] <= 1.0
    assert len(resultado["probabilidades_teste"]) == len(teste)

    metricas = resultado["metricas_holdout"]
    assert metricas["verdadeiros_positivos"] + metricas["falsos_negativos"] == 2
    assert metricas["verdadeiros_negativos"] + metricas["falsos_positivos"] == 78

    importancias = importancia_variaveis(resultado["modelo"], list(teste.columns))
    assert len(importancias) == len(teste.columns)
    assert importancias.index[0] in teste.columns


def test_salvar_e_carregar_modelo(
    tmp_path: Path, transacoes_sinteticas: pd.DataFrame
) -> None:
    atributos, alvo = construir_matriz(transacoes_sinteticas)
    modelo = criar_modelos()["Regressão Logística (baseline)"]
    modelo.fit(atributos, alvo)

    caminho = tmp_path / "modelo.joblib"
    salvar_modelo(modelo, {"limite_decisao": 0.5}, caminho)
    carregado = carregar_modelo(caminho)

    assert carregado["metadados"]["limite_decisao"] == 0.5
    probabilidades = carregado["modelo"].predict_proba(atributos)[:, 1]
    assert probabilidades.shape == (len(atributos),)
