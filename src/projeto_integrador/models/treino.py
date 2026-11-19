"""Treino, seleção de modelo e limite de decisão."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from projeto_integrador.config import NUMERO_FOLDS, SEMENTE
from projeto_integrador.models.avaliacao import metricas_classificacao

METRICAS_VALIDACAO = {
    "auprc": "average_precision",
    "roc_auc": "roc_auc",
    "precisao": "precision",
    "recall": "recall",
    "f1": "f1",
}


def criar_modelos(semente: int = SEMENTE) -> dict[str, Pipeline]:
    """Modelos comparáveis: baseline linear e candidatos de árvores."""
    return {
        "Regressão Logística (baseline)": Pipeline(
            [
                ("padronizacao", StandardScaler()),
                (
                    "classificador",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=semente,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "classificador",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=16,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=semente,
                    ),
                ),
            ]
        ),
        "Gradient Boosting (histogramas)": Pipeline(
            [
                (
                    "classificador",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.1,
                        class_weight="balanced",
                        random_state=semente,
                    ),
                ),
            ]
        ),
    }


def comparar_modelos(
    atributos: pd.DataFrame,
    alvo: pd.Series,
    *,
    folds: int = NUMERO_FOLDS,
    semente: int = SEMENTE,
) -> dict[str, dict[str, float]]:
    """Compara os modelos por validação cruzada estratificada no treino."""
    validacao_cruzada = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=semente
    )
    resultados: dict[str, dict[str, float]] = {}

    for nome, modelo in criar_modelos(semente).items():
        pontuacoes = cross_validate(
            modelo,
            atributos,
            alvo,
            cv=validacao_cruzada,
            scoring=METRICAS_VALIDACAO,
            n_jobs=1,
        )
        resultado: dict[str, float] = {}
        for metrica in METRICAS_VALIDACAO:
            valores = pontuacoes[f"test_{metrica}"]
            resultado[f"{metrica}_media"] = float(np.mean(valores))
            resultado[f"{metrica}_desvio"] = float(np.std(valores))
        resultado["tempo_treino_segundos"] = float(np.mean(pontuacoes["fit_time"]))
        resultados[nome] = resultado

    return resultados


def escolher_melhor_modelo(resultados: dict[str, dict[str, float]]) -> str:
    """Escolhe o modelo com maior AUPRC média (R-05)."""
    if not resultados:
        raise ValueError("Não há resultados de validação cruzada para comparar.")
    return max(resultados, key=lambda nome: resultados[nome]["auprc_media"])


def probabilidades_fora_da_dobra(
    modelo: Pipeline,
    atributos: pd.DataFrame,
    alvo: pd.Series,
    *,
    folds: int = NUMERO_FOLDS,
    semente: int = SEMENTE,
) -> np.ndarray:
    """Probabilidades out-of-fold no conjunto de treino, para o limite."""
    validacao_cruzada = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=semente
    )
    return cross_val_predict(
        modelo,
        atributos,
        alvo,
        cv=validacao_cruzada,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def escolher_limite(
    y_verdadeiro: pd.Series | np.ndarray, probabilidades: np.ndarray
) -> tuple[float, float]:
    """Escolhe o limite que maximiza F1 nas probabilidades informadas.

    O limite é escolhido em probabilidades out-of-fold do treino, nunca no
    holdout, para não contaminar a avaliação final.
    """
    precisao, recall, limites = precision_recall_curve(y_verdadeiro, probabilidades)
    f1 = np.divide(
        2 * precisao[:-1] * recall[:-1],
        precisao[:-1] + recall[:-1],
        out=np.zeros_like(precisao[:-1]),
        where=(precisao[:-1] + recall[:-1]) > 0,
    )
    posicao = int(np.argmax(f1))
    return float(limites[posicao]), float(f1[posicao])


def treinar_e_avaliar(
    atributos_treino: pd.DataFrame,
    alvo_treino: pd.Series,
    atributos_teste: pd.DataFrame,
    alvo_teste: pd.Series,
    *,
    folds: int = NUMERO_FOLDS,
    semente: int = SEMENTE,
) -> dict[str, object]:
    """Compara modelos, escolhe o melhor, define o limite e avalia no holdout."""
    resultados = comparar_modelos(
        atributos_treino, alvo_treino, folds=folds, semente=semente
    )
    melhor_nome = escolher_melhor_modelo(resultados)
    melhor_modelo = criar_modelos(semente)[melhor_nome]

    probabilidades_treino = probabilidades_fora_da_dobra(
        melhor_modelo, atributos_treino, alvo_treino, folds=folds, semente=semente
    )
    limite, f1_treino = escolher_limite(alvo_treino, probabilidades_treino)

    melhor_modelo.fit(atributos_treino, alvo_treino)
    probabilidades_teste = melhor_modelo.predict_proba(atributos_teste)[:, 1]
    metricas_teste = metricas_classificacao(alvo_teste, probabilidades_teste, limite)

    return {
        "validacao_cruzada": resultados,
        "melhor_modelo": melhor_nome,
        "limite_decisao": limite,
        "f1_out_of_fold_treino": f1_treino,
        "metricas_holdout": metricas_teste,
        "modelo": melhor_modelo,
        "probabilidades_teste": probabilidades_teste,
    }


def salvar_modelo(
    modelo: Pipeline, metadados: dict[str, object], caminho: Path
) -> None:
    """Salva o pipeline e seus metadados em um único artefato."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "metadados": metadados}, caminho)


def carregar_modelo(caminho: Path) -> dict[str, object]:
    """Carrega o artefato salvo por :func:`salvar_modelo`."""
    return joblib.load(caminho)
