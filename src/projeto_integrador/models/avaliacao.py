"""Métricas de avaliação e interpretação do modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metricas_classificacao(
    y_verdadeiro: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
    limite: float,
) -> dict[str, float | int]:
    """Calcula as métricas do plano para um limite de decisão explícito."""
    probabilidades = np.asarray(probabilidades, dtype="float64")
    y_verdadeiro = np.asarray(y_verdadeiro, dtype="int64")
    y_previsto = (probabilidades >= limite).astype("int64")

    matriz = confusion_matrix(y_verdadeiro, y_previsto, labels=[0, 1])
    verdadeiros_negativos, falsos_positivos, falsos_negativos, verdadeiros_positivos = (
        matriz.ravel()
    )

    return {
        "limite": float(limite),
        "auprc": float(average_precision_score(y_verdadeiro, probabilidades)),
        "roc_auc": float(roc_auc_score(y_verdadeiro, probabilidades)),
        "precisao": float(precision_score(y_verdadeiro, y_previsto, zero_division=0)),
        "recall": float(recall_score(y_verdadeiro, y_previsto, zero_division=0)),
        "f1": float(f1_score(y_verdadeiro, y_previsto, zero_division=0)),
        "acuracia": float(accuracy_score(y_verdadeiro, y_previsto)),
        "verdadeiros_negativos": int(verdadeiros_negativos),
        "falsos_positivos": int(falsos_positivos),
        "falsos_negativos": int(falsos_negativos),
        "verdadeiros_positivos": int(verdadeiros_positivos),
        "total_negativos": int((y_verdadeiro == 0).sum()),
        "total_positivos": int((y_verdadeiro == 1).sum()),
    }


def importancia_variaveis(modelo, nomes: list[str]) -> pd.Series:
    """Importância das variáveis do estimador final, em ordem decrescente.

    Para árvores, usa ``feature_importances_``; para modelos lineares, usa o
    valor absoluto dos coeficientes. Os atributos são anonimizados por PCA,
    então a leitura deve ser feita com cautela (R-06).
    """
    estimador = modelo.named_steps["classificador"]

    if hasattr(estimador, "feature_importances_"):
        valores = np.asarray(estimador.feature_importances_, dtype="float64")
    elif hasattr(estimador, "coef_"):
        valores = np.abs(np.asarray(estimador.coef_, dtype="float64")).ravel()
    else:
        raise TypeError("O estimador final não expõe feature_importances_ nem coef_.")

    if len(valores) != len(nomes):
        raise ValueError(
            "A quantidade de importâncias não corresponde ao número de atributos."
        )

    return pd.Series(valores, index=nomes, name="importancia").sort_values(
        ascending=False
    )
