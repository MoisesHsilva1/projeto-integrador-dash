"""Figuras e relatório de avaliação do modelo (etapa E7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from projeto_integrador.models.avaliacao import importancia_variaveis  # noqa: E402
from projeto_integrador.relatorios.texto import (  # noqa: E402
    formatar_inteiro,
    formatar_numero,
    formatar_percentual,
)

COR_FRAUDE = "#d62728"
COR_LEGITIMA = "#4c78a8"


def _salvar(figura: plt.Figure, destino: Path) -> Path:
    figura.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)
    return destino


def _figura_comparacao_modelos(
    validacao_cruzada: dict[str, dict[str, float]], destino: Path
) -> Path:
    nomes = list(validacao_cruzada)
    medias = [validacao_cruzada[nome]["auprc_media"] for nome in nomes]
    desvios = [validacao_cruzada[nome]["auprc_desvio"] for nome in nomes]
    posicoes = np.arange(len(nomes))

    figura, eixo = plt.subplots(figsize=(10, 4.5))
    eixo.bar(posicoes, medias, yerr=desvios, capsize=6, color=COR_LEGITIMA)
    eixo.set_xticks(posicoes, nomes, rotation=12, ha="right")
    eixo.set_ylabel("AUPRC média (validação cruzada)")
    eixo.set_ylim(0, 1)
    eixo.set_title("Comparação dos modelos por AUPRC")
    for posicao, media in zip(posicoes, medias, strict=True):
        eixo.text(posicao, media + 0.02, formatar_numero(media, 3), ha="center")
    return _salvar(figura, destino)


def _figura_curva_precisao_recall(
    alvo_teste: pd.Series, probabilidades: np.ndarray, limite: float, destino: Path
) -> Path:
    precisao, recall, _ = precision_recall_curve(alvo_teste, probabilidades)
    figura, eixo = plt.subplots(figsize=(6.5, 5))
    eixo.plot(recall, precisao, color=COR_FRAUDE, linewidth=2)
    eixo.axhline(
        float(alvo_teste.mean()),
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Prevalência (referência)",
    )
    eixo.set_xlabel("Recall")
    eixo.set_ylabel("Precisão")
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.02)
    eixo.set_title(f"Curva precisão-recall no holdout (limite = {limite:.3f})")
    eixo.legend(loc="upper right")
    return _salvar(figura, destino)


def _figura_matriz_confusao(metricas: dict[str, float | int], destino: Path) -> Path:
    matriz = np.array(
        [
            [metricas["verdadeiros_negativos"], metricas["falsos_positivos"]],
            [metricas["falsos_negativos"], metricas["verdadeiros_positivos"]],
        ]
    )
    figura, eixo = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legítima (prevista)", "Fraude (prevista)"],
        yticklabels=["Legítima (real)", "Fraude (real)"],
        ax=eixo,
    )
    eixo.set_title("Matriz de confusão no holdout")
    return _salvar(figura, destino)


def _figura_importancia(importancias: pd.Series, destino: Path) -> Path:
    topo = importancias.head(15).sort_values()
    figura, eixo = plt.subplots(figsize=(8, 5.5))
    eixo.barh(topo.index, topo.values, color=COR_LEGITIMA)
    eixo.set_xlabel("Importância")
    eixo.set_title("15 atributos mais importantes no modelo final")
    return _salvar(figura, destino)


def gerar_figuras_modelo(
    resultado: dict[str, object],
    atributos_teste: pd.DataFrame,
    alvo_teste: pd.Series,
    diretorio: Path,
) -> dict[str, Path]:
    """Gera as figuras da avaliação e devolve os caminhos por nome lógico."""
    diretorio.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    modelo = resultado["modelo"]
    importancias = importancia_variaveis(modelo, list(atributos_teste.columns))
    probabilidades = resultado["probabilidades_teste"]

    return {
        "comparacao": _figura_comparacao_modelos(
            resultado["validacao_cruzada"],
            diretorio / "modelo_01_comparacao.png",
        ),
        "curva_pr": _figura_curva_precisao_recall(
            alvo_teste,
            probabilidades,
            float(resultado["limite_decisao"]),
            diretorio / "modelo_02_curva_precisao_recall.png",
        ),
        "matriz_confusao": _figura_matriz_confusao(
            resultado["metricas_holdout"],
            diretorio / "modelo_03_matriz_confusao.png",
        ),
        "importancia": _figura_importancia(
            importancias, diretorio / "modelo_04_importancia_variaveis.png"
        ),
    }


def gerar_relatorio_modelo(
    resultado: dict[str, object],
    atributos_teste: pd.DataFrame,
    alvo_teste: pd.Series,
    *,
    metadados: dict[str, object],
    diretorio_figuras: Path,
    caminho: Path,
) -> Path:
    """Gera as figuras e o relatório de avaliação em Markdown."""
    figuras = gerar_figuras_modelo(
        resultado, atributos_teste, alvo_teste, diretorio_figuras
    )
    validacao_cruzada = resultado["validacao_cruzada"]
    validacao_cruzada = resultado["validacao_cruzada"]
    metricas = resultado["metricas_holdout"]
    modelo = resultado["modelo"]
    importancias = importancia_variaveis(modelo, list(atributos_teste.columns))

    linhas_cv = "\n".join(
        f"| {nome} | {formatar_numero(valores['auprc_media'], 3)} ± "
        f"{formatar_numero(valores['auprc_desvio'], 3)} | "
        f"{formatar_numero(valores['roc_auc_media'], 3)} | "
        f"{formatar_numero(valores['precisao_media'], 3)} | "
        f"{formatar_numero(valores['recall_media'], 3)} | "
        f"{formatar_numero(valores['f1_media'], 3)} |"
        for nome, valores in validacao_cruzada.items()
    )
    linhas_importancia = "\n".join(
        f"| {atributo} | {formatar_numero(valor, 4)} |"
        for atributo, valor in importancias.head(10).items()
    )
    prevalencia = float(alvo_teste.mean())
    total_teste = formatar_inteiro(
        metricas["total_negativos"] + metricas["total_positivos"]
    )
    positivos_teste = formatar_inteiro(metricas["total_positivos"])
    verdadeiros_negativos = formatar_inteiro(metricas["verdadeiros_negativos"])
    falsos_positivos = formatar_inteiro(metricas["falsos_positivos"])
    falsos_negativos = formatar_inteiro(metricas["falsos_negativos"])
    verdadeiros_positivos = formatar_inteiro(metricas["verdadeiros_positivos"])
    proporcao_teste = formatar_percentual(float(metadados["proporcao_teste"]))

    conteudo = f"""# Relatório de modelagem e avaliação

## 1. Configuração

| Item | Valor |
|---|---|
| Base | MLG/ULB preparada (`data/processed/transacoes.parquet`) |
| Atributos | `Time`, `V1`–`V28`, `Amount` (30 atributos) |
| Alvo | `Class` (1 = fraude) |
| Semente | {metadados["semente"]} |
| Validação cruzada | {metadados["folds"]} dobras estratificadas no treino |
| Modelo selecionado | {resultado["melhor_modelo"]} |
| Limite de decisão | {formatar_numero(float(resultado["limite_decisao"]), 4)} |
| Origem do limite | probabilidades out-of-fold do treino, maximizando F1 |
| Desbalanceamento | `class_weight="balanced"` (sem reamostragem) |
| Split | estratificado, {proporcao_teste} para teste |

## 2. Comparação na validação cruzada (treino)

| Modelo | AUPRC | ROC-AUC | Precisão | Recall | F1 |
|---|---|---|---|---|---|
{linhas_cv}

A seleção usa a maior AUPRC média (R-05), não a acurácia. Na tabela de validação
cruzada, precisão, recall e F1 usam o limiar padrão de 0,5 apenas para comparação;
o modelo final é avaliado com o limiar escolhido no treino.

![Comparação dos modelos](figures/{figuras["comparacao"].name})

## 3. Avaliação no holdout

| Métrica | Valor |
|---|---|
| AUPRC | {formatar_numero(metricas["auprc"], 4)} |
| ROC-AUC | {formatar_numero(metricas["roc_auc"], 4)} |
| Precisão | {formatar_numero(metricas["precisao"], 4)} |
| Recall | {formatar_numero(metricas["recall"], 4)} |
| F1 | {formatar_numero(metricas["f1"], 4)} |
| Acurácia (contexto) | {formatar_numero(metricas["acuracia"], 4)} |

- Prevalência no holdout (AUPRC de referência): {formatar_percentual(prevalencia)}.
- Transações de teste: {total_teste}
  ({positivos_teste} fraudes).

| Matriz de confusão | Previsto legítima | Previsto fraude |
|---|---|---|
| Real legítima | {verdadeiros_negativos} | {falsos_positivos} |
| Real fraude | {falsos_negativos} | {verdadeiros_positivos} |

![Curva precisão-recall](figures/{figuras["curva_pr"].name})

![Matriz de confusão](figures/{figuras["matriz_confusao"].name})

## 4. Fatores associados

| Atributo | Importância |
|---|---|
{linhas_importancia}

![Importância das variáveis](figures/{figuras["importancia"].name})

Os atributos são componentes PCA anonimizados. A importância indica contribuição
para o modelo nesta base, não um fator de negócio nomeável. `Time` é relativo e
`Amount` está na unidade original da base.

## 5. Limitações

- Base com dois dias, tempo relativo e atributos anonimizados: não há como
  interpretar causalidade nem afirmar sazonalidade.
- As duplicatas exatas foram removidas; o número de fraudes é menor que o do
  arquivo bruto e isso está documentado no relatório de qualidade.
- Poucos positivos tornam as métricas sensíveis ao split; a validação cruzada
  reduz, mas não elimina, essa instabilidade.
- O limite escolhido maximiza F1; um limite orientado a negócio pode privilegiar
  recall (menos fraudes perdidas) ao custo de mais falsos positivos.
- O modelo é acadêmico e não deve ser usado para bloqueio automático de
  transações.
"""

    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho
