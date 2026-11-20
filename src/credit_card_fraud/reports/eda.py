"""Figures and exploratory analysis report (step E4)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from credit_card_fraud.analytics.correlation import correlation_with_target  # noqa: E402
from credit_card_fraud.analytics.summary import (  # noqa: E402
    class_distribution,
    general_summary,
)
from credit_card_fraud.analytics.temporal import (  # noqa: E402
    add_relative_time,
    temporal_evolution,
    temporal_heatmap,
)
from credit_card_fraud.analytics.value import frauds_by_value_band  # noqa: E402
from credit_card_fraud.config import AMOUNT_COLUMN, TARGET_COLUMN  # noqa: E402
from credit_card_fraud.reports.text import (  # noqa: E402
    format_integer,
    format_number,
    format_percentage,
)

CLASS_COLORS = {0: "#4c78a8", 1: "#d62728"}
CLASS_LABELS = {0: "Legítima", 1: "Fraude"}


def _save(figure: plt.Figure, destination: Path) -> Path:
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def _class_distribution_figure(transactions: pd.DataFrame, destination: Path) -> Path:
    counts = transactions[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].barh(
        [CLASS_LABELS[0], CLASS_LABELS[1]],
        counts.values,
        color=[CLASS_COLORS[0], CLASS_COLORS[1]],
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Transactions (log scale)")
    axes[0].set_title("Count by class")
    for position, value in enumerate(counts.values):
        axes[0].text(max(value, 1), position, f" {format_integer(value)}", va="center")

    axes[1].pie(
        counts.values,
        labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
        colors=[CLASS_COLORS[0], CLASS_COLORS[1]],
        autopct="%1.2f%%",
        startangle=90,
    )
    axes[1].set_title("Class participation")
    return _save(figure, destination)


def _value_by_class_figure(transactions: pd.DataFrame, destination: Path) -> Path:
    data = transactions.assign(
        amount_log=np.log1p(transactions[AMOUNT_COLUMN]),
        class_=transactions[TARGET_COLUMN].map(CLASS_LABELS),
    )
    palette = {CLASS_LABELS[0]: CLASS_COLORS[0], CLASS_LABELS[1]: CLASS_COLORS[1]}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(
        data=data,
        x="amount_log",
        hue="class_",
        bins=60,
        stat="density",
        common_norm=False,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Value distribution by class")

    sns.boxplot(
        data=data,
        x="class_",
        y=AMOUNT_COLUMN,
        hue="class_",
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Amount (log scale, original unit)")
    axes[1].set_title("Value by class")
    return _save(figure, destination)


def _frauds_by_value_band_figure(transactions: pd.DataFrame, destination: Path) -> Path:
    table = frauds_by_value_band(transactions)
    positions = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(10, 4.5))

    axis.bar(positions, table["frauds"], color=CLASS_COLORS[1])
    axis.set_xticks(positions, table["value_band"])
    axis.set_xlabel("Value band (original unit)")
    axis.set_ylabel("Frauds")
    axis.set_title("Frauds and fraud rate by value band")

    rate_axis = axis.twinx()
    rate_axis.plot(
        positions,
        table["fraud_rate"] * 100,
        color="#222222",
        marker="o",
        linewidth=1.5,
    )
    rate_axis.set_ylabel("Fraud rate (%)")
    return _save(figure, destination)


def _temporal_evolution_figure(transactions: pd.DataFrame, destination: Path) -> Path:
    series = temporal_evolution(transactions)
    figure, axis = plt.subplots(figsize=(11, 4.5))

    axis.bar(
        series["elapsed_hour"],
        series["frauds"],
        color=CLASS_COLORS[1],
        label="Frauds",
    )
    axis.set_xlabel("Elapsed hour since the first transaction")
    axis.set_ylabel("Frauds")

    volume_axis = axis.twinx()
    volume_axis.plot(
        series["elapsed_hour"],
        series["transactions"],
        color=CLASS_COLORS[0],
        alpha=0.7,
        linewidth=1.5,
        label="Transactions",
    )
    volume_axis.set_ylabel("Transactions")

    handles, labels = axis.get_legend_handles_labels()
    volume_handles, volume_labels = volume_axis.get_legend_handles_labels()
    axis.legend(handles + volume_handles, labels + volume_labels, loc="upper left")
    axis.set_title("Relative temporal evolution (1-hour blocks)")
    return _save(figure, destination)


def _temporal_heatmap_figure(transactions: pd.DataFrame, destination: Path) -> Path:
    matrix = temporal_heatmap(transactions)
    figure, axis = plt.subplots(figsize=(12, 3.2))
    sns.heatmap(
        matrix,
        cmap="Reds",
        ax=axis,
        cbar_kws={"label": "Frauds"},
        linewidths=0.3,
        linecolor="white",
    )
    axis.set_xlabel("Hour within the relative 24-hour block")
    axis.set_ylabel("Relative day")
    axis.set_title("Fraud heatmap by relative day and hour block")
    return _save(figure, destination)


def _correlation_figure(transactions: pd.DataFrame, destination: Path) -> Path:
    correlations = correlation_with_target(transactions).head(12).sort_values()
    colors = [
        CLASS_COLORS[1] if value > 0 else CLASS_COLORS[0] for value in correlations
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(correlations.index, correlations.values, color=colors)
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set_xlabel("Pearson correlation with the fraud label")
    axis.set_title("Attributes most correlated with fraud")
    return _save(figure, destination)


def generate_figures(transactions: pd.DataFrame, directory: Path) -> dict[str, Path]:
    """Generate the EDA figures and return paths by logical name."""
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    return {
        "class_distribution": _class_distribution_figure(
            transactions, directory / "eda_01_class_distribution.png"
        ),
        "value_by_class": _value_by_class_figure(
            transactions, directory / "eda_02_value_by_class.png"
        ),
        "value_bands": _frauds_by_value_band_figure(
            transactions, directory / "eda_03_frauds_by_value_band.png"
        ),
        "temporal_evolution": _temporal_evolution_figure(
            transactions, directory / "eda_04_temporal_evolution.png"
        ),
        "temporal_heatmap": _temporal_heatmap_figure(
            transactions, directory / "eda_05_temporal_heatmap.png"
        ),
        "correlation": _correlation_figure(
            transactions, directory / "eda_06_correlation_target.png"
        ),
    }


def generate_eda_report(
    transactions: pd.DataFrame, *, figures_directory: Path, path: Path
) -> Path:
    """Generate the figures and the EDA report in Markdown."""
    figures = generate_figures(transactions, figures_directory)

    summary = general_summary(transactions)
    distribution = class_distribution(transactions)
    bands = frauds_by_value_band(transactions)
    series = temporal_evolution(transactions)
    correlations = correlation_with_target(transactions)
    with_time = add_relative_time(transactions)

    frauds = transactions.loc[transactions[TARGET_COLUMN] == 1]
    legitimate = transactions.loc[transactions[TARGET_COLUMN] == 0]
    up_to_250 = float((frauds[AMOUNT_COLUMN] <= 250).mean())
    up_to_100 = float((frauds[AMOUNT_COLUMN] <= 100).mean())

    band_rows = "\n".join(
        f"| {row.value_band} | {format_integer(row.transactions)} | "
        f"{format_integer(row.frauds)} | "
        f"{format_percentage(row.fraud_rate)} | "
        f"{format_number(row.total_amount)} |"
        for row in bands.itertuples()
    )
    correlation_rows = "\n".join(
        f"| {attribute} | {format_number(value, 3)} |"
        for attribute, value in correlations.head(8).items()
    )
    hours_with_fraud = series.loc[series["frauds"] > 0]
    peak_hour = (
        int(hours_with_fraud.loc[hours_with_fraud["frauds"].idxmax(), "elapsed_hour"])
        if not hours_with_fraud.empty
        else 0
    )
    frauds_by_day = (
        with_time.loc[with_time[TARGET_COLUMN] == 1]
        .groupby("relative_day", observed=True)
        .size()
        .to_dict()
    )
    day_rows = "\n".join(
        f"| {day} | {format_integer(quantity)} |"
        for day, quantity in frauds_by_day.items()
    )
    legitimate_participation = format_percentage(
        float(distribution.loc[0, "participation"])
    )
    fraud_participation = format_percentage(float(distribution.loc[1, "participation"]))
    median_frauds = format_number(float(frauds[AMOUNT_COLUMN].median()))
    median_legitimate = format_number(float(legitimate[AMOUNT_COLUMN].median()))
    mean_frauds = format_number(float(frauds[AMOUNT_COLUMN].mean()))
    mean_legitimate = format_number(float(legitimate[AMOUNT_COLUMN].mean()))
    zero_frauds = format_integer(int((frauds[AMOUNT_COLUMN] == 0).sum()))
    zero_legitimate = format_integer(int((legitimate[AMOUNT_COLUMN] == 0).sum()))
    legitimate_table = format_integer(distribution.loc[0, "transactions"])
    frauds_table = format_integer(distribution.loc[1, "transactions"])

    content = f"""# Relatório de análise exploratória

**Base:** MLG/ULB — *Credit Card Fraud Detection*

**Arquivo preparado:** `data/processed/transactions.parquet`

**Recorte:** {format_integer(summary["total_transactions"])} transações,
sendo {format_integer(summary["total_frauds"])} fraudes
({format_percentage(summary["fraud_rate"])}).

> **Limitações da base:** dois dias de transações, tempo relativo, atributos
> anonimizados por PCA e moeda não fixada no dicionário. As análises de tempo
> usam blocos relativos; não há data absoluta nem hora do dia confirmada.

## 1. Estatísticas gerais

| Indicador | Valor |
|---|---|
| Transações | {format_integer(summary["total_transactions"])} |
| Legítimas | {format_integer(summary["total_legitimate"])} |
| Fraudulentas | {format_integer(summary["total_frauds"])} |
| Taxa de fraude | {format_percentage(summary["fraud_rate"])} |
| Valor total (unidade original) | {format_number(summary["total_amount"])} |
| Valor em fraudes | {format_number(summary["fraud_amount"])} |
| Valor mediano | {format_number(summary["median_amount"])} |
| Ticket médio | {format_number(summary["average_ticket"])} |

### Distribuição das classes

| Classe | Transações | Participação |
|---|---|---|
| Legítima | {legitimate_table} | {legitimate_participation} |
| Fraude | {frauds_table} | {fraud_participation} |

![Distribuição das classes](figures/{figures["class_distribution"].name})

## 2. OE-1 — Padrões e características das fraudes

| Indicador | Fraudulentas | Legítimas |
|---|---|---|
| Transações | {format_integer(len(frauds))} | {format_integer(len(legitimate))} |
| Valor mediano | {median_frauds} | {median_legitimate} |
| Valor médio | {mean_frauds} | {mean_legitimate} |
| Transações com valor zero | {zero_frauds} | {zero_legitimate} |

**Atributos mais correlacionados com o rótulo de fraude:**

| Atributo | Correlação |
|---|---|
{correlation_rows}

![Valor por classe](figures/{figures["value_by_class"].name})

![Correlações](figures/{figures["correlation"].name})

**Leitura:** os atributos são componentes PCA anonimizados. A correlação indica
associação estatística na base, não um fator de negócio nomeável (R-06).

## 3. OE-2 — Fraudes por faixa de valor

| Faixa | Transações | Fraudes | Taxa de fraude | Valor movimentado |
|---|---|---|---|---|
{band_rows}

- {format_percentage(up_to_250)} das fraudes têm valor até 250.
- {format_percentage(up_to_100)} das fraudes têm valor até 100.

![Fraudes por faixa de valor](figures/{figures["value_bands"].name})

## 4. OE-3 — Fraudes ao longo do tempo

A base cobre {format_number(with_time["elapsed_hour"].max() + 1, 0)} blocos de
1 hora desde a primeira transação. O horário de pico de fraudes é o bloco
{format_integer(peak_hour)} (hora decorrida).

| Dia relativo | Fraudes |
|---|---|
{day_rows}

![Evolução temporal](figures/{figures["temporal_evolution"].name})

![Mapa de calor](figures/{figures["temporal_heatmap"].name})

## 5. Achados para a modelagem

- Desbalanceamento extremo: a classe positiva é rara, o que exige AUPRC,
  precision/recall e matriz de confusão — acurácia não é suficiente (R-03).
- `Amount` é assimétrico; a transformação visual usada é `log1p`, mas o modelo
  recebe o valor original padronizado no pipeline.
- As figuras temporais são relativas; nenhuma conclusão sobre sazonalidade ou
  horário real pode ser feita.
- Os valores extremos de `Amount` não foram removidos: fraudes aparecem nos
  extremos e a remoção poderia eliminar sinal.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
