"""Figuras e relatório da análise exploratória (etapa E4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from projeto_integrador.analytics.correlacao import correlacao_com_alvo  # noqa: E402
from projeto_integrador.analytics.resumo import (  # noqa: E402
    distribuicao_classes,
    resumo_geral,
)
from projeto_integrador.analytics.temporal import (  # noqa: E402
    adicionar_tempo_relativo,
    calor_temporal,
    evolucao_temporal,
)
from projeto_integrador.analytics.valor import fraudes_por_faixa_valor  # noqa: E402
from projeto_integrador.config import COLUNA_ALVO, COLUNA_VALOR  # noqa: E402
from projeto_integrador.relatorios.texto import (  # noqa: E402
    formatar_inteiro,
    formatar_numero,
    formatar_percentual,
)

CORES_CLASSE = {0: "#4c78a8", 1: "#d62728"}
ROTULOS_CLASSE = {0: "Legítima", 1: "Fraude"}


def _salvar(figura: plt.Figure, destino: Path) -> Path:
    figura.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)
    return destino


def _figura_distribuicao_classes(transacoes: pd.DataFrame, destino: Path) -> Path:
    contagens = transacoes[COLUNA_ALVO].value_counts().reindex([0, 1], fill_value=0)
    figura, eixos = plt.subplots(1, 2, figsize=(11, 4))

    eixos[0].barh(
        [ROTULOS_CLASSE[0], ROTULOS_CLASSE[1]],
        contagens.values,
        color=[CORES_CLASSE[0], CORES_CLASSE[1]],
    )
    eixos[0].set_xscale("log")
    eixos[0].set_xlabel("Transações (escala log)")
    eixos[0].set_title("Contagem por classe")
    for posicao, valor in enumerate(contagens.values):
        eixos[0].text(
            max(valor, 1), posicao, f" {formatar_inteiro(valor)}", va="center"
        )

    eixos[1].pie(
        contagens.values,
        labels=[ROTULOS_CLASSE[0], ROTULOS_CLASSE[1]],
        colors=[CORES_CLASSE[0], CORES_CLASSE[1]],
        autopct="%1.2f%%",
        startangle=90,
    )
    eixos[1].set_title("Participação das classes")
    return _salvar(figura, destino)


def _figura_valor_por_classe(transacoes: pd.DataFrame, destino: Path) -> Path:
    dados = transacoes.assign(
        valor_log=np.log1p(transacoes[COLUNA_VALOR]),
        classe=transacoes[COLUNA_ALVO].map(ROTULOS_CLASSE),
    )
    paleta = {ROTULOS_CLASSE[0]: CORES_CLASSE[0], ROTULOS_CLASSE[1]: CORES_CLASSE[1]}
    figura, eixos = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(
        data=dados,
        x="valor_log",
        hue="classe",
        bins=60,
        stat="density",
        common_norm=False,
        palette=paleta,
        ax=eixos[0],
    )
    eixos[0].set_xlabel("log(1 + Amount)")
    eixos[0].set_ylabel("Densidade")
    eixos[0].set_title("Distribuição do valor por classe")

    sns.boxplot(
        data=dados,
        x="classe",
        y=COLUNA_VALOR,
        hue="classe",
        palette=paleta,
        legend=False,
        ax=eixos[1],
    )
    eixos[1].set_yscale("log")
    eixos[1].set_xlabel("")
    eixos[1].set_ylabel("Amount (escala log, unidade original)")
    eixos[1].set_title("Valor por classe")
    return _salvar(figura, destino)


def _figura_fraudes_por_faixa(transacoes: pd.DataFrame, destino: Path) -> Path:
    tabela = fraudes_por_faixa_valor(transacoes)
    posicoes = np.arange(len(tabela))
    figura, eixo = plt.subplots(figsize=(10, 4.5))

    eixo.bar(posicoes, tabela["fraudes"], color=CORES_CLASSE[1])
    eixo.set_xticks(posicoes, tabela["faixa_valor"])
    eixo.set_xlabel("Faixa de valor (unidade original)")
    eixo.set_ylabel("Fraudes")
    eixo.set_title("Fraudes e taxa de fraude por faixa de valor")

    eixo_taxa = eixo.twinx()
    eixo_taxa.plot(
        posicoes,
        tabela["taxa_fraude"] * 100,
        color="#222222",
        marker="o",
        linewidth=1.5,
    )
    eixo_taxa.set_ylabel("Taxa de fraude (%)")
    return _salvar(figura, destino)


def _figura_evolucao_temporal(transacoes: pd.DataFrame, destino: Path) -> Path:
    serie = evolucao_temporal(transacoes)
    figura, eixo = plt.subplots(figsize=(11, 4.5))

    eixo.bar(
        serie["hora_decorrida"],
        serie["fraudes"],
        color=CORES_CLASSE[1],
        label="Fraudes",
    )
    eixo.set_xlabel("Hora decorrida desde a primeira transação")
    eixo.set_ylabel("Fraudes")

    eixo_volume = eixo.twinx()
    eixo_volume.plot(
        serie["hora_decorrida"],
        serie["transacoes"],
        color=CORES_CLASSE[0],
        alpha=0.7,
        linewidth=1.5,
        label="Transações",
    )
    eixo_volume.set_ylabel("Transações")

    alças, rotulos = eixo.get_legend_handles_labels()
    alças_volume, rotulos_volume = eixo_volume.get_legend_handles_labels()
    eixo.legend(alças + alças_volume, rotulos + rotulos_volume, loc="upper left")
    eixo.set_title("Evolução temporal relativa (blocos de 1 hora)")
    return _salvar(figura, destino)


def _figura_calor_temporal(transacoes: pd.DataFrame, destino: Path) -> Path:
    matriz = calor_temporal(transacoes)
    figura, eixo = plt.subplots(figsize=(12, 3.2))
    sns.heatmap(
        matriz,
        cmap="Reds",
        ax=eixo,
        cbar_kws={"label": "Fraudes"},
        linewidths=0.3,
        linecolor="white",
    )
    eixo.set_xlabel("Hora dentro do bloco relativo de 24 h")
    eixo.set_ylabel("Dia relativo")
    eixo.set_title("Mapa de calor de fraudes por dia e bloco de hora relativos")
    return _salvar(figura, destino)


def _figura_correlacao(transacoes: pd.DataFrame, destino: Path) -> Path:
    correlacoes = correlacao_com_alvo(transacoes).head(12).sort_values()
    cores = [CORES_CLASSE[1] if valor > 0 else CORES_CLASSE[0] for valor in correlacoes]
    figura, eixo = plt.subplots(figsize=(9, 5))
    eixo.barh(correlacoes.index, correlacoes.values, color=cores)
    eixo.axvline(0, color="#222222", linewidth=0.8)
    eixo.set_xlabel("Correlação de Pearson com o rótulo de fraude")
    eixo.set_title("Atributos mais correlacionados com fraude")
    return _salvar(figura, destino)


def gerar_figuras(transacoes: pd.DataFrame, diretorio: Path) -> dict[str, Path]:
    """Gera as figuras da EDA e devolve os caminhos por nome lógico."""
    diretorio.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    return {
        "distribuicao_classes": _figura_distribuicao_classes(
            transacoes, diretorio / "eda_01_distribuicao_classes.png"
        ),
        "valor_por_classe": _figura_valor_por_classe(
            transacoes, diretorio / "eda_02_valor_por_classe.png"
        ),
        "faixas_valor": _figura_fraudes_por_faixa(
            transacoes, diretorio / "eda_03_fraudes_por_faixa_valor.png"
        ),
        "evolucao_temporal": _figura_evolucao_temporal(
            transacoes, diretorio / "eda_04_evolucao_temporal.png"
        ),
        "calor_temporal": _figura_calor_temporal(
            transacoes, diretorio / "eda_05_calor_temporal.png"
        ),
        "correlacao": _figura_correlacao(
            transacoes, diretorio / "eda_06_correlacao_alvo.png"
        ),
    }


def gerar_relatorio_eda(
    transacoes: pd.DataFrame, *, diretorio_figuras: Path, caminho: Path
) -> Path:
    """Gera as figuras e o relatório da EDA em Markdown."""
    figuras = gerar_figuras(transacoes, diretorio_figuras)

    resumo = resumo_geral(transacoes)
    distribuicao = distribuicao_classes(transacoes)
    faixas = fraudes_por_faixa_valor(transacoes)
    serie = evolucao_temporal(transacoes)
    correlacoes = correlacao_com_alvo(transacoes)
    com_tempo = adicionar_tempo_relativo(transacoes)

    fraudes = transacoes.loc[transacoes[COLUNA_ALVO] == 1]
    legitimas = transacoes.loc[transacoes[COLUNA_ALVO] == 0]
    ate_250 = float((fraudes[COLUNA_VALOR] <= 250).mean())
    ate_100 = float((fraudes[COLUNA_VALOR] <= 100).mean())

    linhas_faixas = "\n".join(
        f"| {linha.faixa_valor} | {formatar_inteiro(linha.transacoes)} | "
        f"{formatar_inteiro(linha.fraudes)} | "
        f"{formatar_percentual(linha.taxa_fraude)} | "
        f"{formatar_numero(linha.valor_movimentado)} |"
        for linha in faixas.itertuples()
    )
    linhas_correlacoes = "\n".join(
        f"| {atributo} | {formatar_numero(valor, 3)} |"
        for atributo, valor in correlacoes.head(8).items()
    )
    horas_com_fraude = serie.loc[serie["fraudes"] > 0]
    hora_pico = (
        int(
            horas_com_fraude.loc[horas_com_fraude["fraudes"].idxmax(), "hora_decorrida"]
        )
        if not horas_com_fraude.empty
        else 0
    )
    fraudes_por_dia = (
        com_tempo.loc[com_tempo[COLUNA_ALVO] == 1]
        .groupby("dia_relativo", observed=True)
        .size()
        .to_dict()
    )
    linhas_dias = "\n".join(
        f"| {dia} | {formatar_inteiro(quantidade)} |"
        for dia, quantidade in fraudes_por_dia.items()
    )
    participacao_legitima = formatar_percentual(
        float(distribuicao.loc[0, "participacao"])
    )
    participacao_fraude = formatar_percentual(
        float(distribuicao.loc[1, "participacao"])
    )
    mediana_fraudes = formatar_numero(float(fraudes[COLUNA_VALOR].median()))
    mediana_legitimas = formatar_numero(float(legitimas[COLUNA_VALOR].median()))
    media_fraudes = formatar_numero(float(fraudes[COLUNA_VALOR].mean()))
    media_legitimas = formatar_numero(float(legitimas[COLUNA_VALOR].mean()))
    zero_fraudes = formatar_inteiro(int((fraudes[COLUNA_VALOR] == 0).sum()))
    zero_legitimas = formatar_inteiro(int((legitimas[COLUNA_VALOR] == 0).sum()))
    legitimas_tabela = formatar_inteiro(distribuicao.loc[0, "transacoes"])
    fraudes_tabela = formatar_inteiro(distribuicao.loc[1, "transacoes"])

    conteudo = f"""# Relatório de análise exploratória

**Base:** MLG/ULB — *Credit Card Fraud Detection*

**Arquivo preparado:** `data/processed/transacoes.parquet`

**Recorte:** {formatar_inteiro(resumo["total_transacoes"])} transações,
sendo {formatar_inteiro(resumo["total_fraudes"])} fraudes
({formatar_percentual(resumo["taxa_fraude"])}).

> **Limitações da base:** dois dias de transações, tempo relativo, atributos
> anonimizados por PCA e moeda não fixada no dicionário. As análises de tempo
> usam blocos relativos; não há data absoluta nem hora do dia confirmada.

## 1. Estatísticas gerais

| Indicador | Valor |
|---|---|
| Transações | {formatar_inteiro(resumo["total_transacoes"])} |
| Legítimas | {formatar_inteiro(resumo["total_legitimas"])} |
| Fraudulentas | {formatar_inteiro(resumo["total_fraudes"])} |
| Taxa de fraude | {formatar_percentual(resumo["taxa_fraude"])} |
| Valor total (unidade original) | {formatar_numero(resumo["valor_total"])} |
| Valor em fraudes | {formatar_numero(resumo["valor_fraude"])} |
| Valor mediano | {formatar_numero(resumo["valor_mediano"])} |
| Ticket médio | {formatar_numero(resumo["ticket_medio"])} |

### Distribuição das classes

| Classe | Transações | Participação |
|---|---|---|
| Legítima | {legitimas_tabela} | {participacao_legitima} |
| Fraude | {fraudes_tabela} | {participacao_fraude} |

![Distribuição das classes](figures/{figuras["distribuicao_classes"].name})

## 2. OE-1 — Padrões e características das fraudes

| Indicador | Fraudulentas | Legítimas |
|---|---|---|
| Transações | {formatar_inteiro(len(fraudes))} | {formatar_inteiro(len(legitimas))} |
| Valor mediano | {mediana_fraudes} | {mediana_legitimas} |
| Valor médio | {media_fraudes} | {media_legitimas} |
| Transações com valor zero | {zero_fraudes} | {zero_legitimas} |

**Atributos mais correlacionados com o rótulo de fraude:**

| Atributo | Correlação |
|---|---|
{linhas_correlacoes}

![Valor por classe](figures/{figuras["valor_por_classe"].name})

![Correlações](figures/{figuras["correlacao"].name})

**Leitura:** os atributos são componentes PCA anonimizados. A correlação indica
associação estatística na base, não um fator de negócio nomeável (R-06).

## 3. OE-2 — Fraudes por faixa de valor

| Faixa | Transações | Fraudes | Taxa de fraude | Valor movimentado |
|---|---|---|---|---|
{linhas_faixas}

- {formatar_percentual(ate_250)} das fraudes têm valor até 250.
- {formatar_percentual(ate_100)} das fraudes têm valor até 100.

![Fraudes por faixa de valor](figures/{figuras["faixas_valor"].name})

## 4. OE-3 — Fraudes ao longo do tempo

A base cobre {formatar_numero(com_tempo["hora_decorrida"].max() + 1, 0)} blocos de
1 hora desde a primeira transação. O horário de pico de fraudes é o bloco
{formatar_inteiro(hora_pico)} (hora decorrida).

| Dia relativo | Fraudes |
|---|---|
{linhas_dias}

![Evolução temporal](figures/{figuras["evolucao_temporal"].name})

![Mapa de calor](figures/{figuras["calor_temporal"].name})

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

    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho
