# Detecção de Fraudes em Cartão de Crédito

> **Projeto Integrador — Grupo 54 (Centro Universitário SENAC)**

> [English](README.md) | **Português (pt-BR)**

Detecção de fraudes em transações de cartão de crédito: preparação dos dados, análise exploratória, Machine Learning e um dashboard interativo. Base: **MLG/ULB** — *Credit Card Fraud Detection*.

## Requisitos

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)

## Instalação

```bash
uv venv
uv pip install -e ".[dev]"
```

Coloque a base em `data/raw/creditcard.csv` — veja [data/README.md](data/README.md). Os arquivos de dados não são versionados no Git.

## Comandos

```bash
# Pipeline
uv run credit-card-fraud-pipeline prepare   # audita a base, remove duplicatas, gera Parquet
uv run credit-card-fraud-pipeline eda       # figuras e relatório de EDA
uv run credit-card-fraud-pipeline train     # treina, avalia e salva os artefatos
uv run credit-card-fraud-pipeline all       # prepare + eda + train

# Dashboard
uv run credit-card-fraud-dashboard
```

Outros subcomandos: `imbalance` (compara estratégias de desbalanceamento) e `experiments` (análises de robustez e sensibilidade). Todas as etapas usam `seed=42`.

## Dashboard

Aplicação em Streamlit, com interface em português brasileiro e cinco abas:

| Aba | Conteúdo |
|---|---|
| **Visão geral** | KPIs, rosca de distribuição das classes, fraudes por faixa de valor, evolução temporal e mapa de calor |
| **Análise de transações** | Fraudes por faixa de valor e a tabela detalhada |
| **Análise de fraudes** | Atributos mais correlacionados com o rótulo de fraude |
| **Explorador de transações** | Tabela paginada do recorte atual |
| **Desempenho do modelo** | Métricas de holdout, matriz de confusão e comparação por validação cruzada |

Os filtros (faixa de valor, tempo decorrido, fraude/legítima) ficam na barra lateral. As métricas do modelo são estáticas e não mudam com eles.

## Estrutura do projeto

```text
src/credit_card_fraud/
├── data/          # leitores, validação de schema, auditoria e limpeza
├── analytics/     # métricas e agregações (fonte única de verdade)
├── application/   # filtros e orquestração do serviço
├── features/      # matriz de atributos e split estratificado
├── models/        # treino, avaliação e persistência
├── presentation/  # interface Streamlit, gráficos Plotly, tema
├── deployment/    # bootstrap de runtime para a nuvem
├── reports/       # geradores de relatório Markdown e figuras
├── config.py      # caminhos, colunas, seed
├── formatting.py  # formatação numérica pt-BR
└── cli.py         # entrada do pipeline

tests/             # suíte de testes pytest
reports/           # relatórios, figuras e artefatos do modelo
data/              # raw, interim, processed (gitignored)
```

## Testes

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Saídas

| Arquivo | Descrição |
|---|---|
| `reports/RELATORIO_QUALIDADE.md` | Auditoria dos dados e registro da remoção de duplicatas |
| `reports/RELATORIO_EDA.md` | Relatório da análise exploratória |
| `reports/RELATORIO_MODELO.md` | Comparação e avaliação dos modelos |
| `reports/figures/eda_*.png` | Figuras da EDA |
| `reports/model/metrics.json` | Métricas de validação cruzada e holdout |

## Limitações

- Dois dias de transações europeias (setembro de 2013); `Time` é relativo — não há datas absolutas nem horas de relógio.
- `V1`–`V28` são componentes PCA anonimizados, não atributos de negócio nomeados.
- Sem moeda declarada: os valores ficam na unidade original, sem símbolo nem conversão.
- Desbalanceamento extremo: 473 fraudes em 283.726 transações (0,17%).
- Ferramenta acadêmica de apoio à decisão — não é um sistema de bloqueio em tempo real.
