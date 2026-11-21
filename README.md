# Credit Card Fraud Detection

> **Projeto Integrador — Grupo 54 (Centro Universitário SENAC)**

> **English** | [Português (pt-BR)](README.pt-BR.md)

Fraud detection in credit card transactions: data preparation, exploratory analysis, Machine Learning, and an interactive dashboard. Base: **MLG/ULB** — *Credit Card Fraud Detection*.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

Place the base at `data/raw/creditcard.csv` — see [data/README.md](data/README.md). Data files are not versioned in Git.

## Commands

```bash
# Pipeline
uv run credit-card-fraud-pipeline prepare   # audit base, remove duplicates, write Parquet
uv run credit-card-fraud-pipeline eda       # figures and EDA report
uv run credit-card-fraud-pipeline train     # train, evaluate, save artifacts
uv run credit-card-fraud-pipeline all       # prepare + eda + train

# Dashboard
uv run credit-card-fraud-dashboard
```

Other subcommands: `imbalance` (compare imbalance strategies) and `experiments` (robustness and sensitivity analyses). All steps use `seed=42`.

## Dashboard

Streamlit app with the interface in Brazilian Portuguese, five tabs:

| Tab | Content |
|---|---|
| **Visão geral** | KPIs, class distribution donut, frauds by value band, temporal evolution, heatmap |
| **Análise de transações** | Frauds by value band plus the detailed table |
| **Análise de fraudes** | Attributes most correlated with the fraud label |
| **Explorador de transações** | Paginated table of the current filter slice |
| **Desempenho do modelo** | Holdout metrics, confusion matrix, cross-validation comparison |

Filters (value range, elapsed time, fraud/legitimate) are in the sidebar. Model metrics are static and do not change with them.

## Project structure

```text
src/credit_card_fraud/
├── data/          # loaders, schema validation, audit and cleaning
├── analytics/     # metrics and aggregations (single source of truth)
├── application/   # filters and service orchestration
├── features/      # feature matrix and stratified split
├── models/        # training, evaluation, persistence
├── presentation/  # Streamlit UI, Plotly charts, theme
├── deployment/    # cloud runtime bootstrap
├── reports/       # Markdown report and figure generators
├── config.py      # paths, columns, seed
├── formatting.py  # pt-BR number formatting
└── cli.py         # pipeline entrypoint

tests/             # pytest suite
reports/           # generated reports, figures, model artifacts
data/              # raw, interim, processed (gitignored)
```

## Tests

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Outputs

| File | Description |
|---|---|
| `reports/RELATORIO_QUALIDADE.md` | Data audit and duplicate removal log |
| `reports/RELATORIO_EDA.md` | Exploratory analysis report |
| `reports/RELATORIO_MODELO.md` | Model comparison and evaluation |
| `reports/figures/eda_*.png` | EDA figures |
| `reports/model/metrics.json` | Cross-validation and holdout metrics |

## Limitations

- Two days of European transactions (September 2013); `Time` is relative — no absolute dates or clock hours.
- `V1`–`V28` are anonymized PCA components, not named business attributes.
- No declared currency: values stay in the original unit, without symbol or conversion.
- Extreme imbalance: 473 frauds out of 283,726 transactions (0.17%).
- Academic decision-support tool — not a real-time blocking system.

