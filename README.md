# Credit Card Fraud Detection

> **Projeto Integrador — Grupo 54 (Centro Universitário SENAC)**

Detection of fraud in credit card transactions using exploratory data analysis, supervised Machine Learning, and an interactive decision-support dashboard over the **MLG/ULB** dataset (*Credit Card Fraud Detection*).

---

## Overview

Credit card fraud is an asymmetric, high-impact problem with extreme class imbalance. This project implements an end-to-end data science lifecycle:
1. **Data Audit and Preparation:** Schema validation, duplicate audit, and parquet serialization.
2. **Exploratory Data Analysis (EDA):** Statistical summaries, value band distributions, and relative temporal patterns.
3. **Machine Learning Pipeline:** Stratified cross-validation comparing a linear baseline (Logistic Regression) against ensemble tree models (Random Forest, HistGradientBoosting), selecting models via **AUPRC** (Area Under the Precision-Recall Curve) and setting optimal decision thresholds.
4. **Interactive Dashboard:** Built with **Streamlit** and **Plotly**, providing risk analysts and managers with exploratory filters and offline model performance views (interface in Brazilian Portuguese).
5. **Deployment:** Production-ready configuration for **Streamlit Community Cloud** with on-demand dataset bootstrapping.

---

## Getting Started

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`/`venv`

### Setup

```bash
# Clone the repository
git clone https://github.com/MoisesHsilva1/projeto-integrador-dash.git
cd projeto-integrador-dash

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"
```

Place the dataset file in `data/raw/creditcard.csv` (see [data/README.md](data/README.md)). The raw and processed data directories are protected and not versioned in Git.

---

## Pipeline Execution

The CLI pipeline coordinates each stage of the analysis:

```bash
# E2/E3: Audit raw data, remove duplicates, and generate prepared Parquet
uv run credit-card-fraud-pipeline prepare

# E4: Generate exploratory analysis figures and Markdown report
uv run credit-card-fraud-pipeline eda

# E5–E7: Train models with cross-validation, evaluate holdout, and save artifacts
uv run credit-card-fraud-pipeline train

# Compare class imbalance strategies (class weights vs random resampling)
uv run credit-card-fraud-pipeline imbalance

# Run offline sensitivity, temporal validation, and permutation importance
uv run credit-card-fraud-pipeline experiments

# Run prepare, eda, and train sequentially
uv run credit-card-fraud-pipeline all
```

All pipeline steps use fixed seeds (`seed=42`) for full reproducibility.

---

## Interactive Dashboard

Start the local Streamlit dashboard:

```bash
uv run credit-card-fraud-dashboard
# Alternatively: uv run streamlit run src/credit_card_fraud/presentation/app.py
```

The dashboard provides five analytical views:
- **Visão geral:** Executive KPIs, class distribution donut chart, and fraud counts by transaction value band.
- **Análise de transações:** Breakdown of legitimate vs fraudulent volume and summary statistics.
- **Análise de fraudes:** Deep dive into fraudulent behaviors across value bands and relative time periods.
- **Explorador de transações:** Paginated, filterable tabular transaction view.
- **Desempenho do modelo:** Model comparison metrics, holdout confusion matrix, precision-recall curve, and cross-validation statistics.

### Hosted Deployment

The dashboard is configured for **Streamlit Community Cloud** using `streamlit_app.py` as the entrypoint. On cloud instances, the prepared dataset is provisioned on demand via release assets without committing data to Git.

---

## Architecture & Code Organization

The codebase follows a clean, layered architectural pattern:

```text
src/credit_card_fraud/
├── config.py              # Central settings, paths, and seeds
├── formatting.py          # Unified pt-BR number and currency formatting
├── data/                  # Data Access: loaders, schema validation, audit & cleaning
├── analytics/             # Domain layer: aggregations and metrics (single source of truth)
├── application/           # Application layer: dashboard filters and service orchestration
├── features/              # Feature matrix construction and stratified splitting
├── models/                # ML: model definitions, cross-validation, evaluation, persistence
├── presentation/          # Streamlit UI, Plotly charts, and theme tokens
├── deployment/            # Cloud runtime bootstrap and asset provisioning
├── reports/               # Markdown report generators and figure generation
└── cli.py                 # Pipeline command-line entrypoint

tests/                     # Comprehensive pytest test suite (84 tests)
reports/                   # Generated evaluation reports, figures, and model artifacts
data/                      # Local data directory (raw, interim, processed - gitignored)
```

---

## Quality & Automated Tests

The test suite validates data integrity, analytics calculations, filter combinations, model evaluation logic, and CLI commands:

```bash
# Run the complete test suite (84 tests)
uv run pytest

# Check code style and linting rules
uv run ruff check .

# Check formatting
uv run ruff format --check .
```

---

## Generated Artifacts

| Artifact | Path | Description |
|---|---|---|
| **Prepared base** | `data/processed/transactions.parquet` | Cleaned transactions (duplicates removed, Parquet format) |
| **Quality report** | `reports/RELATORIO_QUALIDADE.md` | Data audit dimensions, checksums, and duplicate removal log |
| **EDA report** | `reports/RELATORIO_EDA.md` | Statistical analysis, value bands, and temporal patterns |
| **EDA figures** | `reports/figures/eda_*.png` | High-resolution charts of class distribution, values, heatmaps |
| **Model metrics** | `reports/model/metrics.json` | Cross-validation scores, holdout metrics, and thresholds |
| **Model binary** | `reports/model/model_final.joblib` | Serialized scikit-learn pipeline (gitignored) |
| **Evaluation report** | `reports/RELATORIO_MODELO.md` | Model comparison, confusion matrix, PR curve, and importances |

---

## Dataset Characteristics & Caveats

- **Time Horizon:** Two consecutive days of European credit card transactions (September 2013).
- **Time Representation:** Relative elapsed seconds from the first transaction (`Time`). No absolute timestamps or clock hours are present.
- **PCA Attributes (`V1`–`V28`):** Anonymized principal components. Feature importances reflect statistical contribution within this dataset, not named business attributes.
- **Currency:** Values are recorded in the dataset's original unit without a declared currency code; no conversion is applied.
- **Class Imbalance:** 473 fraud transactions out of 283,726 (0.1667%) after removing 1,081 exact duplicates (which included 19 duplicate fraud records).
- **Scope:** Academic decision-support and pattern analysis tool; not intended for real-time automatic payment blocking.
