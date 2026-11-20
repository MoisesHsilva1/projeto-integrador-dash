# credit-card-fraud

Credit card fraud detection — Projeto Integrador do Grupo 54 (Centro Universitário SENAC). The project combines exploratory data analysis, supervised classification, and an interactive dashboard over the base **MLG/ULB** (*Credit Card Fraud Detection*).

## Getting started

Requirements: Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
cd ~/Documents/projects/credit-card-fraud
uv venv
uv pip install -e ".[dev]"
code .
```

Open the project with the **Python** profile in VS Code. The interpreter
is `.venv/bin/python` and the pytest tests are already configured in
`.vscode/settings.json`.

Place the base in `data/raw/creditcard.csv` (the file is not versioned). See
[`data/README.md`](data/README.md) and the dictionary in
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Pipeline

The flow follows the plan steps (`docs/PROJECT_PLAN.md`) and Aula 2:

```bash
uv run credit-card-fraud-pipeline prepare    # E2/E3: audit, clean, write Parquet
uv run credit-card-fraud-pipeline eda        # E4: figures and the EDA report
uv run credit-card-fraud-pipeline train      # E5–E7: models, evaluation, artifacts
uv run credit-card-fraud-pipeline all        # run the three steps in sequence
```

Each command is reproducible and uses seed `42` (`src/credit_card_fraud/config.py`).
Training compares Logistic Regression (baseline), Random Forest and Gradient
Boosting, selects by the mean AUPRC in cross-validation, and evaluates the
holdout with the decision threshold chosen at training time.

## Dashboard

```bash
uv run credit-card-fraud-dashboard
# equivalent: uv run streamlit run src/credit_card_fraud/presentation/app.py
```

The dashboard opens in the browser with the four mandatory visualizations —
general indicators, fraud distribution, frauds by value band, and temporal
evolution — plus the heatmap, correlations, the transaction explorer, and the
model's offline indicators. Filters are value, elapsed time, and
classification (fraud/legitimate). The interface is in Brazilian Portuguese
(pt-BR), per decision D-26.

> The dashboard needs the prepared base. If it does not exist, run
> `uv run credit-card-fraud-pipeline prepare` first.

## Generated artifacts

| Artifact | Path |
|---|---|
| Prepared base (not versioned) | `data/processed/transactions.parquet` |
| Quality report | `reports/RELATORIO_QUALIDADE.md` and `reports/quality.json` |
| EDA report | `reports/RELATORIO_EDA.md` and `reports/figures/eda_*.png` |
| Model metrics | `reports/model/metrics.json` |
| Trained model (not versioned) | `reports/model/model_final.joblib` |
| Evaluation report | `reports/RELATORIO_MODELO.md` and `reports/figures/modelo_*.png` |

## Tests and quality

```bash
uv run pytest              # full suite
uv run pytest tests/test_main.py::test_main
uv run ruff check --fix .  # lint + autofix
uv run ruff format .       # formatting
```

The dashboard smoke test is skipped automatically if the prepared base does
not yet exist.

## Structure

```text
src/credit_card_fraud/
├── config.py              paths, seeds, and constants
├── data/                  Data Access: loading, schema validation, cache, audit
├── analytics/             metrics, aggregations, and analytical rules
├── application/           dashboard filters and services
├── features/              attributes and train/test split (offline track)
├── models/                training, threshold, evaluation, persistence (offline track)
├── presentation/          Streamlit dashboard and chart components
├── reports/               report and figure generation
└── cli.py                 pipeline command line
tests/                     pytest suite
data/                      bases (not versioned)
docs/                      plan, architecture, dictionary, decisions
reports/                   figures, metrics, and generated reports
```

The layer dependency rules are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Documentation, identifiers, and code comments are in English; the dashboard
interface is in Brazilian Portuguese (pt-BR), per decision D-26.

## Base limitations

- Two days of European card transactions (September 2013), with relative time.
- `V1`–`V28` anonymized by PCA: business factors cannot be named.
- Currency not fixed in the dictionary; values are shown in the original unit.
- Extreme imbalance: 473 frauds out of 283,726 transactions after removing
  exact duplicates.
- The project is an academic decision-support tool, not a production fraud
  prevention system with automatic blocking.

## Documentation

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — full plan, requirements, and open items.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layered flow.
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — provenance, columns, treatment.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — decisions made during implementation.
- [`AGENTS.md`](AGENTS.md) — summarized operational context.
