# Engineering Backlog — Credit Card Fraud Detection

**Grupo 54 — Centro Universitário SENAC**

Executable backlog to simplify the architecture, strengthen the Machine
Learning pipeline, and integrate the results into the dashboard. Each task is
small, verifiable, and has explicit acceptance criteria.

- **Last updated:** 19/11/2026.
- **Base adopted:** MLG/ULB (`data/raw/creditcard.csv`) — D-10
- **Reference decisions:** [`../DECISIONS.md`](../DECISIONS.md) (D-10 to D-29)
- **Language rule:** documentation and code in English; dashboard interface in
  pt-BR (D-26).

---

## Context and verified baseline (Phase 00)

### Measured baseline on 19/11/2026

| Item | Measured value |
|---|---|
| Raw transactions | 284,807 |
| Columns | 31 |
| Missing values | 0 |
| Exact duplicates | 1,081 (19 frauds) |
| Raw frauds | 492 (0.1727%) |
| Base after cleaning | 283,726 transactions, 473 frauds (0.1667%) |
| `Time` | 0 to 172,792 s (~48 h) |
| `Amount` | total 25,162,590.01; min 0; max 25,691.16; 1,825 zeros (27 frauds) |
| Selected model | Random Forest (largest mean AUPRC in CV) |
| Holdout | AUPRC 0.7992 · ROC-AUC 0.9592 · precision 0.8372 · recall 0.7579 · F1 0.7956 |
| Decision threshold | 0.4447 (maximizing F1 on out-of-fold training probabilities) |
| Confusion matrix | TN 56,637 · FP 14 · FN 23 · TP 72 |
| Tests | 54 passing (pytest); Ruff without findings |

### Module inventory

| Layer | Modules | Responsibility |
|---|---|---|
| Data Access | `data/loader.py`, `data/schema.py`, `data/auditoria.py` | loading, schema validation, cache, audit/cleaning |
| Analytics/Domain | `analytics/resumo.py`, `valor.py`, `temporal.py`, `correlacao.py` | metrics and aggregations (single source of truth) |
| Application | `application/filtros.py`, `application/servico.py` | filters, orchestration of dashboard services |
| Features/Models | `features/dataset.py`, `models/treino.py`, `avaliacao.py`, `desbalanceamento.py` | split, training, threshold, evaluation, imbalance experiments |
| Presentation | `presentation/app.py`, `componentes.py`, `estado.py`, `temas.py` | Streamlit UI (pt-BR), Plotly graphs, session state |
| Support | `config.py`, `formatacao.py`, `relatorios/*`, `cli.py` | configuration, pt-BR formatting, reports, entrypoints |

### Test coverage (54)

`test_analytics` (13), `test_app` (2), `test_auditoria` (3), `test_desbalanceamento` (5),
`test_features` (3), `test_filtros` (4), `test_formatacao` (4), `test_loader` (7),
`test_main` (2), `test_modelos` (4), `test_relatorios` (2), `test_schema` (6),
`test_servico` (8).

---

## Consolidated status

| Phase | Status |
|---|---|
| 00 Discovery and baseline | **Done** |
| 01 Architecture simplification | **Done** |
| 02 Data engineering | **Done** |
| 03 ML baseline | **Done** |
| 04 Algorithm improvements | **Done** |
| 05 Integration | **Done** |
| 06 Tests and code quality | **Done** |
| 07 Documentation | **Done** |
| 08 Verification | **Done** |

---

# Phase 00 — Discovery and baseline

### ARCH-001 — Repository audit and baseline establishment
**Priority:** P0 · **Complexity:** M · **Dependencies:** none · **Status:** done

Map modules, data flow, base profile, ML pipeline, dashboard, tests, and configuration.

**Evidence:** the "Measured baseline" and "Module inventory" tables above; the `all` pipeline reproduced AUPRC 0.7992 on the first run after the audit.

**Acceptance criteria:** complete inventory; recorded metrics; tests passing before changes. **Met.**

---

### ARCH-002 — Dataset profile
**Priority:** P0 · **Complexity:** S · **Dependencies:** ARCH-001 · **Status:** done

Document distribution, duplicates, value bands, and quality.

**Evidence:** all values recalculated from `data/raw/creditcard.csv` and verified against `docs/DATA_DICTIONARY.md` and `reports/quality.json`. **Met.**

---

### ARCH-003 — Documentation of the existing ML pipeline
**Priority:** P0 · **Complexity:** M · **Dependencies:** ARCH-001 · **Status:** done

Algorithms (LR baseline, RF, HistGradientBoosting), preprocessing (`StandardScaler` only for the linear step; `class_weight="balanced"`), stratified 80/20 split, seed 42, 3-fold CV, AUPRC as the main metric, threshold by F1 out-of-fold.

**Evidence:** `reports/model/metrics.json` and the regenerated report. **Met.**

---

### ARCH-004 — Dashboard feature inventory
**Priority:** P0 · **Complexity:** M · **Dependencies:** ARCH-001 · **Status:** done

Four mandatory visualizations (general indicators, distribution, value bands, temporal evolution) + heatmap, correlations, paginated explorer, and offline model indicators (D-29); value, elapsed-time, and classification filters; Plotly graphs (D-27); pt-BR interface (D-26).

**Evidence:** `presentation/app.py`; the smoke test `test_app.py` validates labels and filters. **Met.**

---

### ARCH-005 — Test coverage assessment
**Priority:** P1 · **Complexity:** S · **Dependencies:** ARCH-001 · **Status:** done

**Finding:** there were no tests for `data/loader.py` and for the imbalance strategies. Fixed in TST-012 and ML-012.

# Phase 01 — Architecture simplification

### ARCH-011 — Centralize configuration
**Priority:** P1 · **Complexity:** S · **Status:** done

All paths, seeds, and constants live in `config.py`; no module defines its own paths. Added `IMBALANCE_PATH` when the experiment required it. **Met.**

### ARCH-012 — Separate training from inference
**Priority:** P1 · **Complexity:** M · **Status:** done

Training in `models/treino.py`; persistence with `joblib` (`save_model`/`load_model`); inference/loading isolated in `data/loader.py` (`load_metrics`). The dashboard never trains. **Met.**

### ARCH-013 — Separate analytical logic from presentation
**Priority:** P1 · **Complexity:** M · **Status:** done

`presentation/` consumes only `application.DashboardService`; aggregations come from `analytics/`. No analytical rule in presentation. **Met.**

### ARCH-014 — Remove duplicated responsibilities
**Priority:** P1 · **Complexity:** S · **Status:** done

**Finding and fix:** pt-BR formatting was reimplemented inline in `presentation/estado.py` and `presentation/componentes.py`; switched to the single source `formatacao.py` (DRY). `relatorios/texto.py` already re-exported `formatacao`. **Met.**

### ARCH-015 — Standardize entry points
**Priority:** P2 · **Complexity:** S · **Status:** done

`credit-card-fraud`, `credit-card-fraud-pipeline {prepare,eda,train,all,imbalance,experiments,dashboard}`, and `credit-card-fraud-dashboard` work. **Met.**

# Phase 02 — Data engineering

### DAT-011 — Validate schema consistency
**Priority:** P1 · **Complexity:** S · **Status:** done

`data/schema.py` validates mandatory columns, numeric type, non-negativity, and target values; covered by `test_schema.py` (6 tests). **Met.**

### DAT-012 — Document duplicate policy
**Priority:** P1 · **Complexity:** S · **Status:** done

Exact-duplicate removal with recorded impact (D-11); confirmed against the baseline. **Met.**

### DAT-013 — Guarantee integrity of the train/test split
**Priority:** P1 · **Complexity:** S · **Status:** done

`split_train_test` is called before any fitting; CV within training; holdout isolated. No scaler fitted outside the pipeline. **Met.**

### DAT-014 — Treatment of missing values
**Priority:** P2 · **Complexity:** S · **Status:** done (documentary)

MLG/ULB has no nulls (verified: 0). `audit()` quantifies `missing_values`, and `validate_schema` guarantees numeric types before transformations. For future bases, absence is detected in the audit and reported in the Quality Report.

### DAT-015 — Document attribute and transformation types
**Priority:** P2 · **Complexity:** S · **Status:** done

`Time`, `V1`–`V28`, `Amount`, `Class`; derived `elapsed_hour`, `relative_day`, `relative_day_hour`, `value_band` documented in `DATA_DICTIONARY.md`. **Met.**

# Phase 03 — ML baseline

### ML-011 — Reproduce the baseline
**Priority:** P1 · **Complexity:** M · **Status:** done

Running the `all` pipeline reproduced AUPRC 0.7992 / F1 0.7956 / recall 0.7579 / precision 0.8372. **Met.**

### ML-012 — Evaluate imbalance strategies
**Priority:** P2 · **Complexity:** M · **Status:** done

Implemented `models/imbalance.py` + `imbalance` subcommand. Compares class weights (current), no weights, undersampling, and oversampling with the same stratified cross-validation; resamples only training folds. SMOTE not adopted (would require `imbalanced-learn`; D-19 preferred class weights for being explainable). Recorded in `reports/model/imbalance.json`.

### ML-013 — Optimize decision threshold
**Priority:** P2 · **Complexity:** S · **Status:** done

`choose_threshold` maximizes F1 on the out-of-fold training probabilities; holdout never participates. Measured threshold: 0.4447. **Met.**

### ML-014 — Document the model comparison
**Priority:** P2 · **Complexity:** S · **Status:** done

Per-model CV recorded in `metrics.json`: LR AUPRC 0.7543 ± 0.0220; RF 0.8061 ± 0.0168; GB 0.7204 ± 0.0236. Selection by largest mean AUPRC (RF). **Met.**

### ML-015 — Verify dashboard metric consistency
**Priority:** P1 · **Complexity:** S · **Status:** done

The dashboard reads `metrics.json` via `DashboardService.model_metrics()` and displays AUPRC/ROC-AUC/F1/precision/recall/threshold/confusion matrix/CV, marked static (D-29). No sketch numbers are used. **Met.**

# Phase 04 — Algorithm improvements

### ML-021 — Add an algorithm (only with justification)
**Priority:** P3 · **Complexity:** M · **Status:** closed without action

LR + RF + GB already compose a significant comparison; no experimental gain justifies XGBoost/LightGBM/Isolation Forest. **Decision:** keep the current set.

### ML-022 — Validate interpretation of attribute importance
**Priority:** P3 · **Complexity:** S · **Status:** done

`feature_importances` covers trees (`feature_importances_`) and linear models (`coef_`); the PCA limitation is documented in the reports and dashboard. **Met.**

### ML-023 — Guarantee seed reproducibility
**Priority:** P1 · **Complexity:** S · **Status:** done

`SEED = 42` in `config.py`; split, CV, models, and sampling use the seed. Two pipeline runs produced the same numbers. **Met.**

# Phase 05 — Integration

### INT-011 — Connect ML outputs to the dashboard
**Priority:** P1 · **Complexity:** M · **Status:** done

The dashboard consumes `model_metrics()` (Application) that reads the artifact (Data Access); presentation does not access models directly. **Met.**

### INT-012 — Separate offline training from interactive inference
**Priority:** P1 · **Complexity:** M · **Status:** done

`open_dashboard` starts only Streamlit; no training on load. Metrics come from pre-generated artifacts. **Met.**

# Phase 06 — Tests and code quality

### TST-011 — Expand the ML pipeline test coverage
**Priority:** P1 · **Complexity:** M · **Status:** done

`test_imbalance.py` (5 tests) includes an end-to-end comparison on synthetic data. **Met.**

### TST-012 — Data validation tests
**Priority:** P1 · **Complexity:** S · **Status:** done

Created `test_loader.py` (7 tests): CSV/Parquet load, schema validation, missing file errors, missing/present metrics. **Met.**

### TST-013 — Preprocessing and transformation tests
**Priority:** P2 · **Complexity:** S · **Status:** done

Existing coverage in `test_features.py` and `test_analytics.py`; no gaps identified. **Met.**

### TST-014 — Metric calculation tests
**Priority:** P2 · **Complexity:** S · **Status:** done

`test_classification_metrics_known` validates confusion matrix, precision, recall, F1, ROC-AUC, and AUPRC. **Met.**

### TST-015 — Dashboard integration tests
**Priority:** P3 · **Complexity:** M · **Status:** done

`test_app.py` validates rendering, labels, and status filter; `test_service.py` covers the services. **Met.**

### TST-016 — Enforce code quality with Ruff
**Priority:** P1 · **Complexity:** S · **Status:** done

`ruff check .` without findings; formatting verified. **Met.**

# Phase 07 — Documentation

### DOC-011 — Keep the backlog updated
**Priority:** P1 · **Complexity:** S · **Status:** done

This document was rewritten with the verified baseline and per-task status. **Met.**

### DOC-012 — Update the data dictionary
**Priority:** P2 · **Complexity:** S · **Status:** done

`DATA_DICTIONARY.md` verified against the verified baseline. **Met.**

### DOC-013 — Update DECISIONS.md
**Priority:** P1 · **Complexity:** S · **Status:** done

Decisions D-26 to D-29 already recorded. **Met.**

### DOC-014 — Update the README
**Priority:** P1 · **Complexity:** S · **Status:** done

Removed the outdated claim that the interface was in English; now references D-26 (pt-BR). **Met.**

# Phase 08 — Verification

| Verification | Command | Result |
|---|---|---|
| Full suite | `uv run pytest tests/ -q` | 54 passed |
| Lint | `uv run ruff check .` | All checks passed |
| Full pipeline | `uv run credit-card-fraud-pipeline all` | Reproduced AUPRC 0.7992 |
| Imbalance experiment | `uv run credit-card-fraud-pipeline imbalance` | See `reports/model/imbalance.json` |
| Dashboard | `uv run credit-card-fraud-dashboard` | AppTest without exceptions |

# Known limitations and future improvements

**Limitations**
- Two days of transactions; relative time, no absolute date/time.
- `V1`–`V28` PCA-anonymized: business factors not nameable.
- Currency not defined; values in original unit.
- Extreme imbalance (0.167% after cleaning); few frauds make metrics sensitive to split.
- 19 frauds removed with the duplicates (impact documented, D-11).
- Academic model; must not be used for automatic transaction blocking.

**Future improvements (outside the current scope)**
- Validate threshold sensitivity to other operational criteria (cost of FN/FP) without inventing monetary assumptions.
- Explore justified feature engineering from EDA (no deep learning/reinforcement).
- Evaluate stability across multiple seeds to quantify uncertainty.
- If approved (P-05), individual prediction reusing the trained pipeline.
