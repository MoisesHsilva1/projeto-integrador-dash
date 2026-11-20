# Decision Record

**Grupo 54 — Centro Universitário SENAC**

This document complements Section 10 of [`PROJECT_PLAN.md`](PROJECT_PLAN.md).
It records the decisions made for the implementation of the pipeline and the
dashboard, with the justification and impact of each. Decisions **D-01 to
D-09** are in the plan; the decisions below (D-10 onward) were made for this
implementation.

## Database

| ID | Decision | Justification | Impact |
|---|---|---|---|
| **D-10** | Adopt the **MLG/ULB** base (`data/raw/creditcard.csv`) as the single base of the project. | It is the base cited in the presentation, compatible with the sketch and real anonymized data. The file was provided by the group in `data/raw/`. Resolves P-01/P-02. | The whole pipeline uses `creditcard.csv`; the synthetic base and *Give Me Some Credit* are not used. |
| **D-11** | Remove **exact duplicates**, keeping the first occurrence, before modeling. | Duplicates between train and test inflate metrics; keeping the same transaction twice creates leakage. The impact was measured and documented. | 1,081 rows removed, including 19 frauds: the base goes from 284,807/492 to 283,726/473 (0.17%). |
| **D-12** | Do not remove outliers or `Amount = 0` transactions. | Frauds can be at the extremes; zero is a valid value in the base. Cleaning must not delete signal. | Outliers and zero values remain; EDA describes them. |

## Time and currency

| ID | Decision | Justification | Impact |
|---|---|---|---|
| **D-13** | Analyze time in **relative blocks**: elapsed hour, relative day (1 and 2), and hour within the 24-hour block. | `Time` is relative to the first transaction; the origin is not documented as midnight. Resolves P-09. | Graphs and heatmap use "relative hours/days"; no claim about real-time fraud or seasonality. |
| **D-14** | Display values in the **original base unit**, without currency symbol and without conversion. | The dictionary fixes neither "euros" nor "R$"; converting without a source would be inventing data. Resolves P-10. | Labels "Amount (original unit)"; no graphs with "R$". |

## Product and interface

| ID | Decision | Justification | Impact |
|---|---|---|---|
| **D-15** | Implement the dashboard in **Streamlit**. | Lower complexity for an academic dashboard in Python (R-07). Resolves P-04. | `streamlit` dependency; run with `uv run credit-card-fraud-dashboard`. |
| **D-16** | Remove the "transaction type" filter. | The field does not exist in MLG/ULB; the interface reports the absence. Resolves P-08/R-09. | Available filters: value, elapsed time, and status (fraud/legitimate). |
| **D-17** | OLD: interface in English; documentation, code, reports, and identifier names in Portuguese. | D-09, kept. | KPIs, filters, graphs, and empty states in English; reports in Portuguese. |
| **D-18** | The individual **prediction tab** is **not part of the MVP**. | Controls scope (R-08); P-05 remains open. | The dashboard is analytical; the model is trained, evaluated, and saved, but there is no single-transaction simulation in the interface. |
| **D-26** | Dashboard interface in **Brazilian Portuguese (pt-BR)**, revoking D-09/D-17. | Explicit refinement requirement for the dashboard; better usability for the academic audience. | All KPIs, filters, graphs, tables, states, and messages in pt-BR; documentation, identifiers, and code comments remain in English (no change). |
| **D-27** | Use **Plotly** for dashboard graphs. | Interactive tooltips and consistent dark theme require interactive graphs; static matplotlib does not meet this. | New `plotly` dependency; components in `presentation/components.py` return Plotly figures. |
| **D-28** | pt-BR number formatting (`1.250,00`, `1,25%`), **no** currency symbol, **no** calendar dates. | Reinforces D-13/D-14: the base defines no currency or absolute date; only formatting follows the Brazilian pattern. | Single source in `formatting.py`; labels keep "original unit" and "relative hours/days". |
| **D-29** | Inclusion of a paginated **transaction explorer** and **offline model indicators**. | Leverages the existing sample/subset; no new modeling. | Filterable table with the available columns (no ID/category); the model metrics section is marked static. |

## Modeling and evaluation

| ID | Decision | Justification | Impact |
|---|---|---|---|
| **D-19** | Compare **Logistic Regression (baseline)**, **Random Forest**, and **Gradient Boosting (histograms)**, with `class_weight="balanced"` and no SMOTE. | Baseline + candidates, per R-06; class weights avoid extra dependency and are explainable. | Selection uses mean AUPRC in stratified 3-fold cross-validation in training. |
| **D-20** | Choose the **decision threshold** that maximizes F1 on out-of-fold training probabilities, and only then evaluate on the holdout. | Avoids choosing the threshold by looking at the test set; documents the precision/recall trade-off. | Threshold recorded in `reports/model/metrics.json` and used in the report. |
| **D-21** | Report **AUPRC** as the main metric, alongside ROC-AUC, precision, recall, F1, accuracy (context), and confusion matrix. | Official base recommendation (R-05) and appropriate for extreme imbalance. | No arbitrary numerical goals; the result is whatever the pipeline produced. |
| **D-22** | Split train/test (80/20, stratified, seed 42) **before** any fitting; standardize only inside the pipeline. | Avoids leakage (R-04). | `StandardScaler` fitted only on training; trees do not require scaling, but the pipeline is uniform. |

## Technical organization

| ID | Decision | Justification | Impact |
|---|---|---|---|
| **D-23** | Prepared base in **Parquet** (`data/processed/transactions.parquet`), not versioned. | Faster reading than CSV and keeps data out of Git. | The dashboard reads only Parquet via Data Access, with cache. |
| **D-24** | Follow the layer flow in `docs/ARCHITECTURE.md`. | Single source of metrics and separation of responsibilities (RNF-08/RNF-09). | `data/` → `analytics/` → `application/` → `presentation/`; ML in `features/` and `models/`. |
| **D-25** | Save figures and reports in `reports/` and the model binary in `reports/model/`. | Academic traceability; the binary is ignored by Git as an artifact. | `reports/model/model_final.joblib` is not versioned; metrics and figures are. |

## Results obtained in this implementation

The pipeline ran on 18/11/2026 with seed 42 and prepared base
(283,726 transactions, 473 frauds):

- selected model: **Random Forest** (largest mean AUPRC in cross-validation);
- AUPRC on holdout: **0.7992**;
- F1, recall, and precision on holdout: **0.7956 / 0.7579 / 0.8372**;
- confusion matrix: 56,637 true negatives, 14 false positives,
  23 false negatives, and 72 true positives.

These values are pipeline results on this base, not goals or sketch numbers.
They may vary if the base, seed, or library versions change.
