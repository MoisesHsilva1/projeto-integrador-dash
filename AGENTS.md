# AGENTS.md

Operational context of **Projeto Integrador do Grupo 54 — Centro Universitário SENAC**.

> **Read first:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). This is the detailed technical and functional plan, with requirements, base analysis, steps, risks, and pending items. This file summarizes what an agent needs to know to work without rebuilding context.

---

## Current status

- **Phase:** implementation of the MVP (E2–E8) over MLG/ULB. Analysis, documentation, and the pipeline exist in the repository; the academic pending items (rubric, deadlines, delivery format, AI policy) remain open.
- **Last update:** 18/11/2026.
- **Existing code:** complete pipeline in `src/credit_card_fraud/` (Data Access, Analytics, Application, Features/Models, and Presentation), reports in `reports/`, and a Streamlit dashboard. The model was evaluated with AUPRC 0.7992 on holdout; these numbers come from the pipeline, not from the sketch.
- **Implementation decisions:** recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md) (D-10 to D-25). The MLG/ULB base was adopted, exact duplicates removed, time treated in relative blocks, and currency kept in the original unit.
- **Rule:** if a new decision is not recorded in the materials, in the plan, or in `docs/DECISIONS.md`, treat it as **pending**, not as a requirement.

---

## Project context

### Theme

Detection of **fraud in credit card transactions made with credit cards**, using data analysis and Machine Learning to identify patterns and suspicious behaviors.

### Problem

Identify which transactions can be fraudulent, distinguishing them from legitimate ones, and searching for patterns that indicate possible fraudulent operations.

### General objective

Create an **interactive dashboard** to analyze credit card transactions, understand their main characteristics and behaviors related to fraud, and support a predictive and efficient analysis.

### Specific objectives

1. Identify patterns and characteristics of fraudulent transactions.
2. Analyze the occurrence of frauds according to the **value** of transactions.
3. Analyze the occurrence of frauds over **time** (time slots, periods, heatmap).

### Minimum identified deliverables

- data treatment and preparation;
- exploratory analysis;
- interactive dashboard with at least **four visualizations**: general indicators, fraud distribution, frauds by value band, and temporal evolution;
- Machine Learning application to predict possible frauds;
- model evaluation and identification of associated factors;
- academic report/presentation.

### Business context

Global losses to card fraud reached **US$ 33.41 billion in 2024** (Nilson Report, edition 1298). The dashboard is a **decision-support tool**, not a production fraud-prevention system with automatic blocking.

### Scope

**In scope:** historical transaction base, EDA, supervised classification, evaluation, dashboard, documentation.

**Out of scope (presumed; confirm):** integration with payment systems, real-time scoring, automatic blocking, deployment in production, identifiable personal data, deep learning/reinforcement without justification.

### Target audience

**Not defined in the materials.** Do not assume the audience is end customers. The working hypothesis is risk/fraud analysis teams and managers, but this must be confirmed by the group.

### Architecture

The **target architecture** — still in planning, no code yet — organizes the service path in five blocks: prepared dataset, `Data Access`, `Analytics/Domain`, `Application`, and `Presentation`, culminating in the `Interactive Dashboard`. Dependencies go down only in this order; Presentation does not read CSVs or access Data Access directly, and Analytics/Domain is the single source of metrics, aggregations, and analytical rules for the dashboard. Model training/evaluation (E5–E7) is an offline trail: its artifacts are consumed by Analytics/Domain, without creating an "ML" layer on the service path. The base (P-01) and the tool (P-04) are still pending, so the design is agnostic to both. Full reference: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Academic references

| Source | Relevant content |
|---|---|
| `aula2.pdf` | Data science lifecycle: understand → collect → preprocess → EDA → model construction. Preprocessing: cleaning, sampling, imbalance treatment, transformation. EDA: descriptive statistics, histograms, boxplots, scatter plots, heatmaps, correlation. Models: supervised (classification/regression, decision trees), unsupervised, and reinforcement. |
| `Codigos_Aula2.zip` | `Aula2.ipynb` (pandas, cleaning, dates, `groupby`, `merge`, `explode`, matplotlib/seaborn) and `Extra_Aula2.ipynb` (`value_counts`, bar/pie charts). Use as a technical reference level, not as a template. |
| `grupo-54.pptx.pdf` | Primary source of the scope: problem, objectives, next steps, dashboard sketch, and references. |

**Concepts that must appear in the project:** lifecycle, data quality, imbalance, transformation, EDA, supervised classification, evaluation with adequate metrics, reproducibility.

**Concepts that must not be added without need:** reinforcement learning, deep learning, clustering/segmentation, real-time systems, microservices architecture.

---

## Data

### Decision situation

The **MLG/ULB** base was adopted as the single project base (D-10). The raw file is in `data/raw/creditcard.csv`. The 10K synthetic base and *Give Me Some Credit* are not used in the pipeline.

| Base | Status | Suitability |
|---|---|---|
| **MLG/ULB — Credit Card Fraud Detection** | **Adopted**; pipeline executed. | Main base, with documented caveats. |
| **Synthetic 10K — `miadul/credit-card-fraud-detection-dataset`** | Not used. | Discarded alternative in this implementation. |
| **Give Me Some Credit** | Not used. | Default risk base, unsuited to the fraud problem. |

**Do not mix bases.** The decision is recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md) (D-10).

### MLG/ULB — Verified facts

- 284,807 transactions; 31 columns: `Time`, `V1`–`V28`, `Amount`, `Class`.
- 492 frauds (**0.1727%**).
- Two days of European card transactions (September 2013).
- `Time` = seconds since the first transaction; **not absolute date/time**.
- `V1`–`V28` = anonymized PCA components; the business meaning was not disclosed.
- No missing values.
- 1,081 exact duplicates, including **19 frauds**; decide and document before removing.
- `Amount` total = 25,162,590.01; 1,825 transactions with zero value (27 frauds).
- Official recommendation: evaluate with **AUPRC**, not accuracy.

### Synthetic 10K — Verified facts

- 10,000 transactions; 10 columns: `transaction_id`, `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, `is_fraud`.
- 151 frauds (**1.51%**) in the file, although the base page declares 4–5%.
- No nulls and no exact duplicates at inspection.
- 24 hours, 5 merchant categories, interpretable attributes.
- **Synthetic** base (Apache 2.0, created in December 2025). Does not support claims about real fraud.

### Give Me Some Credit — why not use it

- 150,000 training records + 101,503 test records; target `SeriousDlqin2yrs` (serious default within two years).
- 10 credit variables; no transaction, purchase amount, merchant, time slot, or fraud label.
- It is a **credit default risk** problem, not fraud. No key to cross-reference with the other bases.

### Where to place the files

The `data/` area already exists and is protected by `.gitignore`. See `data/README.md`.

| Base | Expected path |
|---|---|
| MLG/ULB | `data/raw/creditcard.csv` |
| Synthetic 10K | `data/raw/credit_card_fraud_10k.csv` |
| Give Me Some Credit | `data/raw/cs-training.csv` (and `cs-test.csv`, if needed) |

Place only the chosen base. Do not edit the original file; generate treated copies in `data/interim/` or `data/processed/`.

### Mandatory care

- Do not version raw or processed data in Git. The `data/` area is already in `.gitignore`; keep it so.
- Record URL, access date, version, license, and checksum.
- Separate train/test **before** any transformation that learns parameters.
- Treat imbalance only in training.
- Do not automatically remove outliers: frauds can be at the extremes.
- Do not invent wall-clock time, absolute date, or currency. In MLG/ULB, `Time` is relative and currency is not fixed as “R$”.
- Do not use the sketch numbers as results.

---

## Decisions, recommendations, and pending items

### Already recorded decisions

- Theme, problem, and general objective (APRES).
- Three specific objectives (APRES), with a pending fix for the duplicated wording of objective 2.
- Dashboard with at least four visualizations (APRES).
- Machine Learning application (APRES).
- Kaggle fraud base (APRES) — **which base is still pending**.
- Lesson 2 lifecycle as the methodological guideline.
- Documentation and code in English; dashboard interface in Brazilian Portuguese (D-26).

### Recommendations

- Use **MLG/ULB** as the main base.
- Do not use **Give Me Some Credit**.
- Do not mix bases.
- If using the synthetic base, label it as synthetic everywhere.
- Adopt **AUPRC** as the main metric; report precision, recall, F1, ROC-AUC, and confusion matrix.
- Compare a baseline with at least one candidate model; Random Forest is not a final decision.
- Use **Streamlit** as the default dashboard candidate, unless there is a contrary requirement.
- Build the MVP with the four visualizations before increments.
- Remove or redefine the “transaction type” filter in MLG/ULB.
- Fix currency and temporal axis in the dashboard.
- Confirm rubric, deadlines, and delivery format with the professor.

### Critical pending items (do not implement without resolving)

1. **P-01/P-02:** MLG/ULB or synthetic? The provided link points to a synthetic notebook, not MLG/ULB. Confirm the intention.
2. **P-03:** what is the target audience?
3. **P-04:** which dashboard tool?
4. **P-05:** does the prediction tab belong to the MVP?
5. **P-06/P-07:** which models, decision threshold, and imbalance strategy?
6. **P-08:** will the “transaction type” filter be removed or the base changed?
7. **P-09:** how to treat time in MLG/ULB?
8. **P-10:** which currency to use in the graphs?
9. **P-11:** format, deadlines, and rubric.
10. **P-12/P-13:** group roles and data storage/versioning.
11. **P-14:** AI citation and use policy.
12. **P-15:** are the sketch numbers only illustrative?

See Section 10 of the plan for the forwarding of each item.

---

## Development guidelines

### Approach

Follow the Lesson 2 cycle and the plan sequence:

```text
E1 Scope → E2 Data → E3 Audit/cleaning → E4 EDA → E5 Features/split
→ E6 Modeling → E7 Evaluation → E8 Dashboard → E9 Integration/validation
→ E10 Academic deliverables
```

EDA can revert to cleaning; evaluation can revert to training; the dashboard can require new cuts to EDA.

### Expected organization

- `data/raw/` — raw base, immutable and not versioned; instructions in `data/README.md`.
- `data/interim/` — intermediate data, not versioned.
- `data/processed/` — treated base, not versioned.
- `notebooks/` — EDA and experiments.
- `src/credit_card_fraud/` — reusable code: data, features, models, visualization, and dashboard; proposed layer organization in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- `reports/figures/` — generated figures.
- `docs/` — plan, data dictionary, decisions.
- `tests/` — tests.

Do not create empty structures or unnecessary modules. The project is academic: simplicity and explainability are more valuable than production architecture.

### Good practices

- Keep code in Python ≥3.11, with `uv`.
- Use `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` when needed; do not add dependencies before the decision.
- Fixed seeds; reproducible pipeline; versions recorded.
- Separate train/test before adjusting transformations.
- Avoid duplicated logic between notebooks and the dashboard.
- Report metrics appropriate to imbalance; accuracy is not enough.
- Document every transformation and every line removed.
- Do not present synthetic data, simulations, or sketch values as real results.
- Keep documentation, identifiers, and code comments in English; the dashboard interface (KPIs, filters, graphs, tables, navigation, messages, and empty/error states) in Brazilian Portuguese (pt-BR), per D-26. Do not translate raw base values.

### Quality criteria

- **Data:** provenance, dictionary, quality, and impact of removals.
- **Model:** baseline, comparison, cross-validation, AUPRC, precision/recall/F1, confusion matrix, and justified threshold.
- **Dashboard:** four mandatory visualizations, feasible filters, consistent numbers, correct labels, documented local execution.
- **Academic:** traceability, sources, limitations, reproducibility, and rubric adherence.

### Documentation

- `docs/PROJECT_PLAN.md` is the roadmap; keep it up to date when decisions are made.
- Record new decisions in Section 10 of the plan.
- Create `docs/DATA_DICTIONARY.md` in the data step and a decision record if the group finds it useful.
- Update the README with execution instructions when the pipeline exists.

### Scope control

The implemented MVP includes the dashboard, model training, and the data pipeline.
Do **not** implement, without an identified and justified need:

- automated data download;
- analysis notebooks (reports are generated by code);
- deployment;
- individual prediction tab (D-18/P-05);
- production features.

Also do not add deep learning, reinforcement learning, microservices, or a database without an identified and justified need.

---

## Orientation for agents

1. **Read the plan before proposing significant changes.** Consult `docs/PROJECT_PLAN.md`, especially Sections 3, 4, 7, and 10.
2. **Respect academic requirements.** The Lesson 2 lifecycle and the group presentation are the source of scope.
3. **Do not invent requirements, data, metrics, deadlines, or results.** If information is not in the materials, record it as pending.
4. **Distinguish** explicit requirement, mockup intent, technical recommendation, and group decision.
5. **Justify important technical decisions** with the problem, the data, and the literature.
6. **Avoid unnecessary complexity.** Prioritize solutions the group can explain and defend academically.
7. **Do not use sketch numbers as results.** Appendix A of the plan shows the inconsistencies.
8. **Do not mix bases.** If the synthetic base is used, it must be clearly labeled.
9. **Consult `docs/DECISIONS.md`** before reopening decided decisions (D-10 to D-25); ask before implementing anything that depends on P-03, P-05, P-11, P-12, P-14, or P-15.
10. **Keep the documentation up to date** when decisions are made.
11. **Make small changes, with tests and lint**, preserving the entrypoint behavior (`main.py`) and scaffold.
12. **Do not use sketch numbers or other execution results** without recomputing and recording the seed and base.

---

## Commands and scaffold

- `uv run pytest` — full suite; single test: `uv run pytest tests/test_main.py::test_version`
- `uv run ruff check --fix .` — lint + autofix
- `uv run ruff format .` — formatting
- `uv run credit-card-fraud` — runs the entrypoint (`src/credit_card_fraud/main.py:main`)
- `uv run credit-card-fraud-pipeline prepare|eda|train|all` — data pipeline, EDA, and modeling (see also `imbalance` and `experiments`)
- `uv run credit-card-fraud-dashboard` — starts the Streamlit dashboard
- Initial setup: `uv venv && uv pip install -e ".[dev]"`

Always use `uv run` (keeps `.venv` synced with `pyproject.toml`). There is no typecheck, pre-commit, or CI configured.

### Gotchas

- Distribution name: `credit-card-fraud`; import package: `credit_card_fraud` (underscore, in `src/`). pytest adds `src` to `sys.path` via `[tool.pytest.ini_options].pythonpath`.
- The version is duplicated in `pyproject.toml` (`[project].version`) and `src/credit_card_fraud/__init__.py` (`__version__`). Update both; `tests/test_main.py::test_version` verifies the value.
- The exact output of `main()` is checked in `tests/test_main.py::test_main`; change the test together with the message.
- Ruff is the only quality gate: line 88, target `py311`, rules `E,F,I,UP,B,SIM`. VS Code formats/corrects on save via `.vscode/settings.json`.

---

## Summary in one sentence

Group 54 implemented a credit card fraud analysis and dashboard with Machine Learning over MLG/ULB, following Lesson 2's lifecycle; the MVP covers the four mandatory visualizations, the model evaluated by AUPRC, and reproducible documentation, and the remaining academic pending items are the rubric, deadlines, delivery format, and AI policy.
