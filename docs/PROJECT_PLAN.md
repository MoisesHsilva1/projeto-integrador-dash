# Projeto Integrador — Project Plan: Credit Card Fraud Detection

**Grupo 54 — Centro Universitário SENAC**

- **Members:** Alex Oliveira Veras, Jhonatan da Silva Ribeiro, Magno Henrique Souza Cardoso, Moises Henrique da Silva and Taynan Nunes de Araujo.
- **Theme:** detection of fraud in financial transactions made with credit cards.
- **Document status:** planning consolidated and implementation in progress. The MVP (E2–E8) was implemented over the base MLG/ULB; the implementation decisions are in [`DECISIONS.md`](DECISIONS.md) and the results come from the pipeline executed on 18/11/2026.
- **Last updated:** 18/11/2026.
- **Complementary document:** [`AGENTS.md`](../AGENTS.md), with the summarized operational context for agents.

> **Reading rule:** this plan explicitly separates what was **identified in the materials**, what was **decided by the group**, what is **technical suggestion**, and what is still **pending**. Do not treat suggestions as approved requirements. See Section 10 before taking implementation decisions.

---

## 1. Overview

### 1.1 Purpose

This document organizes the problem, requirements, available data bases, academic foundation, proposed architecture, and the development sequence of the Projeto Integrador do Grupo 54.

It must serve as:

- a single-scope reference during development;
- a roadmap tracking the progress of each step;
- a record of decisions taken and open questions;
- a base for the final report and presentation.

### 1.2 Executive summary

The group intends to build an **interactive dashboard** for analyzing credit card transactions and applying **Machine Learning** to identify patterns associated with fraud. The proposal is recorded in the group presentation and is compatible, in broad terms, with the data science lifecycle presented in Lesson 2.

The main technical question open is the **database**. The presentation cites the classic base *Credit Card Fraud Detection* (MLG/ULB), with 284,807 transactions and 492 frauds, and the dashboard sketch uses numbers compatible with it. However, the provided Kaggle link points to a notebook that uses a **synthetic** base, different from it, with 10,000 transactions and 151 frauds. This divergence needs to be resolved by the group before development.

**Preliminary recommendation:** adopt the classic MLG/ULB base as the main base, because it is the base cited in the presentation, the one backing the sketch numbers, and the one with the greatest credibility as a real anonymized database. The synthetic base should be treated as a documented alternative, not as a silent supplement. The *Give Me Some Credit* base is not adequate to the fraud problem and should be discarded for this scope.

### 1.3 Status legend

| Status | Meaning |
|---|---|
| **Explicit** | Contained directly in a provided material. |
| **Sketch** | Appears in the dashboard sketch, but not formalized as a requirement. |
| **Recommended** | Good practice or technical suggestion from this analysis; not an academic requirement. |
| **Pending** | Depends on a group, professor, or validation with the original source decision. |

### 1.4 Sources used

| Code | Source | What was used |
|---|---|---|
| **AULA2** | `aula2.pdf` — webconference “Introduction to Data Science”, 27 pages | Lifecycle, preprocessing, EDA, model building, learning types. |
| **COD** | `Codigos_Aula2.zip` — `Aula2.ipynb`, `Extra_Aula2.ipynb`, `books.csv`, `dados_populacionais_1000_linhas.csv` | Pandas techniques, cleaning, aggregation, EDA and visualization with matplotlib/seaborn. |
| **APRES** | `grupo-54.pptx.pdf` — Projeto Integrador do Grupo 54 presentation, 12 pages | Problem, objectives, scope, next steps, dashboard sketch, references. |
| **GMSC** | Kaggle — *Give Me Some Credit* | Structure, variables, business problem and limitations. |
| **CCFD-ULB** | Kaggle — *Credit Card Fraud Detection* (MLG/ULB) | Classic fraud base; structure, variables, distribution, evaluation recommendations. |
| **CCFD-10K** | Kaggle — *Credit Card Fraud Detection Dataset* (`miadul/...`), used by the notebook of the provided link | Synthetic base of 10,000 transactions; structure, variables and limitations. |
| **NILSON** | Nilson Report, edition 1298 / notice of January 2026 | Data on global card fraud losses in 2024 (US$ 33.41 billion). |

**Traceability observation:** the presentation cites the data point of US$ 33.41 billion without indicating the source. The reference was located in the Nilson Report and should be included in the final report. The provided Kaggle link for the “Base 2” does not point to the MLG/ULB base page; it points to a notebook that consumes the synthetic base `miadul/credit-card-fraud-detection-dataset`. This discrepancy is treated in Section 4.

**Pacing observation:** the Lesson 2 PDF ends with a slide titled “CALENDÁRIO”, but dates or readable content could not be extracted. No deadlines were assumed in this plan; the academic calendar needs to be confirmed with the professor.

---

## 2. Problem and objectives

### 2.1 Business context

The presentation sets the problem in the growth of global losses caused by card frauds. The cited data is **US$ 33.41 billion in 2024** (NILSON). The group argues that data analysis can:

- reduce losses by identifying potential frauds;
- make detection more efficient with Machine Learning support;
- reveal characteristics and behaviors associated with fraudulent transactions;
- support the analysis and decision-making through a dashboard.

The intended product is therefore an **analytical decision-support tool**, not a production fraud-prevention system with automatic transaction blocking. This distinction is important to control scope.

### 2.2 Problem

> Identify which credit card transactions could fit cases of fraud, distinguishing them from legitimate ones and seeking patterns that indicate possible fraudulent operations.

Source: APRES, p. 3.

In technical terms, the problem is a **binary supervised classification**: predict whether a transaction is fraudulent (`1`) or legitimate (`0`), from labeled historical attributes. The MLG/ULB base already has the label `Class`; the synthetic base has `is_fraud`.

### 2.3 General objective

Create an **interactive dashboard** that allows analyzing credit card transaction data, understanding its main characteristics and behaviors related to frauds, and supporting a predictive and efficient analysis, identifying patterns and characteristics that contribute to fraud detection.

Source: APRES, p. 4 (faithful paraphrase).

### 2.4 Specific objectives

| ID | Objective | What the presentation suggests | Status |
|---|---|---|---|
| **OE-1** | Identify patterns and characteristics of fraudulent transactions | Observe values, frequency, distribution, and behaviors out of the ordinary; use EDA with graphs and statistics. | Explicit |
| **OE-2** | Analyze the occurrence of frauds according to the **value** of transactions | Compare frauds by value band and distribution of values. The slide body repeats the OE-1 text; the title, sketch, and next steps confirm the intent to analyze value. | Explicit, with duplicated text in the presentation |
| **OE-3** | Analyze the occurrence of frauds over **time** | Time graphs, analysis by time slot, and heatmap to identify periods with more occurrences. | Explicit |

**Pending wording:** the OE-2 slide text in the presentation is duplicated with the OE-1 text. Before delivery, the group must rewrite it to clearly describe the analysis by value. This does not change the scope, but avoids inconsistency in the report.

### 2.5 Target audience

**Not defined explicitly in the materials.** The presentation talks about “supporting the data analysis” and “facilitating information visualization”, but does not name the dashboard user.

Working hypothesis, **to be confirmed**: risk analysis/fraud teams and managers who need to track indicators, investigate patterns, and support decisions. Do not treat this hypothesis as a requirement until the group validates it.

### 2.6 Scope

**In scope identified:**

- obtaining and preparing a credit card transaction base;
- exploratory data analysis;
- analysis of fraud patterns by value and over time;
- application of a supervised classification technique;
- model performance evaluation;
- interactive dashboard with at least four visualizations;
- documentation, report, and academic presentation.

**Presumed out of scope (confirm):**

- production fraud system;
- integration with acquirers, issuers, or payment gateways;
- automatic transaction blocking;
- real-time or streaming scoring;
- use of identifiable personal data;
- causal inference on fraud causes;
- combined indiscriminate use of different bases;
- deep learning, reinforcement learning, or techniques not justified by the problem.

### 2.7 Cross-analysis of materials

| Question | Finding | Evidence | Consequence for the plan |
|---|---|---|---|
| **1. What are the academic and technical requirements?** | The lesson presents the lifecycle (understand, collect, preprocess, EDA, model). The presentation defines the product and five next steps. No syllabus, rubric, deadlines, or delivery format are provided in the materials given. | AULA2, p. 3–16; APRES, p. 9–11 | The plan follows the lesson cycle and records the absence of formal requirements as pending. Do not invent deadlines or grading criteria. |
| **2. What lesson concepts should be applied?** | Lifecycle; cleaning; sampling; imbalance treatment; transformation; EDA; supervised learning (classification/regression); decision trees; distinction between supervised, unsupervised, and reinforcement learning. | AULA2, p. 3–16 | Apply the complete cycle. For fraud, use supervised classification. Do not use reinforcement learning or unsupervised learning without justification. |
| **3. What analysis techniques are suitable?** | Descriptive statistics, histograms, boxplots, scatter plots, heatmaps, correlation; cleaning and validation of business rules; aggregations; train/test split; classification performance metrics. | AULA2, p. 12–16; COD, `Aula2.ipynb` | The EDA should answer the three specific objectives. Modeling should compare a baseline with at least one candidate model. |
| **4. Which bases make sense?** | MLG/ULB is the base cited in the presentation and compatible with the sketch. The 10,000-transaction synthetic base is the one used by the provided link, and is more interpretable, but it is not the cited base. *Give Me Some Credit* is from credit default, not fraud. | APRES, p. 9 and references; CCFD-ULB; CCFD-10K; GMSC | Pending decision between MLG/ULB and the synthetic base. *Give Me Some Credit* should not be used for the current problem. |
| **5. What are the limitations and difficulties?** | Extreme imbalance; PCA-anonymized attributes in the real base; two-day window; absence of transaction type, absolute date, and card identifier in the real base; duplicates; currency change in the sketch; data leakage risk; few positives. | CCFD-ULB; sketch; local analysis | Treat imbalance and duplicates explicitly; do not promise analyses the base does not support; document time and currency assumptions. |
| **6. What information is missing to define?** | Exact base and version; target audience; dashboard tool; model and metrics; decision threshold; scope of the prediction tab; format and deadlines of the deliverables; group roles; whether to use the synthetic base. | APRES; material gaps | See Section 10. None of these implementation decisions should be silently assumed. |
| **7. What deliverables are necessary?** | Treated base; EDA; evaluated model; dashboard with four visualizations; results analysis; report/presentation. Academic format is not specified. | APRES, p. 9–11 | See Section 11. Code and documentation must allow reproducing the results. |
---

## 3. Requirements

### 3.1 Functional requirements

| ID | Requirement | Source | Status | Notes |
|---|---|---|---|---|
| **RF-01** | Select and obtain a credit card transaction base on Kaggle. | APRES, p. 9 | Explicit, ambiguous | The presentation cites the base; the provided link points to a different synthetic base. Resolve before implementation. |
| **RF-02** | Perform cleaning, organization, and preparation of the data, including the creation of indicators relevant to fraud analysis. | APRES, p. 9 | Explicit | Document each transformation and each line removed. |
| **RF-03** | Identify patterns and characteristics of fraudulent transactions through EDA. | APRES, p. 5 | Explicit | Use statistics and visualizations; compare frauds and non-frauds. |
| **RF-04** | Analyze the occurrence of frauds by value band. | APRES, p. 5 and 10 | Explicit | Correct the duplicated OE-2 text. Define bands and currency. |
| **RF-05** | Analyze the occurrence of frauds over time, including time slots and heatmap. | APRES, p. 6 | Explicit | On the MLG/ULB base, the temporal axis is relative and covers two days; see the limitation in Section 4. |
| **RF-06** | Develop an interactive dashboard with at least four visualizations: general indicators, fraud distribution, frauds by value band, and temporal evolution. | APRES, p. 10 | Explicit | The minimum product requirement. The sketch has four tabs, but “four visualizations” does not mean “four tabs”. |
| **RF-07** | Allow filters for period, value band, transaction type, and status (fraud/legitimate). | Sketch, p. 10 | Sketch | “Transaction type” does not exist in the MLG/ULB base. Only feasible in the synthetic base (`merchant_category`) or after redefining it. |
| **RF-08** | Apply Machine Learning techniques to identify suspicious patterns and predict likely frauds. | APRES, p. 11 | Explicit | Binary supervised classification. The sketch's model is Random Forest, but that is illustrative. |
| **RF-09** | Evaluate model performance and identify the main factors associated with frauds. | APRES, p. 11 | Explicit | Adequate metrics for imbalance; interpretation compatible with the available attributes. |
| **RF-10** | Display general indicators (transactions, frauds, rate, and volume moved). | Sketch, p. 10 | Sketch | Useful, but the sketch values are not validated results. |
| **RF-11** | Provide a prediction tab with probability, decision threshold, confusion matrix, and variable importance. | Sketch, p. 10 | Sketch | Confirm whether it belongs in the MVP or is just illustration. |
| **RF-12** | Document the process and allow reproducing the results. | Recommended | Recommended | Not explicit in the materials, but necessary for academic defense. |

### 3.2 Non-functional requirements

| ID | Requirement | Status | Initial criterion |
|---|---|---|---|
| **RNF-01** | Reproducibility: seeds, versions, and pipeline documented. | Recommended | One person can redo the treatment, training, and main figures from the README. |
| **RNF-02** | Dashboard performance: loading and filtering without stalling. | Recommended | Set the goal after the prototype; pre-aggregate data if needed. |
| **RNF-03** | Usability: clear labels, coherent filters, consistent colors. | Recommended | Dashboard interface in **pt-BR** (KPIs, filters, graphs, tables, navigation, messages, empty/error states); code and documentation in English (D-09). Raw base values are not translated. |
| **RNF-04** | Privacy: no identifiable personal data. | Explicit (base is anonymized) | MLG/ULB is anonymized; the synthetic base contains no real data. Do not attempt re-identification. |
| **RNF-05** | Portability: documented local execution. | Recommended | No mandatory cloud dependency; decide whether to deploy later. |
| **RNF-06** | Code quality: Ruff, pytest, and typing when it makes sense. | Recommended | Keep the scaffold's existing commands. |
| **RNF-07** | Academic integrity: do not present illustrative numbers as results. | Recommended | All numbers in the report have traceable origins. |
| **RNF-08** | Consistency: dashboard, notebooks, and reports must show the same numbers. | Recommended | Validate against a reconciliation table. |
| **RNF-09** | Maintainability: separate loading, treatment, modeling, and visualization. | Recommended | Avoid duplicated logic between notebook and dashboard. |
| **RNF-10** | Deadline: deliver within the academic calendar. | Pending | The calendar was not provided; confirm with the professor. |

### 3.3 What is not a requirement

The following items appear in the sketch or context, but **must not** be treated as approved requirements:

- the sketch's exact values (284,907 transactions, R$ 25 million, 70% of frauds below R$ 250);
- the sketch's 99% / 94% / 92% / 93% metrics and its confusion matrix;
- the sketch's variable importance for V14, V10, and V17;
- the existence of an operational prediction tab ready to use;
- the “transaction type” filter in the MLG/ULB base;
- the period “01/01/2024 to 30/04/2025”;
- automatic transaction blocking.

Appendix A compares these items against the real base and explains the inconsistencies.

---

## 4. Database bases analysis

### 4.1 Base 1 — Give Me Some Credit

**Origin:** classic Kaggle competition. **Problem:** predict serious default — a person 90+ days overdue on any debt — within a two-year horizon.

**Known structure:**

- 150,000 training records + 101,503 test records;
- 10 explanatory variables and 1 target variable (`SeriousDlqin2yrs`);
- variables on use of rotating credit, age, defaults, debts, monthly income, open credit lines, mortgages, and dependents;
- known missing values in `MonthlyIncome` and `NumberOfDependents`;
- known sentinel and extreme values in default counts (for example 96 and 98) and `RevolvingUtilizationOfUnsecuredLines`;
- imbalanced target, about 6.7% positives in the training base.

**Fit to the project:** **low**. The base is a **credit default risk** problem, not fraud. It has no transactions, purchase amounts, merchants, devices, time slots, or fraud labels. There is also no key to cross-reference with the fraud bases. Using it would change the group's problem.

**Possibilities:** would be suitable for a credit scoring project with logistic regression or trees, evaluated with AUC. Not the case for this project.

**Recommendation:** do not use. Record in the report as an evaluated and discarded base, with justification.

### 4.2 Base 2 — Credit Card Fraud Detection (MLG/ULB)

**Origin:** collaboration between Worldline and the Machine Learning Group of Université Libre de Bruxelles (ULB). It is the base cited in the presentation and corresponds to the dashboard sketch.

**Known structure:**

- 284,807 transactions;
- 31 columns: `Time`, `V1` to `V28`, `Amount`, `Class`;
- 492 frauds, or 0.1727% of transactions;
- European card transactions in September 2013, over two days;
- `Time`: seconds since the first transaction in the base;
- `V1` to `V28`: principal components obtained by PCA; the original meaning was not disclosed due to confidentiality;
- `Amount`: transaction amount;
- `Class`: 1 for fraud, 0 for legitimate transactions;
- no missing values;
- official evaluation recommendation: **AUPRC** (area under the precision-recall curve), because accuracy is misleading on imbalanced bases.

**Local verification actually performed on 17/11/2026** (public file `creditcard.csv`, not versioned in the repository):

| Verification | Result |
|---|---|
| Rows | 284,807 |
| Columns | 31 |
| Frauds | 492 (0.1727%) |
| Missing values | 0 |
| Exact duplicates | 1,081 rows, of which 19 are frauds |
| `Time` | 0 to 172,792 seconds (about 2 days) |
| `Amount` total | 25,162,590.01 (original unit; the dictionary does not fix “euros”) |
| `Amount` min / max | 0.00 / 25,691.16 |
| Transactions with `Amount` = 0 | 1,825, of which 27 are frauds |
| Median of `Amount` | 22.00 in all transactions; 9.25 in frauds |

**Strengths:**

- real anonymized data with a fraud label;
- world-reference base, with literature and benchmarks;
- compatible with the presentation, the sketch, and the numbers 284,807 / 492 / 0.17%;
- allows binary classification and evaluation by precision, recall, F1, AUPRC, and confusion matrix.

**Limitations:**

- PCA-anonymized attributes: business factors behind V1–V28 cannot be named;
- only two days: no seasonality, long-term trend, or absolute dates;
- `Time` is relative; the origin is not documented as midnight, so “time of day” requires a premise;
- no card, merchant, category, device, or transaction type identifier;
- extreme imbalance (492 positives);
- 1,081 exact duplicates, including 19 frauds: removing them changes the positive class;
- data leakage risk if transformations are fitted before the train/test split.

**Fit:** **high**, provided the limitations are assumed and the dashboard does not promise analyses the base does not allow.

### 4.3 Base provided through the “Credit Card Fraud Detection Dataset” link (synthetic, 10,000)

**Origin:** `miadul/credit-card-fraud-detection-dataset` on Kaggle, used by the notebook indicated in the provided link. The title is similar to the classic base's, but the content is different.

**Known structure:**

- 10,000 synthetic transactions;
- 10 columns: `transaction_id`, `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, `is_fraud`;
- target `is_fraud`;
- the base page declares “highly imbalanced (fraud around 4–5%)” distribution;
- file-level inspection on 17/11/2026 found **151 frauds in 10,000 lines (1.51%)**, not 4–5%;
- no missing values and no exact duplicates on inspection;
- 24 hour values (0–23), 5 merchant categories, `foreign_transaction` and `location_mismatch` binary, `device_trust_score` declared 0 to 100 (file has 25 to 99), `velocity_last_24h` from 0 to 9;
- `amount` between 0.00 and 1,471.04 on inspection;
- base declared synthetic, Apache 2.0 license, created in December 2025.

**Strengths:**

- interpretable attributes directly linked to the sketch: value, time, merchant category, international transaction, location mismatch, device trust, speed, age;
- enables the “transaction type” filter if `merchant_category` is used as category;
- small, fast, and easy to explain;
- allows building the dashboard and demonstrating the complete flow without depending on PCA.

**Limitations:**

- it is **synthetic**: does not represent real fraud and must not support claims about the world;
- only 151 frauds, limiting metric stability;
- the base's own page reports a fraud rate that does not match the file;
- not the base cited in the presentation and not the sketch base;
- no absolute dates; currency not clearly defined;
- origin and generation process not documented in detail;
- using this base without updating the presentation would cause an academic inconsistency.

**Fit:** **medium as a documented alternative**, high for prototyping and an interpretable dashboard, low for claims about real fraud.

### 4.4 Comparison

| Criterion | Give Me Some Credit | MLG/ULB | Synthetic 10K |
|---|---|---|---|
| Problem | Default / credit risk | Credit card fraud | Credit card fraud |
| Unit | Person / credit | Transaction | Transaction |
| Records | 150,000 training + 101,503 test | 284,807 | 10,000 |
| Variables | 10 + target | 30 + target | 9 + target |
| Target | `SeriousDlqin2yrs` | `Class` | `is_fraud` |
| Positives | approx. 6.7% | 0.1727% (492) | 1.51% (151) |
| Interpretability | High | Low (PCA) | High |
| Temporal analysis | No | Limited to 2 relative days | Time of day (0–23) |
| Transaction type | No | No | `merchant_category` |
| Absolute date | No | No | No |
| Fit to scope | Low | High, with caveats | Medium, assuming synthetic |
| Recommended role | Discard | Main base | Documented alternative |

### 4.5 Recommendation on bases

1. **Recommended main base: MLG/ULB.** It is the base recorded in the presentation, appears in the references, and explains the sketch numbers. It is also the most defensible in academia as a real anonymized database.
2. **Do not use *Give Me Some Credit*.** It does not address the fraud problem and has no relationship to transactions.
3. **Do not mix bases.** If the synthetic base is used, it must be a separate study or a clearly labeled prototype, not an extension of the real base results.
4. **Group decision required.** If the priority is business interpretability and the “transaction type” filter, the synthetic base is more suitable — but the group must update the scope, presentation, and references, and make clear the findings are on synthetic data.
5. **If MLG/ULB is maintained:** remove or redefine the “transaction type” filter; treat the temporal analysis as relative to the two days; do not use “R$” without justifying the currency; and explain that the factors are PCA components, not business variables.

### 4.6 Mandatory treatment care

- **Duplicates:** decide and document. On MLG/ULB, removing all duplicates eliminates 19 frauds. If maintained, explain why; if removed, record the impact on the positive class.
- **Missing values:** check on the chosen base; MLG/ULB has no nulls and the synthetic one none either, by inspection. Even so, the pipeline must handle nulls.
- **Outliers:** do not remove automatically. Frauds may be exactly at the extremes. Investigate and document.
- **Imbalance:** treat only in training (class weights, undersampling, or oversampling). Never apply SMOTE before the train/test split.
- **Leakage:** split train/test before fitting scaling, imputation, attribute selection, or resampling.
- **Time:** on MLG/ULB, `Time` is not an absolute date. If deriving “time of day”, state the premise and test sensitivity; otherwise analyze hour blocks relative to elapsed time.
- **Currency:** MLG/ULB is European card and the dictionary does not fix “euros” explicitly. The sketch uses “R$”. Correct labels or justify conversion; do not convert without a source.
- **Scales:** `Amount` and `Time` have scales very different from the PCA components. Scale-sensitive models need standardization; trees do not, but the pipeline must be consistent.
- **Few positives:** avoid definitive conclusions from subgroups with few cases. Use cross-validation and report uncertainty when possible.

**Where to place files:** the `data/` area is already prepared. See `data/README.md` for the expected names (`data/raw/creditcard.csv`, `data/raw/credit_card_fraud_10k.csv`, or `data/raw/cs-training.csv`), links, and no-versioning rules. Place only the chosen base.

---

## 5. Technical and academic foundation

### 5.1 Lesson 2 lifecycle

Lesson 2 presents five stages: **understand the problem → data collection → preprocessing → exploratory data analysis → model construction**. The plan below preserves this logic and expands model construction into attribute preparation, training, and evaluation, because they are real dependencies.

| Lesson 2 stage | What the lesson highlights | How it is applied |
|---|---|---|
| Understand the problem | Define the problem before modeling | Section 2 and scope decisions (E1) |
| Data collection | Obtain and know the base | Acquisition, license, and dictionary (E2) |
| Preprocessing | Cleaning, sampling, imbalance, transformation | Audit and cleaning (E3) and final preparation (E5) |
| EDA | Inspect, summarize, and visualize; guide variables, transformations, and models | EDA of the three objectives (E4) |
| Model construction | Learn patterns and generate predictions/explanations | Training (E6) and evaluation (E7) |

### 5.2 Preprocessing

Lesson 2 defines four fronts:

- **Cleaning:** remove or treat missing values, duplicates, and inconsistent values. In this project, includes validating ranges, detecting duplicates, and recording removals.
- **Sampling:** select representative subsets. In the project, the main use is the stratified train/test split; additional sampling only if necessary for experiments.
- **Imbalance:** prevent biased models. In the project, this is central: 0.1727% frauds on MLG/ULB and 1.51% on the synthetic base.
- **Transformation:** normalize, standardize, and encode. In the project, apply `Amount`, `Time`, and the categorical variables of the synthetic base; V1–V28 are already PCA components.

### 5.3 Exploratory data analysis

Lesson 2 indicates that EDA should answer:

- how the data is distributed;
- whether there are patterns or trends;
- whether there are relationships between variables;
- whether there are outliers or unexpected behavior.

Recommended techniques: descriptive statistics (mean, median, standard deviation), visualizations (histograms, boxplots, scatter plots, heatmaps) and correlation analysis.

In the project, EDA must produce evidence for:

- **OE-1:** frauds vs non-frauds; distributions; frequency; outliers; correlations.
- **OE-2:** values and value bands; proportion of frauds by band; care with the `Amount` asymmetry.
- **OE-3:** temporal distribution; time of day/hour block; heatmap; limitations of the two-day window.

### 5.4 Model construction

Lesson 2 distinguishes:

- **supervised:** labeled data; classification and regression; decision tree example;
- **unsupervised:** search of patterns by clustering and association;
- **reinforcement:** learns from errors, reward, and policy.

For this project, the problem has a binary label and a predictive objective. Therefore, the main approach is **supervised classification**. Unsupervised techniques can be studied as a complement, but they are not necessary to fulfill the scope. Reinforcement learning does not apply and must not be added.

Reasonable candidate models, to compare:

- **baseline:** logistic regression or a simple decision tree, to establish reference;
- **main model:** Random Forest, Gradient Boosting, or another tree ensemble;
- **optional:** decision threshold adjustment and class weights;
- **optional:** simple risk rules, as in the synthetic base notebook, only as an interpretable baseline.

The sketch shows “Random Forest”, but that is an illustration. The final choice must be justified by performance, interpretability, and simplicity, not by the design.

### 5.5 Techniques observed in lesson notebooks

`Aula2.ipynb` demonstrates a complete flow with pandas:

- CSV reading and inspection (`head`, `sample`, `info`);
- null checks, descriptive statistics, and duplicates;
- validation of business rules (for example, count of ratings smaller than count of comments);
- column removal, rename, filtering, and index reset;
- date conversion and treatment of invalid dates;
- grouping and aggregation (`groupby`, `agg`);
- table joins (`merge`) and list explosion (`explode`);
- visualizations with matplotlib and seaborn (barplot, violinplot).

`Extra_Aula2.ipynb` adds:

- `value_counts` for age bands, professions, and states;
- bar and horizontal bar charts, and pie charts;
- titles and labels.

Lesson 2 also presents the **Google Colab** environment as the practices environment. The project can use Colab or local environment, as long as execution is documented and results can be reproduced. For the dashboard, local execution tends to be simpler to demonstrate.

**How to use:** these notebooks are technical reference and style level, not a mold to copy. The project should adapt the techniques to the fraud problem and maintain reproducible code.

### 5.6 Evaluation

MLG/ULB recommends **AUPRC** because of the imbalance. The plan adopts:

- **AUPRC** as the main ranking metric;
- **precision, recall, and F1** for the fraud class;
- **confusion matrix**;
- **ROC-AUC** as a complementary metric;
- **accuracy** only as context, never as the main metric;
- an **explicit decision threshold**, with discussion of the trade-off between false positives and false negatives;
- **cross-validation** in training and **holdout** for final evaluation.

Do not define arbitrary numerical goals before training the baseline. The initial criterion: the model must be better than baseline on AUPRC and show recall/precision compatible with the chosen threshold, with business justification.

### 5.7 Academic rigor

- Every fraud claim must indicate the base and limitation.
- Synthetic base results cannot be presented as evidence of real fraud.
- Skew numbers from the sketch cannot be reused without recomputation.
- Citations must follow the format requested by the institution (probably ABNT; confirm).
- The report must record decisions, transformations, metrics, limitations, and discarded alternatives.
- If AI tools are used, follow the institution's policy and declare usage when required.

---

## 6. Architecture and organization

### 6.1 Principles

- **Simplicity:** an academic project does not need production architecture.
- **Reproducibility:** anyone must be able to redo the flow.
- **Separation of responsibilities:** data, features, model, visualization, and dashboard must not be mixed.
- **Traceability:** each dashboard number must come from a pipeline-generated artifact.
- **Scope control:** the MVP is the four required visualizations; the rest is increments.

### 6.2 Logical flow

```text
Kaggle
  │
  ▼
data/raw/ (immutable, not versioned)
  │
  ▼
validation and audit ──► quality report
  │
  ▼
cleaning and transformation ──► data/processed/
  │
  ├──► EDA ──► figures and insights
  │
  ▼
features + train/test split
  │
  ▼
training ──► model ──► evaluation ──► metrics and artifacts
  │
  ▼
dashboard (aggregations + model results)
```

#### 6.2.1 Canonical layered flow (proposal)

The diagram above describes the data pipeline and modeling. The flow below is the **canonical** reference to the dashboard service path; it is a proposal for the implementation phase, without code created at this stage:

```text
Prepared Dataset (CSV)
    |
    v
Data Access
  - loading
  - schema validation
  - caching
    |
    v
Analytics / Domain
  - metrics
  - aggregations
  - analytical rules
    |
    v
Application Layer
  - filter orchestration
  - dashboard services
  - session state coordination
    |
    v
Presentation Layer
  - pages
  - filters
  - KPI components
  - charts
  - data tables
    |
    v
Interactive Dashboard
```

The dependency rules, the proposed folder mapping, and the role of the offline Machine Learning trail are in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), the reference document of this design. The base (P-01) and the tool (P-04) are still pending: the flow is agnostic to both.

### 6.3 Folder structure

The **data area was already created** to receive the bases: `data/raw/`, `data/interim/`, `data/processed/`, and `data/README.md`, with instructions on where to place each file. The structure below is what was implemented in the MVP: the layer folders exist in `src/credit_card_fraud/` and reports are generated in `reports/`. The original scaffold (`main.py`, tests, and Ruff/pytest settings) was preserved.

```text
credit-card-fraud/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── ARCHITECTURE.md           # canonical reference of the layered flow
│   ├── DATA_DICTIONARY.md        # to be created in the data step
│   └── DECISIONS.md              # optional: decision record
├── data/
│   ├── README.md                 # instructions for the data area
│   ├── raw/                      # original files, not versioned
│   ├── interim/                  # intermediate data, not versioned
│   └── processed/                # final data, not versioned
├── notebooks/
│   ├── 01_auditoria.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modelagem.ipynb
├── src/credit_card_fraud/
│   ├── __init__.py
│   ├── config.py                 # paths, seeds, constants
│   ├── data/                     # Data Access: load, validation, cache
│   ├── analytics/                # Analytics/Domain: metrics, aggregations, rules
│   ├── application/              # Application: filters, services, session state
│   ├── features/                 # offline trail: transformations for modeling
│   ├── models/                   # offline trail: training, evaluation
│   └── presentation/             # Presentation: pages, filters, KPIs, graphs, tables
├── reports/
│   └── figures/                  # generated figures
└── tests/
```

**Proposed changes (not approved):** `analytics/` and `application/` are new directories proposed; the previous proposal's `viz/` and `dashboard/` are **absorbed** by `presentation/`; `features/` and `models/` remain in the ML offline trail (E5–E7), without renaming. The complete per-layer mapping is in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md). No folder should be created while the phase is planning.

**Cautions:**

- the `data/` area is already protected in `.gitignore`; keep data files out of Git;
- consult `data/README.md` for base names and links;
- do not edit `data/raw/` files; generate treated copies in `data/interim/` or `data/processed/`;
- do not duplicate logic between notebook and module: the notebook can import functions from `src/`;
- keep the existing entrypoint and tests working;
- create only the modules needed, not an empty structure.

### 6.4 Technologies

| Layer | Current situation | Proposal | Status |
|---|---|---|---|
| Language | Python ≥3.11 | Keep | Current |
| Environment | `uv` + `.venv` | Keep | Current |
| Lint/format | Ruff | Keep | Current |
| Tests | pytest | Keep and expand | Current |
| Manipulation | — | pandas, numpy | Suggested |
| Visualization | — | matplotlib, seaborn | Suggested |
| ML | — | scikit-learn | Suggested |
| Imbalance | — | `class_weight`; optional `imbalanced-learn` | Suggested |
| Dashboard | — | Streamlit (simpler) or Dash/Plotly | **Decide** |
| Notebooks | — | Jupyter/Colab for EDA and experiments | Suggested |
| Deployment | — | Not needed in MVP | **Decide** |

**Recommendation:** use Streamlit if there is no institutional requirement for another tool, because it reduces interface effort and keeps the dashboard in Python. Dash/Plotly offers more visual control, but increases complexity. Do not add dependencies before the tool is chosen.

### 6.5 Dashboard data layer

For the MLG/ULB base, the dashboard should not load 284,000 lines and recompute everything on each interaction if it harms performance. Options:

- pre-aggregate indicators and time series in light files;
- cache loading;
- load only the necessary columns.

Individual prediction, if it enters the scope, should use the same transformation pipeline as training, not a parallel reimplementation.
---

## 7. Development steps

### 7.1 Sequence and dependencies

```text
E1 Scope
  │
  ▼
E2 Data
  │
  ▼
E3 Audit/cleaning ◄──┐
  │                     │
  ▼                     │
E4 EDA ─────────────────┘
  │
  ▼
E5 Features/split
  │
  ▼
E6 Modeling ◄──┐
  │             │
  ▼             │
E7 Evaluation ───┘
  │
  ▼
E8 Dashboard
  │
  ▼
E9 Integration/validation
  │
  ▼
E10 Academic deliverables
```

The feedback arrows indicate expected iterations: EDA can reveal problems that go back to cleaning; evaluation can require retraining; the dashboard can reveal new cuts for EDA.

### E1 — Scope and problem definition

**Goal:** align the group on problem, objectives, audience, scope, base, tool, and success criteria before coding.

**Activities:**

- review the presentation and this plan with the group;
- confirm the base and version (MLG/ULB or synthetic);
- confirm the target audience and dashboard uses;
- define the MVP: four required visualizations and feasible filters;
- decide the dashboard tool and whether there will be a prediction tab;
- confirm format, deadlines, and rubric with the professor;
- record the decisions in Section 10 and update the [`AGENTS.md`](AGENTS.md).

**Knowledge:** problem definition, lifecycle, requirements, scope.

**Dependencies:** provided materials, group availability, professor guidance.

**Expected result:** approved scope, recorded decisions, resolved critical pending items.

**Done when:**

- the chosen base is recorded with link and version;
- the audience and MVP are defined;
- the dashboard tool is chosen;
- the delivery format and calendar are confirmed;
- there are no conflicting requirements without a decision.

### E2 — Data acquisition and governance

**Goal:** obtain the correct base, ensure provenance, license, and storage.

**Activities:**

- download the base from the official source;
- record URL, access date, version, and license;
- compute hash or checksum and verify rows/columns;
- create the data dictionary;
- store the raw file in `data/raw/` without changes, following [`data/README.md`](data/README.md);
- verify the data area is still protected by `.gitignore`;
- check for personal data or usage restrictions.

**Knowledge:** collection, governance, licensing, privacy.

**Dependencies:** E1 (chosen base).

**Expected result:** raw base, data dictionary, provenance record.

**Done when:**

- the base can be obtained again from a documented source;
- the checksum and dimensions check out;
- the license is known;
- no PII in the repository.

### E3 — Audit and preprocessing

**Goal:** diagnose data quality and produce a cleaned base with documented decisions.

**Activities:**

- inspect types, ranges, nulls, duplicates, and invalid values;
- measure the target distribution;
- investigate duplicates (on MLG/ULB: 1,081 rows, of which 19 frauds);
- investigate zero values, outliers, and inconsistencies;
- decide what to remove, correct, or keep;
- document every rule and the impact on the fraud count;
- create the quality report.

**Knowledge:** cleaning, validation, descriptive statistics, missing-data analysis.

**Dependencies:** E2.

**Expected result:** cleaned base, quality report, transformation log.

**Done when:**

- all quality problems are quantified;
- each removal has a justification;
- the impact on the positive class is recorded;
- the cleaned base is reproducible by script or notebook.

### E4 — Exploratory data analysis

**Goal:** understand distributions, patterns, relationships, and anomalies, and answer the three specific objectives.

**Activities:**

- general and per-class descriptive statistics;
- histograms, boxplots, and scatter plots for value and relevant attributes;
- frauds vs non-frauds comparison;
- value-band analysis;
- temporal analysis (with the base limitation);
- heatmap and correlation;
- outlier and unexpected behavior identification;
- record insights and hypotheses for modeling;
- draft the dashboard visualizations.

**Knowledge:** EDA, descriptive statistics, visualization, correlation.

**Dependencies:** E3.

**Expected result:** EDA report, figures, findings list.

**Done when:**

- each specific objective has visual or statistical evidence;
- base limitations are explicit;
- no causal claims without support;
- findings guide variable selection and model type.

### E5 — Attributes and train/test split

**Goal:** build the modeling base reproducibly and without leakage.

**Activities:**

- define input and target variables;
- create justified indicators (value bands, time/hour block, categories);
- encode categorical variables;
- standardize numeric variables when needed;
- split train and test stratified;
- define the imbalance strategy (class weights, undersampling, or oversampling) only in training;
- assemble a preprocessing pipeline;
- document the features dictionary.

**Knowledge:** transformation, encoding, scaling, sampling, leakage prevention.

**Dependencies:** E3 and E4; decision on the model.

**Expected result:** feature pipeline, train/test sets, attribute dictionary.

**Done when:**

- the pipeline runs end to end;
- the test set does not participate in any fitting;
- the imbalance strategy is justified;
- the features are comprehensible and documented.

### E6 — Modeling

**Goal:** train and compare classification models to estimate fraud probability.

**Activities:**

- train a simple baseline;
- train at least one candidate model (tree, Random Forest, or boosting);
- use cross-validation in training;
- tune hyperparameters with a defined criterion;
- test class weights or resampling;
- record seeds, versions, and results;
- choose the final model with justification.

**Knowledge:** supervised learning, classification, trees, ensembles, cross-validation.

**Dependencies:** E5.

**Expected result:** trained models, recorded experiments, selected model.

**Done when:**

- at least two models compared;
- baseline is defeated or justified otherwise;
- experiments are reproducible;
- the final model is saved with its corresponding pipeline.

### E7 — Evaluation and interpretation

**Goal:** measure performance on holdout, choose the decision threshold, and interpret associated factors.

**Activities:**

- compute AUPRC, precision, recall, F1, ROC-AUC, and confusion matrix;
- analyze the decision threshold effect;
- compare with the baseline;
- analyze errors (false positives and false negatives);
- extract variable or permutation importance;
- discuss the meaning of factors, respecting MLG/ULB's PCA limitation;
- record limitations and generalization risks;
- produce a summarized “model card”.

**Knowledge:** classification evaluation under imbalance, interpretability, error analysis.

**Dependencies:** E6.

**Expected result:** evaluation report, figures, documented model.

**Done when:**

- metrics are reproducible;
- accuracy is not the main metric;
- the threshold is justified;
- factors are presented with proper caution;
- limitations are in the report.

### E8 — Dashboard

**Goal:** deliver an interactive dashboard with the four minimum visualizations and consistent data.

**Activities:**

- define layout and navigation;
- implement general indicators;
- implement fraud distribution;
- implement frauds by value band;
- implement temporal evolution;
- add feasible filters (period, value band, status);
- decide and implement the prediction tab if approved;
- correct currency and time labels;
- validate numbers against EDA and the model;
- test usability and performance;
- document how to run it.

**Knowledge:** visualization, dashboard design, data storytelling.

**Dependencies:** E4 (visualizations) and E7 (model results); tool chosen in E1.

**Expected result:** executable application with the four visualizations and usage documentation.

**Done when:**

- the four required visualizations are present and working;
- filters do not break;
- numbers match the artifacts;
- no sketch dummy values;
- local execution is documented;
- the dashboard is understandable by the target audience.

### E9 — Integration and final validation

**Goal:** ensure the whole works end to end and is reproducible.

**Activities:**

- run the pipeline in a clean environment;
- verify dependencies and README instructions;
- run tests and lint;
- validate the requirements traceability matrix;
- review data, metrics, and figures;
- review security and privacy;
- fix inconsistencies;
- freeze the version for delivery.

**Knowledge:** tests, reproducibility, documentation, review.

**Dependencies:** E2 to E8.

**Expected result:** integrated, validated, and documented repository.

**Done when:**

- the chosen base is documented and treated;
- EDA answers the three specific objectives;
- the model is trained, evaluated, and interpreted with adequate metrics;
- the dashboard has the four required visualizations and consistent numbers;
- the pipeline can be reproduced in a clean environment;
- the report and presentation align with the rubric;
- all critical pending items in Section 10 are resolved or explicitly assumed as limitations.

### E10 — Academic deliverables

**Goal:** produce the report, presentation, and demo aligned to the rubric.

**Activities:**

- write introduction, problem, objectives, and justification;
- document methodology, data, treatment, EDA, modeling, and evaluation;
- present results with figures and tables;
- discuss limitations and future work;
- review citations and references;
- prepare slides and a demonstration script;
- rehearse the defense;
- review rubric adherence.

**Knowledge:** scientific writing, communication, ethics, presentation.

**Dependencies:** E9 and the official rubric.

**Expected result:** report, slides, demo, and organized code.

**Done when:**

- the rubric is met;
- sources are cited;
- results are traceable;
- the demo works;
- the group can explain and defend the decisions.

---

## 8. Dependencies and risks

### 8.1 Critical dependencies

| Dependency | Blocks | Note |
|---|---|---|
| Base definition (MLG/ULB or synthetic) | E2–E8 | Otherwise the dashboard scope and metrics change. |
| Dashboard tool | E8 | Streamlit, Dash, or other; decide before implementation. |
| Target audience | E8 and E10 | Affects language, filters, and narrative. |
| Rubric and deadlines | E10 | Not provided in the materials. |
| Metrics strategy and threshold | E7 | Depends on the business objective. |
| Prediction tab decision | E8 | Can be MVP or increment. |
| AI usage policy and citations | E10 | Confirm with the institution. |

### 8.2 Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| **R-01** | Ambiguity between MLG/ULB and the synthetic base. | High: changes scope, numbers, and conclusions. | Decide in E1; record version and link; do not mix. |
| **R-02** | Using sketch numbers as if they were results. | High: academic integrity. | Recompute everything; remove or label the sketch as illustrative. |
| **R-03** | Extreme imbalance. | High: apparently good but useless model. | AUPRC, precision, recall, F1, confusion matrix; class weights/resampling. |
| **R-04** | Data leakage. | High: inflated metrics. | Split before transforming; pipeline; validation. |
| **R-05** | Duplicates in MLG/ULB. | Medium: 19 frauds removed if discarded. | Decide and document; compare scenarios. |
| **R-06** | PCA-anonymized attributes. | Medium: limited business explanation. | Explain the limitation; use component importance and `Amount`. |
| **R-07** | Relative time and two-day window. | Medium: limited temporal analysis. | Analyze relative blocks; do not claim seasonality; state assumptions. |
| **R-08** | Sketch currency (R$) differs from the base. | Medium: wrong charts and conclusions. | Define the unit; fix labels; do not convert without a source. |
| **R-09** | “Transaction type” filter nonexistent in MLG/ULB. | Medium: impossible requirement. | Remove, redefine, or migrate to the synthetic base. |
| **R-10** | Few positives (492 or 151). | Medium: unstable metrics. | Cross-validation; avoid tiny subgroups; report uncertainty. |
| **R-11** | Dashboard scope growing too much. | Medium: delay. | MVP with four visualizations; increments later. |
| **R-12** | Poorly chosen dashboard tool. | Medium: rework. | Quick prototype; decide in E1. |
| **R-13** | Kaggle access or base version change. | Low/medium: rework. | Download early; keep raw copy; record checksum. |
| **R-14** | Presenting the synthetic base as real fraud data. | High: credibility. | Label clearly; do not use for claims about real fraud. |
| **R-15** | Missing rubric/deadlines. | Medium: delivery outside expectations. | Confirm with the professor before E10. |
| **R-16** | Inconsistency between notebook and dashboard numbers. | Medium: divergent numbers. | Single-source artifacts; reconciliation tests. |

### 8.3 Explicit premises

These premises are **not** facts from the materials; they must be validated:

- the dashboard will be used by risk/fraud analysts and managers, not final customers;
- local execution is enough; there is no deployment requirement;
- the project will be delivered as report + presentation + code;
- the institution accepts the use of public anonymized data;
- the group can use Python and open-source libraries;
- no size restriction or tool prevents using MLG/ULB.

### 8.4 Constraints

- limited academic time;
- team of five, with undocumented availability;
- public base, no possibility of collecting own data;
- real base has only two days and anonymized attributes;
- project is academic, not commercial product.

---

## 9. Validation criteria

### 9.1 Gates per step

| Step | Acceptance criterion |
|---|---|
| E1 | Base, audience, MVP, tool, and delivery format defined and recorded. |
| E2 | Base obtained; checksum/dimensions verified; license and dictionary documented. |
| E3 | Quality audited; removals justified; positive-class impact recorded. |
| E4 | Each specific objective answered with statistics/figures; limitations explicit. |
| E5 | Reproducible pipeline; stratified split; no leakage; imbalance handled in training. |
| E6 | At least two models compared; cross-validation; seeds recorded. |
| E7 | Complete metrics; justified threshold; error analysis; limitations recorded. |
| E8 | Four mandatory visualizations; working filters; numbers reconciled. |
| E9 | Clean-environment execution; tests/lint; documentation; traceability. |
| E10 | Rubric met; sources cited; demo; rehearsal done. |

### 9.2 Data checklist

- [ ] URL, access date, version, and license recorded.
- [ ] Dimensions and checksum verified.
- [ ] Data dictionary created.
- [ ] Nulls, duplicates, outliers, and invalid values evaluated.
- [ ] Target distribution recorded.
- [ ] Duplicate decision documented.
- [ ] No identifiable personal data in the repository.
- [ ] Raw data immutable and not versioned.

### 9.3 Modeling checklist

- [ ] Train/test split defined before transformations.
- [ ] Baseline trained.
- [ ] At least two models compared.
- [ ] Cross-validation used in training.
- [ ] Imbalance handled only in training.
- [ ] AUPRC, precision, recall, F1, ROC-AUC, and confusion matrix computed.
- [ ] Decision threshold justified.
- [ ] Variable importance interpreted with caution.
- [ ] Seeds and versions recorded.
- [ ] Model and pipeline saved together.

### 9.4 Dashboard checklist

- [ ] General indicators.
- [ ] Fraud distribution.
- [ ] Frauds by value band.
- [ ] Temporal evolution.
- [ ] Feasible and coherent filters.
- [ ] Currency and time labels correct.
- [ ] Numbers reconciled against EDA/model.
- [ ] No sketch dummy values.
- [ ] Local execution documented.
- [ ] Usability tested with someone from the group.

### 9.5 Academic checklist

- [ ] Problem, objectives, and justification clear.
- [ ] Methodology aligned to Lesson 2 lifecycle.
- [ ] Bases described and choice justified.
- [ ] Treatment and limitations documented.
- [ ] Results reproducible.
- [ ] Figures and tables with sources.
- [ ] Citations in the required format.
- [ ] Limitations and future work included.
- [ ] AI use declared, if required.
- [ ] Presentation and demo rehearsed.

### 9.6 “Done” criterion for the project

The project is ready for delivery when:

1. the chosen base is documented and treated;
2. EDA answers the three specific objectives;
3. the model is trained, evaluated, and interpreted with adequate metrics;
4. the dashboard has the four required visualizations and consistent numbers;
5. the pipeline can be reproduced in a clean environment;
6. the report and presentation align with the rubric;
7. all critical pending items in Section 10 are resolved or explicitly assumed as limitations.

---

## 10. Pending, decisions, and recommendations

### 10.1 Recorded decisions

| ID | Decision | Evidence | Status |
|---|---|---|---|
| **D-01** | The theme is credit card fraud detection. | APRES, p. 2–3 | Done |
| **D-02** | The general objective is an interactive dashboard with predictive analysis. | APRES, p. 4 | Done |
| **D-03** | The specific objectives are patterns, value, and time. | APRES, p. 5–6 | Done |
| **D-04** | The dashboard must have at least four visualizations. | APRES, p. 10 | Done |
| **D-05** | Machine Learning will be applied. | APRES, p. 11 | Done |
| **D-06** | The base will be a Kaggle fraud base. | APRES, p. 9 | Partial: which one is pending |
| **D-07** | The project follows the Lesson 2 lifecycle. | AULA2, p. 3–16 | Academic guidance |
| **D-08** | OLD: documentation and code in Portuguese. | Repository convention and materials | Superseded by D-26 (documentation, code, reports, and identifiers now in English). |
| **D-09** | OLD: dashboard interface in English; documentation and code remain in Portuguese. | Confirmed by the group | Superseded by D-26 (revoked). |
| **D-10–D-25** | Implementation decisions: MLG/ULB base, exact duplicate removal, relative time, currency in original unit, Streamlit, feasible filters, models, decision threshold, metrics, split, and layered organization. | [`DECISIONS.md`](DECISIONS.md) | Done and applied in the pipeline. |
| **D-26** | Dashboard interface in **Brazilian Portuguese (pt-BR)**, revoking D-09/D-17. | Explicit refinement requirement for the dashboard; better usability for the academic audience. | All KPIs, filters, graphs, tables, states, and messages in pt-BR; documentation, identifiers, and code comments remain in English (no change). |
| **D-27** | Use **Plotly** for dashboard graphs. | Interactive tooltips and consistent dark theme require interactive graphs; static matplotlib does not meet this. | New `plotly` dependency; components in `presentation/components.py` return Plotly figures. |
| **D-28** | pt-BR number formatting (`1.250,00`, `1,25%`), **no** currency symbol, **no** calendar dates. | Reinforces D-13/D-14: the base defines no currency or absolute date; only formatting follows the Brazilian pattern. | Single source in `formatting.py`; labels keep “original unit” and “relative hours/days”. |
| **D-29** | Inclusion of a paginated **transaction explorer** and **offline model indicators**. | Leverages existing `random_sample`/`subset` and `reports/model/metrics.json`; no new modeling. | Filterable table with available columns (no ID/category); model metrics section marked static. |

> **Note:** the detailed records of D-10 to D-25, with justification and impact, are in [`DECISIONS.md`](DECISIONS.md).

### 10.2 Recommendations

| ID | Recommendation | Rationale |
|---|---|---|
| **R-01** | Use MLG/ULB as the main base. | It is the base cited in the presentation, compatible with the sketch and real anonymized data. |
| **R-02** | Do not use *Give Me Some Credit*. | It is about default risk, not fraud, and has no transactional variables. |
| **R-03** | Do not mix bases. | Mixing sources confuses results and conclusions. |
| **R-04** | If using the synthetic base, label it as synthetic everywhere. | Avoids claims about real fraud. |
| **R-05** | Adopt AUPRC as the main metric. | Official base recommendation and appropriate for imbalance. |
| **R-06** | Compare baseline and at least one candidate model. | Avoids choosing Random Forest just because it appears in the sketch. |
| **R-07** | Use Streamlit unless there is a contrary institutional requirement. | Lower complexity for an academic dashboard in Python. |
| **R-08** | Build the MVP with the four visualizations before increments. | Controls scope and timeline. |
| **R-09** | Remove or redefine “transaction type” in MLG/ULB. | The field does not exist; do not invent data. |
| **R-10** | Fix currency and temporal axis in the dashboard. | The sketch uses R$ and dates that do not exist in the base. |
| **R-11** | Create a data dictionary and a decision record. | Academic traceability. |
| **R-12** | Confirm rubric, deadlines, and format with the professor. | Not provided in the materials. |

### 10.3 Pending

| ID | Pending item | Why it matters | Suggested path |
|---|---|---|---|
| **P-01** | Which base is used: MLG/ULB or synthetic 10K? | Changes scope, metrics, dashboard, and narrative. | **Resolved (D-10):** MLG/ULB adopted as the single base. |
| **P-02** | The provided “Base 2” link points to a notebook of the synthetic base, not to MLG/ULB. What was the intention? | Avoids using the wrong base. | **Resolved (D-10):** the pipeline uses `data/raw/creditcard.csv` (MLG/ULB). |
| **P-03** | What is the target audience? | Affects language, filters, and narrative. | Define with the group/professor. |
| **P-04** | What dashboard tool? | Defines dependencies and architecture. | **Resolved (D-15):** Streamlit. |
| **P-05** | Will there be a prediction tab? | Can greatly increase scope. | **Out of MVP (D-18):** future increment; decide before implementation. |
| **P-06** | Which models and decision threshold? | Defines metrics and results. | **Defined in pipeline (D-19/D-20):** baseline + RF + boosting; threshold by F1 out-of-fold. |
| **P-07** | What imbalance strategy? | Affects performance and interpretation. | **Defined in pipeline (D-19):** `class_weight="balanced"`, no resampling. |
| **P-08** | Will the “transaction type” filter be removed or should the base change? | MLG/ULB has no such field. | **Resolved (D-16):** filter removed; interface reports the absence. |
| **P-09** | How to treat time in MLG/ULB? | `Time` is relative and covers two days. | **Resolved (D-13):** relative blocks (elapsed hour, relative day). |
| **P-10** | Which currency in the graphs? | The sketch uses R$ and the base is in euros. | **Resolved (D-14):** original unit, no currency symbol. |
| **P-11** | Format, deadlines, and rubric? | Not in the materials; the Aula 2 “CALENDAR” slide has no readable dates. | Confirm with the professor. |
| **P-12** | Group roles and data storage/versioning? | Affects the schedule. | Define responsible persons per step. |
| **P-13** | Data storage and versioning? | Avoids accidental commits and lost versions. | `data/` in `.gitignore`; checksum; raw copy. |
| **P-14** | AI citation and use policy? | Can affect the grade. | Consult the professor/advisor. |
| **P-15** | Can sketch numbers be used as visual reference? | Risk of confusing with results. | Use only as layout; never as results. |

### 10.4 Decisions that must not be made by assumption

- do not assume the synthetic base is the presentation's base;
- do not assume “R$” is the correct currency;
- do not assume `Time = 0` is midnight;
- do not assume the model is Random Forest;
- do not assume accuracy is the main metric;
- do not assume the prediction tab is part of the MVP;
- do not assume the dashboard is published on the internet;
- do not assume the audience is end customers;
- do not assume the sketch metrics are real.

---

## 11. Planned deliverables

| ID | Deliverable | Type | Step | Status |
|---|---|---|---|---|
| **EN-01** | `AGENTS.md` with project context | Documentation | E1 | **Created in this phase** |
| **EN-02** | `docs/PROJECT_PLAN.md` | Documentation | E1 | **Created in this phase** |
| **EN-03** | Data dictionary and provenance record | Data | E2 | **Completed** — `docs/DATA_DICTIONARY.md` |
| **EN-04** | Treated base + quality report | Data | E3 | **Completed** — `data/processed/` and `reports/RELATORIO_QUALIDADE.md` |
| **EN-05** | EDA notebook/report with figures | Analysis | E4 | **Completed** — `reports/RELATORIO_EDA.md` and figures |
| **EN-06** | Feature pipeline and train/test sets | Modeling | E5 | **Completed** — `features/` |
| **EN-07** | Trained model + experiments | Modeling | E6 | **Completed** — `models/` and `reports/model/metrics.json` |
| **EN-08** | Evaluation report + documented model | Modeling | E7 | **Completed** — `reports/RELATORIO_MODELO.md` |
| **EN-09** | Interactive dashboard | Product | E8 | **Completed** — `presentation/app.py` |
| **EN-10** | Integrated repo + validation | Engineering | E9 | Partial — tests and lint executed; final review pending |
| **EN-11** | Final report | Academic | E10 | Planned; format pending |
| **EN-12** | Slides + demo | Academic | E10 | Planned; format pending |
| **EN-13** | Code, tests, and run instructions | Engineering | E9–E10 | **Completed** — README and pytest suite |
| **EN-14** | `docs/ARCHITECTURE.md` | Documentation | E1 | **Created in this phase** |

**Academic deliverables not specified:** in the materials, there is no indication of format (PDF, article, video, use), page count, deadline, or rubric. Confirm before starting E10.

---

## 12. Traceability matrix

| Requirement | Source | Step | Evidence of completion |
|---|---|---|---|
| RF-01 | APRES, p. 9 | E1–E2 | Link, version, and checksum recorded |
| RF-02 | APRES, p. 9 | E3 | Treated base + log of transformations |
| RF-03 | APRES, p. 5 | E4 | EDA with frauds vs non-frauds comparison |
| RF-04 | APRES, p. 5 and 10 | E4/E8 | Graph of frauds by value band |
| RF-05 | APRES, p. 6 | E4/E8 | Time series + heatmap |
| RF-06 | APRES, p. 10 | E8 | Four visualizations in the dashboard |
| RF-07 | Sketch | E1/E8 | Filters implemented or redefined |
| RF-08 | APRES, p. 11 | E6 | Model trained |
| RF-09 | APRES, p. 11 | E7 | Metrics + interpretation |
| RF-10 | Sketch | E8 | KPIs in the dashboard |
| RF-11 | Sketch | E1/E8 | Prediction tab decided and implemented or discarded |
| RF-12 | Recommended | E9 | README + reproducible pipeline |
| RNF-01 to RNF-10 | See Section 3.2 | E1–E10 | Section 9 checklist |

---

## 13. Immediate next steps

1. **Review results with the group:** check the generated reports
   (`reports/RELATORIO_QUALIDADE.md`, `reports/RELATORIO_EDA.md`, and
   `reports/RELATORIO_MODELO.md`) and validate the decisions D-10 to D-29.
2. **Confirm with the professor (P-11/P-14):** rubric, deadlines, delivery
   format, and AI use policy.
3. **Define the target audience (P-03):** the hypothesis of risk/fraud
   analysts and risk managers is not yet validated.
4. **Decide about the prediction tab (P-05):** out of MVP (D-18); if
   included, reuse the saved pipeline in `reports/model/`.
5. **Prepare the academic deliverables (E10):** report, slides, and demo
   script, with the numbers recalculated by the pipeline.
6. **Final review (E9):** run the pipeline in a clean environment, run
   tests and lint, and review the Section 9 checklists.

---

## Appendix A — Dashboard sketch reconciliation

The sketch from APRES, p. 10 is useful as visual reference, but **it is not a validated result**. The table below compares the sketch values with the real MLG/ULB base, verified on 17/11/2026.

| Sketch item | Sketch value | MLG/ULB check | Conclusion |
|---|---|---|---|
| Transactions | 284,907 | 284,807 | Probable typo; the donut sums 284,315 + 492 = 284,807. |
| Frauds | 492 | 492 | Correct. |
| Fraud rate | 0.17% | 0.1727% | Correct with rounding. |
| Non-frauds | 284,315 | 284,315 | Correct. |
| Volume moved | R$ 25 million | 25,162,590.01 in original unit | Same order of magnitude, but “R$” is not justified; the base is European cards. |
| Frauds by band (0–10, 10–50, 50–100, 100–250, 250–500, 500-1k, 1k+) | 130, 70, 45, 100, 70, 55, 22 | 249, 56, 57, 56, 39, 26, 9 | The sketch graph does not match the real distribution. Do not reuse. |
| Insight “70% of frauds below R$ 250” | 70% | 85.0% of frauds with value ≤ 250; 73.6% ≤ 100 | The sketch insight does not check. Recompute. |
| Model metrics | 99% accuracy, 94% precision, 92% recall, 93% F1 | Not validated | The sketch says “illustrative”; recompute and document. |
| Confusion matrix | TN 56,858; FP 6; FN 8; TP 90 | Consistent with a test of about 20% (56,962 rows, 98 frauds), but not validated | Can be a real result or a simulation; do not use without reproduction. |
| Variable importance | V14 18%, V10 11%, V17 10% | Not validated | The sketch says “values illustrative”; recompute. |
| “Transaction type” filter | Present | Does not exist in MLG/ULB | Remove or switch base. |
| Period “01/01/2024–30/04/2025” | Present | Does not exist; base covers two days of 2013 in relative time | Correct the axis; do not invent dates. |
| Heatmap “0h–21h” | Present | `Time` is relative; origin not documented as midnight | State the premise or use relative blocks. |
| “High risk: block and investigate” | Present | Not a project policy | Treat as UI illustration, not as functionality. |

**Rule:** the sketch serves for layout and inspiration. No number, insight, or metric from it may enter the report without recomputation and documentation.

---

## Appendix B — Summary data dictionary

### B.1 MLG/ULB — *Credit Card Fraud Detection*

| Column | Type | Description |
|---|---|---|
| `Time` | numeric | Seconds since the first transaction in the base; not absolute date/time. |
| `V1`–`V28` | numeric | Anonymized principal components (PCA); original meaning not disclosed. |
| `Amount` | numeric | Transaction value. |
| `Class` | binary | 1 = fraud; 0 = legitimate. |

### B.2 Synthetic — *Credit Card Fraud Detection Dataset* (10K)

| Column | Type | Description |
|---|---|---|
| `transaction_id` | integer | Transaction identifier. |
| `amount` | numeric | Transaction value. |
| `transaction_hour` | integer | Transaction hour (0–23). |
| `merchant_category` | categorical | Merchant category (5 values). |
| `foreign_transaction` | binary | International transaction (0/1). |
| `location_mismatch` | binary | Mismatch between billing location and transaction (0/1). |
| `device_trust_score` | numeric | Device trust (0–100). |
| `velocity_last_24h` | integer | Number of transactions in the last 24 hours. |
| `cardholder_age` | integer | Age of the cardholder. |
| `is_fraud` | binary | 1 = fraud; 0 = legitimate. |

### B.3 Give Me Some Credit

| Column | Description |
|---|---|
| `SeriousDlqin2yrs` | Person 90+ days overdue in two years (target). |
| `RevolvingUtilizationOfUnsecuredLines` | Use of rotating unsecured credit. |
| `age` | Age. |
| `NumberOfTime30-59DaysPastDueNotWorse` | Defaults of 30–59 days. |
| `DebtRatio` | Monthly debt divided by gross income. |
| `MonthlyIncome` | Monthly income. |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines and loans. |
| `NumberOfTimes90DaysLate` | Number of 90+ days defaults. |
| `NumberRealEstateLoansOrLines` | Real estate loans and lines of credit. |
| `NumberOfTime60-89DaysPastDueNotWorse` | Defaults of 60–89 days. |
| `NumberOfDependents` | Number of dependents. |

---

## Appendix C — References

- DAL POZZOLO, Andrea; CAELEN, Olivier; JOHNSON, Reid A.; BONTEMPI, Gianluca. **Credit Card Fraud Detection Dataset**. Kaggle, 2015. Available at: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Accessed: 17 Nov. 2026.
- GÉRON, Aurélien. **Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow**. 2. ed. Rio de Janeiro: Alta Books, 2021.
- KAGGLE. **Give Me Some Credit**. Available at: https://www.kaggle.com/c/GiveMeSomeCredit. Accessed: 17 Nov. 2026.
- KAGGLE. **Credit Card Fraud Detection Dataset** (`miadul/credit-card-fraud-detection-dataset`). Available at: https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset. Accessed: 17 Nov. 2026.
- NILSON REPORT. **Payment Card Fraud Worldwide**. Ed. 1298, Jan. 2026. Available at: https://nilsonreport.com/newsletters/1298/. Accessed: 17 Nov. 2026.
- OLIVEIRA, Vinicius Dias. Methods of machine learning in credit card fraud detection: a comparative study. **Revista Científica ACERTTE**, v. 5, n. 9, 2025. Available at: https://acertte.org/acertte/article/view/265. Accessed: 17 Nov. 2026.
- PROVOST, Foster; FAWCETT, Tom. **Data Science for Business: What You Need to Know About Data Mining and Data Analytic Thinking**. Rio de Janeiro: Alta Books, 2016.
- SANTOS, Aline Bertolazo dos; SANTOS, Isadora Disposti Bueno dos; TOMA, Karina Mitiko. **Fraudulent credit card transactions: a comparative machine learning study**. Available at: https://ojs.focopublicacoes.com.br/foco/article/view/8328. Accessed: 17 Nov. 2026.
- SENAC. **Introduction to Data Science — Webconference 2** (`aula2.pdf`). Course material, 2026.
- SENAC. **Códigos da Aula 2** (`Codigos_Aula2.zip`). Course material, 2026.
