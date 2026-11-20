# Proposals for the dashboard architecture

**Grupo 54 — Centro Universitário SENAC**

- **Status:** **implemented MVP** of the architecture **proposal**. The layered path, the dependency rules, and the offline ML trail are reflected in the code of `src/credit_card_fraud/`, the tests, and the generated reports.
- **Object:** describe the path by which the prepared dataset feeds the dashboard, without anticipating pending decisions.
- **Canonical reference:** this is the reference document of the layered design. The summary is in Section 6.2.1 of [`PROJECT_PLAN.md`](PROJECT_PLAN.md); the operational context is in [`../AGENTS.md`](../AGENTS.md).
- **Language:** the dashboard interface is in **Brazilian Portuguese (pt-BR)**; documentation, identifiers, and code comments remain in English, per **D-26** (revoking D-09/D-17).

> **Update:** **P-01** was resolved by **D-10** (base MLG/ULB) and **P-04** by **D-15** (Streamlit). The design remains valid and was implemented; the synthetic base is not used. See [`DECISIONS.md`](DECISIONS.md).

## 1. Canonical flow

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

The flow represents the dashboard's **service path**. `Prepared Dataset (CSV)` is the treated artifact and documented at step E3, usually in `data/processed/`; column names, units, currency, and time axis depend on the base chosen at P-01 and are not fixed here. The flow also does not assume any interface API: the tool is decided at P-04.

## 2. Layers

| Layer | Responsibility | Mapped repository path | Allowed downstream | Should not |
|---|---|---|---|---|
| **Presentation Layer** | Render pages, filters, KPIs, charts, and tables, and convert user interactions into requests for the Application Layer. | `src/credit_card_fraud/presentation/` — implemented; absorbs the previous proposal's `viz/` and `dashboard/`. | `Application Layer` | Not read CSVs; not access `Data Access` directly; not compute metrics or aggregations; not contain analytical rules; not depend on `models/` or on an ML library. |
| **Application Layer** | Orchestrate filters, expose dashboard services, and coordinate session state; transform interactions into parameters for Analytics/Domain. | `src/credit_card_fraud/application/` — implemented. | `Analytics / Domain` | Not compute metrics or aggregations; not load, validate, or cache data; not implement charts; not duplicate analytical rules; not manipulate files. |
| **Analytics / Domain** | Be the single source of truth for metrics, aggregations, and analytical rules, answering for the same definitions in notebooks and in the dashboard. | `src/credit_card_fraud/analytics/` — implemented. | `Data Access` | Not read files directly; not implement UI or session state; not train models; not duplicate calculation between notebook and dashboard; not depend on a dashboard framework. |
| **Data Access** | Load the prepared dataset, validate the expected schema, and manage the cache, isolating input and output details. | `src/credit_card_fraud/data/` — implemented with loading, schema validation, cache, and audit. | `Prepared Dataset (CSV)` | Not compute business indicators; not create UI filters; not decide analytical rules; not edit `data/raw/`; not implement another base's rules. |
| **Prepared Dataset (CSV)** | Immutable flow entry: treated base, with provenance and dictionary recorded. | `data/processed/` — existing convention, not versioned. | None. | Not be read directly by Presentation, Application, or Analytics; not be modified by the reading path; not contain logic or business rules. |

## 3. Mandatory rules

1. Dependencies flow **only downwards**: `Presentation → Application → Analytics/Domain → Data Access → Prepared Dataset`. No layer depends on an upper layer; data returns upwards, but calls do not invert.
2. **Presentation never reads CSVs** and **never accesses Data Access directly**. All data input goes through `Application → Analytics/Domain → Data Access`.
3. **Data Access** is the only responsible for `loading`, `schema validation`, and `caching`. The validated schema must be parameterizable according to the base chosen at P-01.
4. **Application** is the only responsible for `filter orchestration`, `dashboard services`, and `session state coordination`.
5. **Analytics/Domain** is the single source of truth for `metrics`, `aggregations`, and `analytical rules`. Notebooks and the dashboard must consume the same functions or artifacts, so that the numbers remain consistent (RNF-08 and risk R-16 of the plan).
6. No analytical rule may be duplicated in another layer: every number displayed in the dashboard must originate in Analytics/Domain and be traceable to the prepared dataset.
7. The design does not fix the UI library, the session-state mechanism, or the cache format; these choices depend on P-04 and later decisions.

## 4. Where Machine Learning fits

The 5-layer diagram is the dashboard's **service path**. It does not include a "ML" layer at runtime, nor should it gain a sixth layer with that name.

Model training and evaluation (steps **E5–E7**) form an **offline trail**, kept in `features/` and `models/`. This trail produces artifacts — a fitted model, feature pipeline, and evaluation metrics — that are **consumed by the Analytics/Domain layer**, publishing them to the rest of the flow as metrics and analytical rules. The prediction tab, if approved at P-05, must reuse the same transformation pipeline as training, without a parallel reimplementation.

Thus the offline trail feeds the architecture, but does not change the direction of the service path's dependencies.

## 5. Proposed mapping to folders

The mapping below refers to the proposal of Section 6.3 of `PROJECT_PLAN.md` and was implemented in the MVP. The original scaffold (`src/credit_card_fraud/main.py`, tests, and Ruff/pytest settings) continues to work.

| Layer / trail | Proposed folder | Status relative to the previous proposal (6.3) |
|---|---|---|
| **Prepared Dataset** | `data/processed/` | Already planned as the non-versioned data area; remains the flow entry. |
| **Data Access** | `src/credit_card_fraud/data/` | Already proposed as "loading and validation"; explicitly adds schema validation and cache. |
| **Analytics / Domain** | `src/credit_card_fraud/analytics/` | **New proposed directory** centralizing metrics, aggregations, and analytical rules. |
| **Application** | `src/credit_card_fraud/application/` | **New proposed directory** for dashboard filters, services, and session state. |
| **Presentation** | `src/credit_card_fraud/presentation/` | **Proposed change:** absorbs the previous proposal's `viz/` (graphs) and `dashboard/` (application) directories. |
| **ML offline trail** | `src/credit_card_fraud/features/` and `src/credit_card_fraud/models/` | Kept with their previous roles (transformations, training, evaluation). `features/` is **not** renamed and not absorbed by Presentation. |
| **Support** | `src/credit_card_fraud/config.py` | Kept for paths, seeds, and constants, without introducing circular inter-layer dependencies. |

**Implemented structure:** `analytics/` and `application/` were created, and `viz/` and `dashboard/` were absorbed by `presentation/`. The implementation follows the Section 3 dependency rules and is covered by tests.

## 6. Language and pending items

### 6.1 Dashboard interface language

After the **D-26** decision, all text aimed at the user of the dashboard is in **Brazilian Portuguese (pt-BR)**: KPIs and filter labels, graph titles/axes/legends, table headers, navigation, messages, and empty or error states. The earlier interface-in-English rule (D-09/D-17) is revoked. Documentation, identifiers, and code comments are in English, following the language standardization applied to the codebase and the project documentation. The rule applies to interface labels, not raw base values from the chosen base.

### 6.2 Pending items limiting the design

**P-01** and **P-04** were resolved by D-10 (MLG/ULB) and D-15 (Streamlit). The base's fixed schema is in `src/credit_card_fraud/data/schema.py`, the relative time unit is documented in `docs/DATA_DICTIONARY.md`, and Presentation components follow Application's services.
