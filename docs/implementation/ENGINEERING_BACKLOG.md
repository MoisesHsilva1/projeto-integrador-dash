# Engineering Backlog — Projeto Integrador Fraude em Cartão de Crédito

## Working Context

This backlog serves as the engineering roadmap for improving the existing academic project on credit card fraud detection. It addresses architecture simplification, ML pipeline optimization, code quality, and dashboard integration. All decisions must have technical justification supported by the actual project requirements and experimental evidence.

The current project state (as of 18/11/2026):
- **Base:** MLG/ULB (`creditcard.csv`) — 284,807 transactions, 492 frauds (0.1727%)
- **Architecture:** 5-layer flow (Data Access → Analytics/Domain → Application → Presentation) implemented in `src/projeto_integrador/`
- **ML Pipeline:** Logistic Regression + Random Forest + Gradient Boosting with `class_weight="balanced"`, AUPRC optimization, stratified 80/20 split, semente 42
- **Dashboard:** Streamlit with 4 mandatory visualizations + filters
- **Tests:** 41 tests covering data, features, models, reports, filters, schema, and service
- **Result:** AUPRC 0.7992, F1 0.7956, recall 0.7579, precision 0.8372 (Random Forest holdout)

### Important Language Constraint (D-26)

Conforme a atualização mais recente nos materiais: a interface do dashboard deve estar em **português brasileiro**, não em inglês. Documentação, identificadores e comentários de código permanecem em português. A interface do dashboard (KPIs, filtros, gráficos, tabelas, navegação, mensagens e estados vazios/de erro) deve estar em português brasileiro conforme D-26. Não traduzir valores brutos da base.

---

# Phase 00: Discovery and Baseline

## ARCH-001: Repository Audit and Baseline Establishment

**Priority:** P0  
**Complexity:** M  
**Dependencies:** None

**Objective:**
Perform a complete audit of the repository to establish a baseline for all subsequent improvements.

**Implementation:**
1. Map all Python modules in `src/projeto_integrador/`
2. Document data flow from raw CSV to dashboard
3. Profile the dataset characteristics (distribution, duplicates, missing values, etc.)
4. Record existing ML pipeline: algorithms, preprocessing, training, evaluation
5. Inventory dashboard components and their data sources
6. List all existing tests and their coverage
7. Document current configuration and constants

**Expected Files Affected:**
- `docs/implementation/ENGINEERING_BACKLOG.md` (this file will record the audit findings)

**Acceptance Criteria:**
- Complete inventory of all source files and their responsibilities
- Dataset profile documented (class distribution, duplicates, value ranges, etc.)
- ML pipeline description: algorithms, preprocessing, training methodology, evaluation metrics
- Dashboard component inventory with data sources
- Test coverage map
- Configuration inventory

**Validation Commands:**
```bash
uv run pytest tests/ --tb=short  # verify all 41 tests pass
uv run projeto-integrador-pipeline tudo  # run full pipeline
uv run projeto-integrador-dashboard  # verify dashboard starts
```

---

## ARCH-002: Dataset Profile Documentation

**Priority:** P0  
**Complexity:** S  
**Dependencies:** ARCH-001

**Objective:**
Document the complete dataset profile for the MLG/ULB base, including class distribution, duplicate analysis, value ranges, and quality metrics.

**Implementation:**
1. Document total transactions: 284,807
2. Document class distribution: 492 frauds (0.1727%), 284,315 legitimates (99.8273%)
3. Document duplicate analysis: 1,081 exact duplicates, 19 are frauds
4. Document Amount statistics: total 25,162,590.01, min 0, max 25,691.16, 1,825 zero-value transactions (27 are frauds)
5. Document Time range: 0 to 172,792 seconds (~2 days)
6. Document V1-V28: PCA components, no business meaning disclosed
7. Record all findings in the backlog

**Expected Files Affected:**
- `docs/implementation/ENGINEERING_BACKLOG.md` — add dataset profile section

**Acceptance Criteria:**
- Dataset profile accurately reflects the MLG/ULB base
- All key statistics documented and verified against raw data

**Validation Commands:**
```bash
python -c "
from projeto_integrador.data.loader import carregar_brutas
from projeto_integrador.data.auditoria import auditar
df = carregar_brutas()
aud = auditar(df)
print(f'Linhas: {aud[\"linhas\"]}')
print(f'Fraudes: {aud[\"fraudes\"]}')
print(f'Taxa de fraude: {aud[\"taxa_fraude\"]}')
print(f'Duplicatas: {aud[\"duplicatas_exatas\"]}')
print(f'Amount total: {aud[\"valor_total\"]}')
print(f'Time range: {aud[\"inicio_segundos\"]} a {aud[\"fim_segundos\"]} seg')
"
```
