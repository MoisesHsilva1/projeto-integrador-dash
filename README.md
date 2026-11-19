# projeto-integrador

Detecção de fraudes em transações com cartão de crédito — Projeto Integrador do
Grupo 54 (Centro Universitário SENAC). O projeto combina análise exploratória,
classificação supervisionada e um dashboard interativo sobre a base **MLG/ULB**
(*Credit Card Fraud Detection*).

## Começar

Requisitos: Python ≥ 3.11 e [uv](https://docs.astral.sh/uv/).

```bash
cd ~/Documents/projects/projeto-integrador
uv venv
uv pip install -e ".[dev]"
code .
```

Abra o projeto com o perfil **Python** do VS Code. O interpretador
`.venv/bin/python` e os testes pytest já vêm configurados em
`.vscode/settings.json`.

Coloque a base em `data/raw/creditcard.csv` (o arquivo não é versionado). Veja
[`data/README.md`](data/README.md) e o dicionário em
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Pipeline

O fluxo segue as etapas do plano (`docs/PROJECT_PLAN.md`) e da aula 2:

```bash
uv run projeto-integrador-pipeline preparar   # E2/E3: audita, limpa e grava o Parquet
uv run projeto-integrador-pipeline eda        # E4: figuras e relatório da EDA
uv run projeto-integrador-pipeline treinar    # E5–E7: modelos, avaliação e artefatos
uv run projeto-integrador-pipeline tudo       # executa as três etapas em sequência
```

Cada comando é reprodutível e usa a semente `42` (`src/projeto_integrador/config.py`).
O treino compara Regressão Logística (baseline), Random Forest e Gradient
Boosting, seleciona pela AUPRC média em validação cruzada e avalia o holdout com
o limite de decisão escolhido no treino.

## Dashboard

```bash
uv run projeto-integrador-dashboard
# equivalente: uv run streamlit run src/projeto_integrador/presentation/app.py
```

O dashboard abre no navegador com as quatro visualizações obrigatórias —
indicadores gerais, distribuição de fraudes, fraudes por faixa de valor e
evolução temporal — além do mapa de calor, das correlações, do explorador de
transações e dos indicadores offline do modelo. Os filtros são de valor, tempo
decorrido e classificação (fraude/legítima). A interface é em português
brasileiro, conforme a decisão D-26.

> O dashboard precisa da base preparada. Se ela não existir, execute
> `uv run projeto-integrador-pipeline preparar` antes.

## Artefatos gerados

| Artefato | Caminho |
|---|---|
| Base tratada (não versionada) | `data/processed/transacoes.parquet` |
| Relatório de qualidade | `reports/RELATORIO_QUALIDADE.md` e `reports/qualidade.json` |
| Relatório de EDA | `reports/RELATORIO_EDA.md` e `reports/figures/eda_*.png` |
| Métricas do modelo | `reports/model/metricas.json` |
| Modelo treinado (não versionado) | `reports/model/modelo_final.joblib` |
| Relatório de avaliação | `reports/RELATORIO_MODELO.md` e `reports/figures/modelo_*.png` |

## Testes e qualidade

```bash
uv run pytest              # suíte completa
uv run pytest tests/test_main.py::test_main
uv run ruff check --fix .  # lint + autofix
uv run ruff format .       # formatação
```

O teste de fumaça do dashboard é ignorado automaticamente se a base preparada
ainda não existir.

## Estrutura

```text
src/projeto_integrador/
├── config.py            caminhos, sementes e constantes
├── data/                Data Access: carga, validação de esquema, cache, auditoria
├── analytics/           métricas, agregações e regras analíticas
├── application/         filtros e serviços do dashboard
├── features/            atributos e split treino/teste (trilha offline)
├── models/              treino, limite, avaliação e persistência (trilha offline)
├── presentation/        dashboard Streamlit e componentes de gráfico
├── relatorios/          geração dos relatórios e figuras
└── cli.py               linha de comando do pipeline
tests/                   testes pytest
data/                    bases de dados (não versionadas)
docs/                    plano, arquitetura, dicionário e decisões
reports/                 figuras, métricas e relatórios gerados
```

As regras de dependência entre camadas estão em
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Documentação e código ficam em
português; a interface do dashboard fica em inglês.

## Limitações da base

- Dois dias de transações europeias (setembro de 2013), com tempo relativo.
- Atributos `V1`–`V28` anonimizados por PCA: não é possível nomear fatores de
  negócio.
- Moeda não fixada no dicionário; os valores são exibidos na unidade original.
- Desbalanceamento extremo: 473 fraudes em 283.726 transações após remover as
  duplicatas exatas.
- O projeto é uma ferramenta acadêmica de apoio à decisão, não um sistema
  antifraude de produção com bloqueio automático.

## Documentação

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — plano completo, requisitos e pendências.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — fluxo em camadas.
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — proveniência, colunas e tratamento.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — decisões tomadas na implementação.
- [`AGENTS.md`](AGENTS.md) — contexto operacional resumido.
