# Dicionário de dados — MLG/ULB *Credit Card Fraud Detection*

**Projeto:** Projeto Integrador do Grupo 54 — Centro Universitário SENAC
**Base:** Credit Card Fraud Detection Dataset (Worldline e Machine Learning Group — Université Libre de Bruxelles)
**Arquivo bruto:** `data/raw/creditcard.csv` (não versionado)
**Arquivo preparado:** `data/processed/transacoes.parquet` (não versionado, gerado pelo pipeline)

## 1. Proveniência

| Campo | Valor |
|---|---|
| Fonte oficial | Kaggle — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Autores | Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi |
| Data de acesso registrada no plano | 17/11/2026 |
| Data de geração desta base preparada | 18/11/2026 |
| Arquivo | `creditcard.csv` (150.828.752 bytes) |
| MD5 do arquivo bruto | `e90efcb83d69faf99fcab8b0255024de` |
| Licença declarada na página da base | Open Database License (ODbL) — confirmar na página oficial antes da entrega |
| Dados pessoais | Não há identificadores diretos; os atributos são anonimizados por PCA |

O arquivo bruto é imutável e está protegido pelo `.gitignore`. Toda transformação
gera uma cópia em `data/processed/`, nunca uma edição no original.

## 2. Estrutura do arquivo bruto

| Item | Valor |
|---|---|
| Registros | 284.807 |
| Colunas | 31 |
| Valores ausentes | 0 |
| Duplicatas exatas | 1.081 |
| Transações fraudulentas | 492 (0,1727%) |
| Transações legítimas | 284.315 |
| Período coberto | 172.792 segundos (~2 dias) a partir da primeira transação |
| `Amount` total | 25.162.590,01 (unidade original) |
| `Amount` mínimo / máximo | 0,00 / 25.691,16 |
| Transações com `Amount` = 0 | 1.825 (27 fraudes) |

## 3. Colunas

| Coluna | Tipo | Descrição | Observações |
|---|---|---|---|
| `Time` | float | Segundos desde a primeira transação da base | Tempo relativo; não é data/hora absoluta |
| `V1`–`V28` | float | Componentes principais obtidas por PCA | Significado de negócio não divulgado; não nomear os fatores |
| `Amount` | float | Valor da transação | Unidade original da base; a moeda não é fixada no dicionário |
| `Class` | int | Rótulo: 1 = fraude, 0 = legítima | Variável-alvo |

**Colunas derivadas na análise (não gravadas na base preparada):**

| Coluna derivada | Origem | Definição |
|---|---|---|
| `hora_decorrida` | `Time` | `floor(Time / 3600)` |
| `dia_relativo` | `Time` | `floor(Time / 86400) + 1` (dias 1 e 2) |
| `hora_do_dia_relativo` | `Time` | hora dentro do bloco relativo de 24 h; **não** é hora de relógio |
| `faixa_valor` | `Amount` | faixas `0-10`, `10-50`, `50-100`, `100-250`, `250-500`, `500-1,000`, `1,000+` (intervalo fechado à esquerda) |

## 4. Tratamento aplicado

- Duplicatas exatas: removidas (mantida a primeira ocorrência). Impacto: 1.081
  linhas e 19 fraudes a menos; a base preparada fica com **283.726 linhas e 473
  fraudes (0,17%)**. A decisão está registrada em `docs/DECISIONS.md` (D-11).
- Valores zero de `Amount`: mantidos; não são inválidos na base.
- Outliers de `Amount`: mantidos; fraudes podem estar nos extremos.
- Nulos: não existem no arquivo; o pipeline mantém a validação de esquema.
- Nenhuma coluna foi renomeada, descartada ou reescalada na base preparada.

## 5. Limitações obrigatórias de leitura

- A base cobre **dois dias** de transações europeias de setembro de 2013.
- `Time` é relativo; não há data absoluta nem hora do dia confirmada.
- `V1`–`V28` são componentes PCA anonimizados: não é possível nomear os fatores
  de negócio por trás deles.
- A moeda não está definida no dicionário; o dashboard usa a **unidade original**,
  sem conversão e sem o símbolo `R$`.
- Não há identificador de cartão, comerciante, categoria, dispositivo ou tipo de
  transação. O filtro “tipo de transação” do mockup não é viável nesta base.
- Poucos positivos (473 após a limpeza) tornam as métricas sensíveis ao split.

## 6. Reprodutibilidade

```bash
uv run projeto-integrador-pipeline preparar   # audita, limpa e grava o Parquet
uv run projeto-integrador-pipeline eda        # figuras e relatório da EDA
uv run projeto-integrador-pipeline treinar    # modelo, métricas e relatório
```

Saídas:

| Artefato | Caminho |
|---|---|
| Base preparada | `data/processed/transacoes.parquet` |
| Auditoria e limpeza | `reports/RELATORIO_QUALIDADE.md`, `reports/qualidade.json` |
| EDA | `reports/RELATORIO_EDA.md`, `reports/figures/eda_*.png` |
| Modelo e métricas | `reports/model/modelo_final.joblib`, `reports/model/metricas.json` |
| Avaliação | `reports/RELATORIO_MODELO.md`, `reports/figures/modelo_*.png` |

Sementes: `SEMENTE = 42` (`src/projeto_integrador/config.py`).
