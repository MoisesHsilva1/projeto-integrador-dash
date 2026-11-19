# Relatório de qualidade dos dados

**Base:** MLG/ULB — *Credit Card Fraud Detection* (`data/raw/creditcard.csv`)

**MD5 do arquivo bruto:** `e90efcb83d69faf99fcab8b0255024de`

## 1. Dimensões e alvo

| Verificação | Resultado |
|---|---|
| Linhas | 284.807 |
| Colunas | 31 |
| Valores ausentes | 0 |
| Transações legítimas | 284.315 |
| Transações fraudulentas | 492 |
| Taxa de fraude | 0,17% |

## 2. Qualidade

| Verificação | Resultado |
|---|---|
| Duplicatas exatas | 1.081 |
| Fraudes entre as duplicatas | 19 |
| `Time` mínimo / máximo (s) | 0,00 / 172.792,00 |
| Duração coberta | 48,0 horas |
| `Amount` mínimo / máximo | 0,00 / 25.691,16 |
| `Amount` mediano | 22,00 |
| `Amount` total | 25.162.590,01 (unidade original) |
| Transações com `Amount` = 0 | 1.825 (fraudes: 27) |

## 3. Tratamento aplicado

| Regra | Linhas removidas | Fraudes removidas |
|---|---|---|
| Remoção de duplicatas exatas (mantida a primeira ocorrência) | 1.081 | 19 |

**Resultado após a limpeza:** 283.726 linhas,
473 fraudes
(0,17%).

**Itens mantidos deliberadamente:**

- Transações com Amount = 0 (não são valores inválidos na base).
- Outliers de Amount: não removidos; fraudes podem estar nos extremos.
- Atributos V1–V28: mantidos como componentes PCA anonimizados.

A base tratada é gravada em `data/processed/transacoes.parquet` e não substitui o
arquivo bruto em `data/raw/`.

## 4. Limitações

- A base cobre apenas dois dias de transações de cartões europeus (setembro de 2013).
- `Time` é relativo à primeira transação; não há data/hora absoluta.
- `V1`–`V28` são componentes PCA anonimizados, sem significado de negócio divulgado.
- A moeda não está fixada no dicionário; os valores são exibidos na unidade original.
- A remoção de duplicatas também remove fraudes; por isso o número de positivos
  cai em relação ao arquivo bruto. O impacto está registrado acima.
