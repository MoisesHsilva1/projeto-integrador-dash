# Área de dados

Esta pasta está reservada para você inserir as bases do projeto. **Os arquivos de dados não devem ser versionados no Git.** Os diretórios e este README são apenas a estrutura de apoio.

## Estrutura

```text
data/
├── README.md
├── raw/          # arquivos originais, sem edição
├── interim/      # dados intermediários de tratamento
└── processed/    # dados prontos para análise/modelagem
```

## Onde colocar cada base

A base definitiva ainda é uma decisão pendente do grupo (P-01/P-02 no `docs/PROJECT_PLAN.md`). Coloque **apenas a base escolhida**; não misture as três no mesmo fluxo.

| Base | Arquivo esperado em `data/raw/` | Link | Observações |
|---|---|---|---|
| **MLG/ULB — Credit Card Fraud Detection** (recomendada) | `creditcard.csv` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | 284.807 linhas, 31 colunas, 492 fraudes; cerca de 150 MB; dados anonimizados por PCA. |
| **Sintética 10K — Credit Card Fraud Detection Dataset** | `credit_card_fraud_10k.csv` | https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset | 10.000 linhas, 10 colunas, 151 fraudes no arquivo; base **sintética**. |
| **Give Me Some Credit** | `cs-training.csv` e, se necessário, `cs-test.csv` | https://www.kaggle.com/c/GiveMeSomeCredit | Base de inadimplência, **não recomendada** para o problema de fraude. |

Se o arquivo baixado tiver outro nome, mantenha o nome original ou renomeie de forma explícita e registre a alteração no dicionário de dados.

## Depois de inserir

1. Confira as dimensões e a distribuição da variável-alvo.
2. Registre URL, data de acesso, versão, licença e checksum em `docs/DATA_DICTIONARY.md` (a ser criado na etapa E2).
3. Não edite o arquivo em `data/raw/`; todo tratamento deve gerar uma cópia em `data/interim/` ou `data/processed/`.
4. Não faça commit dos arquivos de dados.

## O que não fazer

- Não baixar ou tratar os dados automaticamente sem uma decisão registrada.
- Não usar os números do mockup do dashboard como resultados.
- Não misturar a base real com a sintética.
- Não apagar os arquivos originais depois do tratamento.

Consulte as Seções 4 e 7 do `docs/PROJECT_PLAN.md` para os cuidados de tratamento e a sequência das etapas.
