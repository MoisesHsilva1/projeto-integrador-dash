# Registro de decisões

**Grupo 54 — Centro Universitário SENAC**

Este documento complementa a Seção 10 de [`PROJECT_PLAN.md`](PROJECT_PLAN.md).
Ele registra as decisões tomadas para a implementação do pipeline e do
dashboard, com a justificativa e o impacto de cada uma. As decisões **D-01 a
D-09** estão no plano; as decisões abaixo (D-10 em diante) foram tomadas para
esta implementação.

## Base de dados

| ID | Decisão | Justificativa | Impacto |
|---|---|---|---|
| **D-10** | Adotar a base **MLG/ULB** (`data/raw/creditcard.csv`) como base única do projeto. | É a base citada na apresentação, compatível com o esboço e com dado real anonimizado. O arquivo foi fornecido pelo grupo em `data/raw/`. Resolve P-01/P-02. | Todo o pipeline usa `creditcard.csv`; a base sintética e a *Give Me Some Credit* não são usadas. |
| **D-11** | Remover **duplicatas exatas**, mantendo a primeira ocorrência, antes da modelagem. | Duplicatas entre treino e teste inflam métricas; manter a mesma transação duas vezes cria vazamento. O impacto foi medido e documentado. | 1.081 linhas removidas, incluindo 19 fraudes: a base passa de 284.807/492 para 283.726/473 (0,17%). |
| **D-12** | Não remover outliers nem transações com `Amount = 0`. | Fraudes podem estar nos extremos; zero é um valor válido na base. A limpeza não deve apagar sinal. | Outliers e valores zero permanecem; a EDA os descreve. |

## Tempo e moeda

| ID | Decisão | Justificativa | Impacto |
|---|---|---|---|
| **D-13** | Analisar o tempo em **blocos relativos**: hora decorrida, dia relativo (1 e 2) e hora dentro do bloco de 24 h. | `Time` é relativo à primeira transação; a origem não é meia-noite documentada. Resolve P-09. | Gráficos e heatmap usam “relative hours/days”; nenhuma afirmação sobre hora real ou sazonalidade. |
| **D-14** | Exibir valores na **unidade original da base**, sem símbolo de moeda e sem conversão. | O dicionário não fixa “euros” nem “R$”; converter sem fonte seria inventar dado. Resolve P-10. | Rótulos “Amount (original units)”; nenhum gráfico com `R$`. |

## Produto e interface

| ID | Decisão | Justificativa | Impacto |
|---|---|---|---|
| **D-15** | Implementar o dashboard em **Streamlit**. | Menor complexidade para um dashboard acadêmico em Python (R-07). Resolve P-04. | Dependência `streamlit`; execução com `uv run projeto-integrador-dashboard`. |
| **D-16** | Remover o filtro “tipo de transação”. | O campo não existe na MLG/ULB; a interface informa a ausência. Resolve P-08/R-09. | Filtros disponíveis: valor, tempo decorrido e status (fraude/legítima). |
| **D-17** | Interface em inglês; documentação, código, relatórios e nomes de identificadores em português. | D-09, mantida. | Rótulos de KPIs, filtros, gráficos e estados vazios em inglês; relatórios em português. |
| **D-18** | A aba de predição individual **não faz parte do MVP**. | Controla escopo (R-08) e P-05 continua em aberto. | O dashboard é analítico; o modelo é treinado, avaliado e salvo, mas não há simulação de transação única na interface. |
| **D-26** | Interface do dashboard em **português brasileiro (pt-BR)**, revogando D-09/D-17. | Requisito explícito de refinamento do dashboard; melhora a usabilidade para o público-alvo acadêmico. | Todos os rótulos de KPIs, filtros, gráficos, tabelas, estados e mensagens em pt-BR; documentação, identificadores e comentários de código permanecem em português (sem mudança). |
| **D-27** | Uso de **Plotly** para os gráficos do dashboard. | Tooltips interativos e tema dark consistente exigem gráficos interativos; matplotlib estático não atende. | Nova dependência `plotly`; componentes em `presentation/componentes.py` retornam figuras Plotly. |
| **D-28** | Formatação pt-BR para números (`1.250,00`, `1,25%`), **sem** símbolo de moeda e **sem** datas de calendário. | Reafirma D-13/D-14: a base não define moeda nem data absoluta; apenas a formatação segue o padrão brasileiro. | Fonte única em `formatacao.py`; rótulos mantêm "unidade original" e "horas/dias relativos". |
| **D-29** | Inclusão de **explorador de transações** paginado e de **indicadores offline do modelo**. | Aproveita `amostra_aleatoria`/`recorte` existentes e `reports/model/metricas.json`; sem nova modelagem. | Tabela filtrável com as colunas disponíveis (sem ID/categoria); seção de métricas sinalizada como estática. |

## Modelagem e avaliação

| ID | Decisão | Justificativa | Impacto |
|---|---|---|---|
| **D-19** | Comparar **Regressão Logística (baseline)**, **Random Forest** e **Gradient Boosting (histogramas)**, com `class_weight="balanced"` e sem SMOTE. | Baseline + candidatos, conforme R-06; pesos de classe evitam dependência extra e são explicáveis. | A seleção usa AUPRC média em validação cruzada estratificada de 3 dobras. |
| **D-20** | Escolher o **limite de decisão** que maximiza F1 nas probabilidades out-of-fold do **treino**, e só então avaliar no holdout. | Evita escolher o limite olhando o teste; documenta o trade-off precision/recall. | Limite registrado em `reports/model/metricas.json` e usado no relatório. |
| **D-21** | Reportar **AUPRC** como métrica principal, acompanhada de ROC-AUC, precision, recall, F1, acurácia (contexto) e matriz de confusão. | Recomendação oficial da base (R-05) e adequação ao desbalanceamento extremo. | Nenhuma meta numérica arbitrária; o resultado é o obtido pelo pipeline. |
| **D-22** | Separar treino/teste (80/20, estratificado, semente 42) **antes** de qualquer ajuste; padronizar apenas dentro do pipeline. | Evita vazamento (R-04). | `StandardScaler` é ajustado somente no treino; as árvores não exigem escala, mas o pipeline é uniforme. |

## Organização técnica

| ID | Decisão | Justificativa | Impacto |
|---|---|---|---|
| **D-23** | Base preparada em **Parquet** (`data/processed/transacoes.parquet`), não versionada. | Leitura mais rápida que CSV e mantém os dados fora do Git. | O dashboard lê apenas o Parquet via Data Access, com cache. |
| **D-24** | Seguir o fluxo em camadas de `docs/ARCHITECTURE.md`. | Fonte única de métricas e separação de responsabilidades (RNF-08/RNF-09). | `data/` → `analytics/` → `application/` → `presentation/`; ML em `features/` e `models/`. |
| **D-25** | Salvar figuras e relatórios em `reports/` e o binário do modelo em `reports/model/`. | Rastreabilidade acadêmica; o binário é ignorado pelo Git por ser artefato. | `reports/model/modelo_final.joblib` não é versionado; métricas e figuras são. |

## Resultado obtido nesta implementação

O pipeline foi executado em 18/11/2026 com semente 42 e base preparada
(283.726 transações, 473 fraudes):

- modelo selecionado: **Random Forest** (maior AUPRC média na validação cruzada);
- AUPRC no holdout: **0,7992**;
- F1, recall e precision no holdout: **0,7956 / 0,7579 / 0,8372**;
- matriz de confusão: 56.637 verdadeiros negativos, 14 falsos positivos,
  23 falsos negativos e 72 verdadeiros positivos.

Esses valores são resultados do pipeline desta base, não metas nem números do
mockup. Podem variar se a base, a semente ou as versões das bibliotecas mudarem.
