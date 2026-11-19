# Plano do Projeto Integrador — Detecção de Fraudes em Transações com Cartão de Crédito

**Grupo 54 — Centro Universitário SENAC**

- **Integrantes:** Alex Oliveira Veras, Jhonatan da Silva Ribeiro, Magno Henrique Souza Cardoso, Moises Henrique da Silva e Taynan Nunes de Araujo.
- **Tema:** detecção de fraudes em transações financeiras realizadas com cartão de crédito.
- **Status deste documento:** planejamento consolidado e implementação iniciada. O MVP (E2–E8) foi implementado sobre a base MLG/ULB; as decisões da implementação estão em [`DECISIONS.md`](DECISIONS.md) e os resultados vêm do pipeline executado em 18/11/2026.
- **Última atualização:** 18/11/2026.
- **Documento complementar:** [`AGENTS.md`](../AGENTS.md), com o contexto operacional resumido para o agente.

> **Regra de leitura:** este plano separa explicitamente o que foi **identificado nos materiais**, o que foi **decidido pelo grupo**, o que é **sugestão técnica** e o que ainda é **pendência**. Não trate sugestões como requisitos aprovados. Consulte a Seção 10 antes de tomar decisões de implementação.

---

## 1. Visão geral

### 1.1 Propósito

Este documento organiza o problema, os requisitos, as bases de dados disponíveis, a fundamentação acadêmica, a arquitetura proposta e a sequência de desenvolvimento do Projeto Integrador do Grupo 54.

Ele deve servir como:

- referência única de escopo durante o desenvolvimento;
- roadmap para acompanhar o avanço de cada etapa;
- registro das decisões já tomadas e das questões ainda abertas;
- base para o relatório e para a apresentação final.

### 1.2 Resumo executivo

O grupo pretende construir um **dashboard interativo** para analisar transações de cartão de crédito e aplicar **Machine Learning** para identificar padrões associados a fraudes. A proposta está registrada na apresentação do grupo e é compatível, em linhas gerais, com o ciclo de vida de ciência de dados apresentado na aula 2.

A principal questão técnica em aberto é a **base de dados**. A apresentação cita a base clássica *Credit Card Fraud Detection* (MLG/ULB), com 284.807 transações e 492 fraudes, e o esboço do dashboard usa números compatíveis com ela. Porém, o link de Kaggle fornecido como “Base 2” aponta para um notebook que utiliza uma base **sintética** diferente, com 10.000 transações e 151 fraudes. Essa divergência precisa ser resolvida pelo grupo antes do desenvolvimento.

**Recomendação preliminar:** adotar a base clássica MLG/ULB como base principal, por ser a base citada na apresentação, a que sustenta os números do esboço e a que tem maior credibilidade como dado real anonimizado. A base sintética deve ser tratada como alternativa documentada, não como complemento silencioso. A base *Give Me Some Credit* não é adequada ao problema de fraude e deve ser descartada para este escopo.

### 1.3 Legenda de status

| Status | Significado |
|---|---|
| **Explícito** | Consta diretamente em um material fornecido. |
| **Esboço** | Aparece no mockup do dashboard, mas não foi formalizado como requisito. |
| **Recomendado** | Boa prática ou sugestão técnica desta análise; não é exigência acadêmica documentada. |
| **Pendente** | Depende de decisão do grupo, do professor ou de validação na fonte original. |

### 1.4 Fontes utilizadas

| Código | Fonte | O que foi usado |
|---|---|---|
| **AULA2** | `aula2.pdf` — webconferência “Introdução à Ciência de Dados”, 27 páginas | Ciclo de vida, pré-processamento, EDA, construção de modelos, tipos de aprendizado. |
| **COD** | `Codigos_Aula2.zip` — `Aula2.ipynb`, `Extra_Aula2.ipynb`, `books.csv`, `dados_populacionais_1000_linhas.csv` | Técnicas de pandas, limpeza, agregação, EDA e visualização com matplotlib/seaborn. |
| **APRES** | `grupo-54.pptx.pdf` — apresentação do Grupo 54, 12 páginas | Problema, objetivos, escopo, próximos passos, esboço do dashboard e referências. |
| **GMSC** | Kaggle — *Give Me Some Credit* | Estrutura, variáveis, problema de negócio e limitações. |
| **CCFD-ULB** | Kaggle — *Credit Card Fraud Detection* (MLG/ULB) | Base clássica de fraude; estrutura, variáveis, distribuição e recomendações de avaliação. |
| **CCFD-10K** | Kaggle — *Credit Card Fraud Detection Dataset* (`miadul/...`), usado pelo notebook do link fornecido | Base sintética de 10.000 transações; estrutura, variáveis e limitações. |
| **NILSON** | Nilson Report, edição 1298 / comunicado de janeiro de 2026 | Fonte do dado de perdas globais com fraude de cartões em 2024 (US$ 33,41 bilhões). |

**Observação sobre rastreabilidade:** a apresentação cita o dado de US$ 33,41 bilhões sem indicar a fonte. A referência foi localizada no Nilson Report e deve ser incluída no relatório final. Já o link de Kaggle fornecido para a “Base 2” não aponta para a página da base MLG/ULB; ele aponta para um notebook que consome a base sintética `miadul/credit-card-fraud-detection-dataset`. Essa diferença é tratada na Seção 4.

**Observação sobre prazos:** o PDF da AULA2 termina com um slide intitulado “CALENDÁRIO”, mas não foi possível extrair dele datas ou conteúdo legível. Nenhum prazo foi assumido neste plano; o cronograma acadêmico precisa ser confirmado com o professor.

---

## 2. Problema e objetivos

### 2.1 Contexto de negócio

A apresentação situa o problema no crescimento das perdas globais causadas por fraudes em cartões. O dado citado é de **US$ 33,41 bilhões em 2024** (NILSON). O grupo argumenta que a análise de dados pode:

- reduzir prejuízos ao identificar possíveis fraudes;
- tornar a detecção mais eficiente com apoio de Machine Learning;
- revelar características e comportamentos associados a transações fraudulentas;
- apoiar a análise e a tomada de decisão por meio de um dashboard.

O produto pretendido é, portanto, uma **ferramenta analítica de apoio à decisão**, e não um sistema antifraude de produção com bloqueio automático de transações. Essa distinção é importante para controlar o escopo.

### 2.2 Problema

> Identificar quais transações com cartão de crédito podem se encaixar em casos de fraude, diferenciando-as das transações legítimas e buscando padrões que indiquem possíveis operações fraudulentas.

Fonte: APRES, p. 3.

Em termos técnicos, o problema é uma **classificação binária supervisionada**: prever se uma transação é fraudulenta (`1`) ou legítima (`0`), a partir de atributos históricos rotulados. A base MLG/ULB já possui o rótulo `Class`; a base sintética possui `is_fraud`.

### 2.3 Objetivo geral

Criar um **dashboard interativo** que permita analisar dados de transações com cartão de crédito, compreender suas principais características e comportamentos relacionados a possíveis fraudes, auxiliando na análise preditiva e eficiente, de modo a identificar padrões e características que contribuam para a detecção de fraudes.

Fonte: APRES, p. 4 (paráfrase fiel).

### 2.4 Objetivos específicos

| ID | Objetivo | O que a apresentação indica | Status |
|---|---|---|---|
| **OE-1** | Identificar padrões e características das transações fraudulentas | Observar valores, frequência, distribuição e comportamentos fora do comum; usar EDA com gráficos e estatísticas. | Explícito |
| **OE-2** | Analisar a ocorrência de fraudes de acordo com o valor das transações | Comparar fraudes por faixa de valor e distribuição dos valores. O corpo do slide repete o texto do OE-1; o título, o esboço e os próximos passos confirmam a intenção de analisar valor. | Explícito, com texto duplicado na apresentação |
| **OE-3** | Analisar a ocorrência de fraudes ao longo do tempo | Gráficos temporais, análise por horário e mapa de calor para identificar períodos com mais ocorrências. | Explícito |

**Pendência de redação:** o texto do OE-2 na apresentação está duplicado com o do OE-1. Antes da entrega, o grupo deve corrigir a redação para descrever claramente a análise por valor. Isso não muda o escopo, mas evita inconsistência no relatório.

### 2.5 Público-alvo

**Não definido explicitamente nos materiais.** A apresentação fala em “auxiliar na análise dos dados” e “facilitar a visualização das informações”, mas não nomeia o usuário do dashboard.

Hipótese de trabalho, **a confirmar**: equipes de análise de risco/fraude e gestores que precisam acompanhar indicadores, investigar padrões e apoiar decisões. Não trate essa hipótese como requisito até que o grupo a valide.

### 2.6 Escopo

**Dentro do escopo identificado:**

- obtenção e preparação de uma base de transações com cartão de crédito;
- análise exploratória dos dados;
- análise de padrões de fraude, por valor e ao longo do tempo;
- aplicação de técnica supervisionada de classificação;
- avaliação do desempenho do modelo;
- dashboard interativo com, no mínimo, quatro visualizações;
- documentação, relatório e apresentação acadêmica.

**Fora do escopo presumido (confirmar):**

- sistema antifraude em produção;
- integração com adquirentes, emissores ou gateways de pagamento;
- bloqueio automático de transações;
- scoring em tempo real ou streaming;
- uso de dados pessoais identificáveis;
- inferência causal sobre as causas da fraude;
- uso conjunto e indiscriminado de bases diferentes;
- deep learning, aprendizado por reforço ou técnicas não justificadas pelo problema.

### 2.7 Análise cruzada dos materiais

| Pergunta | Constatação | Evidência | Consequência para o plano |
|---|---|---|---|
| **1. Quais são os requisitos acadêmicos e técnicos?** | A aula apresenta o ciclo de vida (entender, coletar, pré-processar, EDA, modelar). A apresentação define o produto e cinco próximos passos. Não há enunciado, rubrica, prazos ou formato de entrega nos materiais fornecidos. | AULA2, p. 3–16; APRES, p. 9–11 | O plano segue o ciclo da aula e registra a ausência de requisitos formais como pendência. Não inventar prazos nem critérios de nota. |
| **2. Quais conceitos das aulas devem ser aplicados?** | Ciclo de vida; limpeza; amostragem; tratamento de desbalanceamento; transformação; EDA; aprendizado supervisionado (classificação/regressão); árvore de decisão; distinção entre supervisionado, não supervisionado e reforço. | AULA2, p. 3–16 | Aplicar o ciclo completo. Para fraude, usar classificação supervisionada. Não usar reforço ou não supervisionado sem justificativa. |
| **3. Quais técnicas de análise são adequadas?** | Estatísticas descritivas, histogramas, boxplots, dispersão, heatmaps, correlação; limpeza e validação de regras de negócio; agregações; treino/teste; métricas de classificação. | AULA2, p. 12–16; COD, `Aula2.ipynb` | A EDA deve responder aos três objetivos específicos. A modelagem deve comparar um baseline com pelo menos um modelo candidato. |
| **4. Quais bases fazem sentido?** | MLG/ULB é a base citada na apresentação e compatível com o esboço. A base sintética de 10 mil transações é a usada pelo link fornecido e é mais interpretável, mas não é a base citada. *Give Me Some Credit* é de inadimplência, não de fraude. | APRES, p. 9 e referências; CCFD-ULB; CCFD-10K; GMSC | Decisão pendente entre MLG/ULB e sintética. *Give Me Some Credit* não deve ser usada para o problema atual. |
| **5. Quais são as limitações e dificuldades?** | Desbalanceamento extremo; atributos anonimizados por PCA na base real; janela de apenas dois dias; ausência de tipo de transação, data absoluta e identificador de cartão na base real; duplicatas; mudança de moeda no esboço; risco de vazamento de dados; poucos positivos. | CCFD-ULB; esboço; análise local | Tratar desbalanceamento e duplicatas explicitamente; não prometer análises que a base não permite; documentar premissas de tempo e moeda. |
| **6. Quais informações faltam definir?** | Base e versão exatas; público-alvo; ferramenta do dashboard; modelo e métricas; limite de decisão; escopo da aba de predição; formato e prazos das entregas; papéis do grupo; uso ou não da base sintética. | APRES; lacunas dos materiais | Ver Seção 10. Nenhuma dessas decisões deve ser assumida silenciosamente. |
| **7. Quais entregáveis são necessários?** | Base tratada; EDA; modelo avaliado; dashboard com quatro visualizações; análise de resultados; relatório/apresentação. O formato acadêmico não está especificado. | APRES, p. 9–11 | Ver Seção 11. O código e a documentação devem permitir reproduzir os resultados. |

---

## 3. Requisitos

### 3.1 Requisitos funcionais

| ID | Requisito | Origem | Status | Observações |
|---|---|---|---|---|
| **RF-01** | Selecionar e obter uma base de transações com cartão de crédito no Kaggle. | APRES, p. 9 | Explícito, mas ambíguo | A apresentação cita a base; o link fornecido aponta para uma base sintética diferente. Resolver antes de implementar. |
| **RF-02** | Realizar limpeza, organização e preparação dos dados, incluindo a criação de indicadores relevantes para a análise de fraudes. | APRES, p. 9 | Explícito | Documentar cada transformação e cada linha removida. |
| **RF-03** | Identificar padrões e características das transações fraudulentas por meio de EDA. | APRES, p. 5 | Explícito | Usar estatísticas e visualizações; comparar fraudes e não fraudes. |
| **RF-04** | Analisar a ocorrência de fraudes por faixa de valor. | APRES, p. 5 e 10 | Explícito | Corrigir o texto duplicado do OE-2. Definir faixas e moeda. |
| **RF-05** | Analisar a ocorrência de fraudes ao longo do tempo, incluindo horários e mapa de calor. | APRES, p. 6 | Explícito | Na base MLG/ULB, o eixo temporal é relativo e cobre dois dias; ver limitação na Seção 4. |
| **RF-06** | Desenvolver dashboard interativo com, no mínimo, quatro visualizações: indicadores gerais, distribuição de fraudes, fraudes por faixa de valor e evolução temporal. | APRES, p. 10 | Explícito | É o requisito mínimo de produto. O mockup tem quatro abas, mas “quatro visualizações” não significa “quatro abas”. |
| **RF-07** | Permitir filtros de período, faixa de valor, tipo de transação e status (fraude/normal). | Mockup, p. 10 | Esboço | “Tipo de transação” não existe na base MLG/ULB. Só é viável na base sintética (`merchant_category`) ou após redefinição. |
| **RF-08** | Aplicar técnica de Machine Learning para identificar padrões suspeitos e prever possíveis fraudes. | APRES, p. 11 | Explícito | Classificação binária supervisionada. O modelo do mockup é Random Forest, mas isso é ilustrativo. |
| **RF-09** | Avaliar o desempenho do modelo e identificar os principais fatores associados às fraudes. | APRES, p. 11 | Explícito | Métricas adequadas ao desbalanceamento; interpretação compatível com os atributos disponíveis. |
| **RF-10** | Exibir indicadores gerais (transações, fraudes, taxa e valor movimentado). | Mockup, p. 10 | Esboço | Útil, mas os valores do mockup não são resultados validados. |
| **RF-11** | Disponibilizar aba de predição com probabilidade, limite de decisão, matriz de confusão e importância das variáveis. | Mockup, p. 10 | Esboço | Confirmar se entra no MVP ou se é apenas ilustração. |
| **RF-12** | Documentar o processo e permitir reprodução dos resultados. | Recomendado | Recomendado | Não está explícito nos materiais, mas é necessário para defesa acadêmica. |

### 3.2 Requisitos não funcionais

| ID | Requisito | Status | Critério inicial |
|---|---|---|---|
| **RNF-01** | Reprodutibilidade: sementes, versões e pipeline documentados. | Recomendado | Uma pessoa consegue refazer o tratamento, o treino e as principais figuras a partir do README. |
| **RNF-02** | Desempenho do dashboard: carregar e filtrar sem travar. | Recomendado | Definir meta após protótipo; pré-agregar dados se necessário. |
| **RNF-03** | Usabilidade: rótulos claros, filtros coerentes, cores consistentes. | Recomendado | Interface do dashboard em **inglês** (KPIs, filtros, gráficos, tabelas, navegação, mensagens e estados vazios/de erro); documentação e código em português (D-09). Valores brutos da base não são traduzidos. |
| **RNF-04** | Privacidade: não usar dados pessoais identificáveis. | Explícito (base anonimizada) | MLG/ULB é anonimizada; a sintética não contém dados reais. Não tentar reidentificar. |
| **RNF-05** | Portabilidade: execução local documentada. | Recomendado | Sem dependência obrigatória de nuvem; decidir se haverá deploy. |
| **RNF-06** | Qualidade de código: Ruff, pytest e tipagem quando fizer sentido. | Recomendado | Manter os comandos existentes do scaffold. |
| **RNF-07** | Integridade acadêmica: não apresentar números ilustrativos como resultados. | Recomendado | Todo número no relatório deve ter origem rastreável. |
| **RNF-08** | Consistência: dashboard, notebooks e relatório devem mostrar os mesmos números. | Recomendado | Validar com uma tabela de conferência. |
| **RNF-09** | Manutenibilidade: separar carregamento, tratamento, modelo e visualização. | Recomendado | Evitar lógica duplicada entre notebook e dashboard. |
| **RNF-10** | Prazo: entregar dentro do calendário acadêmico. | Pendente | O calendário não foi fornecido; confirmar com o professor. |

### 3.3 O que não é requisito

Os itens abaixo aparecem no esboço ou no contexto, mas **não** devem ser tratados como requisitos aprovados:

- os valores exatos do mockup (284.907 transações, R$ 25 Mi, 70% das fraudes abaixo de R$ 250);
- as métricas 99% / 94% / 92% / 93% e a matriz de confusão do mockup;
- a importância das variáveis V14, V10 e V17;
- a existência de uma aba de predição pronta para uso operacional;
- o filtro “tipo de transação” na base MLG/ULB;
- o período “01/01/2024 a 30/04/2025”;
- o bloqueio automático de transações.

O Apêndice A compara esses itens com a base real e explica as inconsistências.

---

## 4. Análise das bases de dados

### 4.1 Base 1 — Give Me Some Credit

**Origem:** competição histórica do Kaggle. **Problema:** prever inadimplência grave — pessoa que ficou 90 dias ou mais em atraso — em um horizonte de dois anos.

**Estrutura conhecida:**

- 150.000 registros de treino e 101.503 registros de teste;
- 10 variáveis explicativas e 1 variável-alvo (`SeriousDlqin2yrs`);
- variáveis sobre utilização de crédito rotativo, idade, atrasos, dívida, renda mensal, linhas de crédito abertas, empréstimos imobiliários e dependentes;
- valores ausentes conhecidos em `MonthlyIncome` e `NumberOfDependents`;
- valores-sentinela e extremos conhecidos nas contagens de atraso (por exemplo, 96 e 98) e em `RevolvingUtilizationOfUnsecuredLines`;
- alvo desbalanceado, com aproximadamente 6,7% de positivos na base de treino.

**Adequação ao projeto:** **baixa**. A base é de **risco de crédito**, não de fraude. Ela não contém transações, valor de compra, comerciante, dispositivo, horário ou rótulo de fraude. Também não há chave para cruzá-la com as bases de fraude. Usá-la exigiria mudar o problema do grupo.

**Possibilidades:** seria adequada a um projeto de scoring de crédito, com regressão logística ou árvores e avaliação por AUC. Não é o caso deste projeto.

**Recomendação:** não utilizar. Registrar no relatório como base avaliada e descartada, com justificativa.

### 4.2 Base 2 — Credit Card Fraud Detection (MLG/ULB)

**Origem:** colaboração entre Worldline e o Machine Learning Group da Université Libre de Bruxelles (ULB). É a base citada na apresentação e a que corresponde ao esboço do dashboard.

**Estrutura conhecida:**

- 284.807 transações;
- 31 colunas: `Time`, `V1` a `V28`, `Amount` e `Class`;
- 492 fraudes, ou 0,1727% das transações;
- transações de cartões europeus, em setembro de 2013, ao longo de dois dias;
- `Time`: segundos decorridos desde a primeira transação da base;
- `V1` a `V28`: componentes principais obtidos por PCA; o significado original não foi divulgado por confidencialidade;
- `Amount`: valor da transação;
- `Class`: 1 para fraude, 0 para transação legítima;
- sem valores ausentes;
- recomendação oficial de avaliação: **AUPRC** (área sob a curva precisão-recall), porque a acurácia é enganosa em bases desbalanceadas.

**Verificação local realizada em 17/11/2026** (arquivo público `creditcard.csv`, sem versionamento no repositório):

| Verificação | Resultado |
|---|---|
| Linhas | 284.807 |
| Colunas | 31 |
| Fraudes | 492 (0,1727%) |
| Valores ausentes | 0 |
| Duplicatas exatas | 1.081 linhas, das quais 19 são fraudes |
| `Time` | 0 a 172.792 segundos (≈ 2 dias) |
| `Amount` total | 25.162.590,01 (unidade original: moeda europeia; o dicionário não fixa “euros”) |
| `Amount` mínimo / máximo | 0,00 / 25.691,16 |
| Transações com valor zero | 1.825, das quais 27 são fraudes |
| Mediana de `Amount` | 22,00 em todas as transações; 9,25 entre as fraudes |

**Pontos fortes:**

- dado real anonimizado, com rótulo de fraude;
- base de referência mundial, com literatura e benchmarks;
- compatível com a apresentação, com o esboço e com os números 284.807 / 492 / 0,17%;
- permite classificação binária e avaliação por precision, recall, F1, AUPRC e matriz de confusão.

**Limitações:**

- atributos anonimizados por PCA: não é possível nomear os fatores de negócio por trás de V1–V28;
- apenas dois dias: não há sazonalidade, tendência de longo prazo ou datas absolutas;
- `Time` é relativo; a origem não é documentada como meia-noite, então “hora do dia” exige uma premissa;
- não há identificador de cartão, comerciante, categoria, dispositivo ou tipo de transação;
- desbalanceamento extremo (492 positivos);
- 1.081 duplicatas exatas, incluindo 19 fraudes: removê-las altera a classe positiva;
- risco de vazamento se transformações forem ajustadas antes da separação treino/teste.

**Adequação:** **alta**, desde que as limitações sejam assumidas e o dashboard não prometa análises que a base não permite.

### 4.3 Base fornecida no link “Credit Card Fraud Detection Dataset” (sintética, 10 mil)

**Origem:** conjunto `miadul/credit-card-fraud-detection-dataset`, no Kaggle, usado pelo notebook indicado no link fornecido. O título é parecido com o da base clássica, mas o conteúdo é diferente.

**Estrutura conhecida:**

- 10.000 transações sintéticas;
- 10 colunas: `transaction_id`, `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, `is_fraud`;
- alvo `is_fraud`;
- a página da base declara distribuição “altamente desbalanceada (fraude ≈ 4–5%)”;
- a inspeção do arquivo público em 17/11/2026 encontrou **151 fraudes em 10.000 linhas (1,51%)**, e não 4–5%;
- sem valores ausentes e sem duplicatas exatas na inspeção;
- 24 valores de hora (0–23), 5 categorias de comerciante, `foreign_transaction` e `location_mismatch` binários, `device_trust_score` declarado de 0 a 100 (no arquivo, 25 a 99), `velocity_last_24h` de 0 a 9;
- `amount` entre 0,00 e 1.471,04 na inspeção;
- base declarada como sintética, com licença Apache 2.0, criada em dezembro de 2025.

**Pontos fortes:**

- atributos interpretáveis e diretamente ligados ao esboço: valor, hora, categoria de comerciante, transação internacional, incompatibilidade de localização, confiança do dispositivo, velocidade e idade;
- permite o filtro “tipo de transação” se `merchant_category` for usado como categoria;
- base pequena, rápida e fácil de explicar;
- permite construir o dashboard e demonstrar o fluxo completo sem depender de PCA.

**Limitações:**

- é **sintética**: não representa fraude real e não sustenta afirmações sobre o mundo;
- tem apenas 151 fraudes, o que limita a estabilidade das métricas;
- a própria página da base informa uma taxa de fraude que não corresponde ao arquivo;
- não é a base citada na apresentação nem a base do esboço;
- não há datas absolutas; a moeda não está claramente definida;
- a origem e o processo de geração não estão documentados em detalhe;
- usar essa base sem atualizar a apresentação criaria uma inconsistência acadêmica.

**Adequação:** **média como alternativa**, alta para prototipagem e para um dashboard interpretável, baixa para afirmações sobre fraude real.

### 4.4 Comparação

| Critério | Give Me Some Credit | MLG/ULB | Sintética 10K |
|---|---|---|---|
| Problema | Inadimplência / risco de crédito | Fraude em cartão | Fraude em cartão |
| Unidade | Pessoa / crédito | Transação | Transação |
| Registros | 150.000 treino + 101.503 teste | 284.807 | 10.000 |
| Variáveis | 10 + alvo | 30 + alvo | 9 + alvo |
| Alvo | `SeriousDlqin2yrs` | `Class` | `is_fraud` |
| Positivos | ≈ 6,7% | 0,1727% (492) | 1,51% (151) |
| Interpretabilidade | Alta | Baixa (PCA) | Alta |
| Análise temporal | Não | Limitada a 2 dias relativos | Hora do dia (0–23) |
| Tipo de transação | Não | Não | `merchant_category` |
| Data absoluta | Não | Não | Não |
| Adequação ao escopo | Baixa | Alta, com ressalvas | Média, se assumida como sintética |
| Papel recomendado | Descartar | Base principal | Alternativa documentada |

### 4.5 Recomendação sobre as bases

1. **Base principal recomendada: MLG/ULB.** É a base registrada na apresentação, a que aparece nas referências e a que explica os números do esboço. Também é a mais defensável academicamente como dado real anonimizado.
2. **Não usar *Give Me Some Credit*.** Ela não responde ao problema de fraude e não tem relação com as transações.
3. **Não misturar as bases.** Se a sintética for usada, ela deve ser um estudo separado ou um protótipo claramente rotulado, não uma extensão dos resultados da base real.
4. **Decisão do grupo necessária.** Se a prioridade for explicabilidade de negócio e o filtro “tipo de transação”, a base sintética é mais adequada — mas o grupo precisa atualizar o escopo, a apresentação e as referências, e deixar claro que os achados são de dados sintéticos.
5. **Se a base MLG/ULB for mantida:** remover ou redefinir o filtro “tipo de transação”; tratar a análise temporal como relativa aos dois dias; não usar “R$” sem justificar a moeda; e explicar que os fatores são componentes PCA, não variáveis de negócio.

### 4.6 Cuidados obrigatórios de tratamento

- **Duplicatas:** decidir e documentar. Na MLG/ULB, remover todas as duplicatas elimina 19 fraudes. Se forem mantidas, explicar por que; se forem removidas, registrar o impacto na classe positiva.
- **Valores ausentes:** verificar na base escolhida; a MLG/ULB não tem nulos, a sintética também não na inspeção. Ainda assim, o pipeline deve tratar nulos.
- **Outliers:** não remover automaticamente. Fraudes podem estar justamente nos extremos. Investigar e documentar.
- **Desbalanceamento:** tratar apenas no treino (pesos de classe, undersampling ou oversampling). Nunca aplicar SMOTE antes da separação treino/teste.
- **Vazamento:** separar treino/teste antes de ajustar escalonamento, imputação, seleção de atributos ou resampling.
- **Tempo:** na MLG/ULB, `Time` não é data absoluta. Se for derivar “hora do dia”, declarar a premissa e testar a sensibilidade; caso contrário, analisar blocos de horas decorridas.
- **Moeda:** a MLG/ULB é de cartões europeus e o dicionário não fixa “euros” explicitamente. O esboço usa “R$”. Corrigir rótulos ou justificar a conversão; não converter sem fonte.
- **Escalas:** `Amount` e `Time` têm escalas muito diferentes das componentes PCA. Modelos sensíveis a escala precisam de padronização; árvores não precisam, mas o pipeline deve ser consistente.
- **Poucos positivos:** evitar conclusões definitivas a partir de subgrupos com poucos casos. Usar validação cruzada e reportar incerteza quando possível.

**Onde inserir os arquivos:** a área `data/` já está preparada. Consulte `data/README.md` para os nomes esperados (`data/raw/creditcard.csv`, `data/raw/credit_card_fraud_10k.csv` ou `data/raw/cs-training.csv`), os links e as regras de não versionamento. Coloque apenas a base escolhida.

---

## 5. Fundamentação técnica e acadêmica

### 5.1 Ciclo de vida da aula 2

A AULA2 apresenta cinco etapas: **entender o problema → coleta de dados → pré-processamento → análise exploratória → construção de modelos**. O plano abaixo preserva essa lógica e desdobra a construção de modelos em preparação de atributos, treino e avaliação, porque são dependências reais.

| Etapa da AULA2 | O que a aula destaca | Como será aplicada |
|---|---|---|
| Entender o problema | Definir o problema antes de modelar | Seção 2 e decisão de escopo (E1) |
| Coleta de dados | Obter e conhecer a base | Aquisição, licença e dicionário (E2) |
| Pré-processamento | Limpeza, amostragem, desbalanceamento, transformação | Auditoria e limpeza (E3) e preparação final (E5) |
| EDA | Examinar, resumir e visualizar; orientar variáveis, transformações e modelo | EDA dos três objetivos (E4) |
| Construção de modelos | Aprender padrões e gerar previsões/explicações | Treino (E6) e avaliação (E7) |

### 5.2 Pré-processamento

A AULA2 define quatro frentes:

- **Limpeza:** remover ou tratar valores ausentes, duplicados e inconsistentes. No projeto, isso inclui validar faixas, detectar duplicatas e registrar remoções.
- **Amostragem:** selecionar subconjuntos representativos. No projeto, o uso principal é a separação treino/teste estratificada; amostragem adicional só se necessária para experimentos.
- **Desbalanceamento:** prevenir modelos tendenciosos. No projeto, isso é central: 0,1727% de fraudes na MLG/ULB e 1,51% na sintética.
- **Transformação:** normalizar, padronizar e codificar. No projeto, aplicar a `Amount`, `Time` e variáveis categóricas da sintética; as V1–V28 já são componentes PCA.

### 5.3 Análise exploratória

A AULA2 indica que a EDA deve responder:

- como os dados estão distribuídos;
- se existem padrões ou tendências;
- se há relações entre variáveis;
- se existem outliers ou comportamentos inesperados.

Técnicas indicadas: estatísticas descritivas (média, mediana, desvio padrão), visualizações (histogramas, boxplots, dispersão, heatmaps) e análise de correlação.

No projeto, a EDA deve produzir evidências para:

- **OE-1:** comparação entre fraudes e não fraudes; distribuições; frequência; outliers; correlações.
- **OE-2:** valores e faixas de valor; proporção de fraudes por faixa; cuidado com a assimetria de `Amount`.
- **OE-3:** distribuição temporal; hora/bloco de hora; heatmap; limitações da janela de dois dias.

### 5.4 Construção de modelos

A AULA2 diferencia:

- **aprendizado supervisionado:** dados rotulados; classificação e regressão; exemplo de árvore de decisão;
- **não supervisionado:** busca de padrões por agrupamento e associação;
- **reforço:** aprende com erros, valor e política.

Para este projeto, o problema tem rótulo binário e objetivo preditivo. Portanto, a abordagem principal é **classificação supervisionada**. Técnicas não supervisionadas podem ser estudadas como complemento, mas não são necessárias para cumprir o escopo. Reforço não se aplica ao problema e não deve ser adicionado.

Candidatos razoáveis, a comparar:

- **baseline:** regressão logística ou árvore de decisão simples, para estabelecer referência;
- **modelo principal:** Random Forest, Gradient Boosting ou outro ensemble de árvores;
- **opcional:** ajuste de limite de decisão e pesos de classe;
- **opcional:** regras de risco simples, como no notebook da base sintética, apenas como baseline interpretável.

O mockup mostra “Random Forest”, mas isso é uma ilustração. A escolha final deve ser justificada por desempenho, interpretabilidade e simplicidade, não pelo desenho.

### 5.5 Técnicas observadas nos notebooks da aula

O `Aula2.ipynb` demonstra um fluxo completo com pandas:

- leitura de CSV e inspeção (`head`, `sample`, `info`);
- verificação de nulos, descrição estatística e duplicatas;
- validação de regras de negócio (por exemplo, contagem de avaliações menor que contagem de comentários);
- remoção de colunas, renomeação, filtros e reset de índice;
- conversão de datas e tratamento de datas inválidas;
- agrupamento e agregação (`groupby`, `agg`);
- junção de tabelas (`merge`) e explosão de listas (`explode`);
- visualizações com matplotlib e seaborn (barplot, violinplot).

O `Extra_Aula2.ipynb` acrescenta:

- `value_counts` para faixas etárias, profissões e estados;
- gráficos de barras, barras horizontais e pizza;
- títulos e rótulos.

A AULA2 também apresenta o **Google Colab** como ambiente das práticas. O projeto pode usar Colab ou ambiente local, desde que a execução seja documentada e os resultados possam ser reproduzidos. Para o dashboard, a execução local tende a ser mais simples de demonstrar.

**Como usar:** esses notebooks são referência de nível técnico e de estilo, não um molde a copiar. O projeto deve adaptar as técnicas ao problema de fraude e manter o código reproduzível.

### 5.6 Avaliação

A base MLG/ULB recomenda **AUPRC** por causa do desbalanceamento. O plano adota:

- **AUPRC** como métrica principal de ranking;
- **precision, recall e F1** para a classe fraude;
- **matriz de confusão**;
- **ROC-AUC** como métrica complementar;
- **acurácia** apenas como contextualização, nunca como métrica principal;
- **limite de decisão** explícito, com discussão do trade-off entre falsos positivos e falsos negativos;
- **validação cruzada** no treino e **holdout** para a avaliação final.

Não definir metas numéricas arbitrárias antes de treinar o baseline. O critério inicial deve ser: o modelo precisa ser melhor que o baseline em AUPRC e apresentar recall/precision compatíveis com o limite escolhido, com justificativa de negócio.

### 5.7 Rigor acadêmico

- Toda afirmação sobre fraude deve indicar a base e a limitação.
- Resultados da base sintética não podem ser apresentados como evidência de fraude real.
- Números do mockup não podem ser reaproveitados sem recálculo.
- Citações devem seguir o padrão solicitado pela instituição (provavelmente ABNT; confirmar).
- O relatório deve registrar decisões, transformações, métricas, limitações e alternativas descartadas.
- Se forem usadas ferramentas de IA, seguir a política da instituição e declarar o uso quando exigido.

---

## 6. Arquitetura e organização proposta

### 6.1 Princípios

- **Simplicidade:** um projeto acadêmico não precisa de arquitetura de produção.
- **Reprodutibilidade:** qualquer pessoa deve conseguir refazer o fluxo.
- **Separação de responsabilidades:** dados, features, modelo, visualização e dashboard não devem ficar misturados.
- **Rastreabilidade:** cada número do dashboard deve vir de um artefato gerado pelo pipeline.
- **Escopo controlado:** o MVP são as quatro visualizações exigidas; o resto é incremento.

### 6.2 Fluxo lógico

```text
Kaggle
  │
  ▼
data/raw/ (imutável, não versionado)
  │
  ▼
validação e auditoria ──► relatório de qualidade
  │
  ▼
limpeza e transformação ──► data/processed/
  │
  ├──► EDA ──► figuras e insights
  │
  ▼
features + split treino/teste
  │
  ▼
treino ──► modelo ──► avaliação ──► métricas e artefatos
  │
  ▼
dashboard (agregações + resultados do modelo)
```

#### 6.2.1 Fluxo lógico canônico em camadas (proposto)

O diagrama acima descreve o pipeline de dados e modelagem. O fluxo abaixo é a referência **canônica** para o caminho de serviço do dashboard; é uma proposta para a fase de implementação, sem código criado nesta etapa:

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

As regras de dependência, o mapeamento proposto para as pastas e o papel da trilha offline de Machine Learning estão em [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), documento de referência deste desenho. A base (P-01) e a ferramenta (P-04) seguem pendentes: o fluxo é agnóstico a ambas.

### 6.3 Estrutura de pastas

A **área de dados já foi criada** para receber as bases: `data/raw/`, `data/interim/`, `data/processed/` e `data/README.md`, com instruções de onde colocar cada arquivo. A estrutura abaixo foi implementada no MVP: as pastas de camadas existem em `src/projeto_integrador/` e os relatórios são gerados em `reports/`. O scaffold original (`main.py`, testes e configuração de Ruff/pytest) foi preservado.

```text
projeto-integrador/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── ARCHITECTURE.md           # referência canônica do fluxo em camadas
│   ├── DATA_DICTIONARY.md        # a criar na etapa de dados
│   └── DECISIONS.md              # opcional: registro de decisões
├── data/
│   ├── README.md                 # instruções da área de dados
│   ├── raw/                      # arquivos originais, não versionados
│   ├── interim/                  # dados intermediários, não versionados
│   └── processed/                # dados finais, não versionados
├── notebooks/
│   ├── 01_auditoria.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modelagem.ipynb
├── src/projeto_integrador/
│   ├── __init__.py
│   ├── config.py                 # caminhos, sementes, constantes
│   ├── data/                     # Data Access: carga, validação e cache
│   ├── analytics/                # Analytics/Domain: métricas, agregações e regras
│   ├── application/              # Application: filtros, serviços e estado da sessão
│   ├── features/                 # trilha offline: transformações para modelagem
│   ├── models/                   # trilha offline: treino e avaliação
│   └── presentation/             # Presentation: páginas, filtros, KPIs, gráficos e tabelas
├── reports/
│   └── figures/                  # figuras geradas
└── tests/
```

**Mudanças propostas (não aprovadas):** `analytics/` e `application/` são diretórios novos propostos; `viz/` e `dashboard/` da proposta anterior ficam **absorvidos** por `presentation/`; `features/` e `models/` permanecem na trilha offline de ML (E5–E7), sem renomeação. O mapeamento completo por camada está em [`docs/ARCHITECTURE.md`](ARCHITECTURE.md). Nenhuma pasta deve ser criada enquanto a fase for de planejamento.

**Cuidados:**

- a área `data/` já está protegida no `.gitignore`; mantenha os arquivos de dados fora do Git;
- consulte `data/README.md` para os nomes e links das bases;
- não editar os arquivos de `data/raw/`; gerar cópias tratadas em `data/interim/` ou `data/processed/`;
- não duplicar lógica entre notebook e módulo: o notebook pode importar funções de `src/`;
- manter o entrypoint e os testes existentes funcionando;
- criar apenas os módulos necessários, não uma estrutura vazia.

### 6.4 Tecnologias

| Camada | Situação atual | Sugestão | Status |
|---|---|---|---|
| Linguagem | Python ≥3.11 | Manter | Atual |
| Ambiente | `uv` + `.venv` | Manter | Atual |
| Lint/formatação | Ruff | Manter | Atual |
| Testes | pytest | Manter e ampliar | Atual |
| Manipulação | — | pandas, numpy | Sugerido |
| Visualização | — | matplotlib, seaborn | Sugerido |
| ML | — | scikit-learn | Sugerido |
| Desbalanceamento | — | `class_weight`; opcional `imbalanced-learn` | Sugerido |
| Dashboard | — | Streamlit (mais simples) ou Dash/Plotly | **Decidir** |
| Notebooks | — | Jupyter/Colab para EDA e experimentos | Sugerido |
| Deploy | — | Não necessário no MVP | **Decidir** |

**Recomendação:** usar Streamlit se não houver exigência institucional de outra ferramenta, porque reduz o esforço de interface e mantém o dashboard em Python. Dash/Plotly oferece mais controle visual, mas aumenta a complexidade. Não adicionar dependências antes de a ferramenta ser escolhida.

### 6.5 Camada de dados do dashboard

Para a base MLG/ULB, o dashboard não deve carregar 284 mil linhas e recalcular tudo a cada interação se isso prejudicar o desempenho. Opções:

- pré-agregar indicadores e séries em arquivos leves;
- cachear o carregamento;
- carregar apenas as colunas necessárias.

A predição individual, se entrar no escopo, deve usar o mesmo pipeline de transformação do treino, não uma reimplementação paralela.

---

## 7. Etapas de desenvolvimento

### 7.1 Sequência e dependências

```text
E1 Escopo
  │
  ▼
E2 Dados
  │
  ▼
E3 Auditoria/limpeza ◄──┐
  │                     │
  ▼                     │
E4 EDA ─────────────────┘
  │
  ▼
E5 Features/split
  │
  ▼
E6 Modelagem ◄──┐
  │             │
  ▼             │
E7 Avaliação ───┘
  │
  ▼
E8 Dashboard
  │
  ▼
E9 Integração/validação
  │
  ▼
E10 Entregáveis acadêmicos
```

As setas de retorno indicam iterações esperadas: a EDA pode revelar problemas que voltam para a limpeza; a avaliação pode exigir novo treino; o dashboard pode revelar a necessidade de novos cortes na EDA.

### E1 — Entendimento e definição do problema

**Objetivo:** alinhar o grupo sobre problema, objetivos, público, escopo, base, ferramenta e critérios de sucesso antes de codar.

**Atividades:**

- revisar a apresentação e este plano com o grupo;
- confirmar a base e a versão (MLG/ULB ou sintética);
- confirmar o público-alvo e os usos do dashboard;
- definir o MVP: quatro visualizações obrigatórias e filtros viáveis;
- decidir ferramenta do dashboard e se haverá aba de predição;
- confirmar formato, prazos e rubrica com o professor;
- registrar as decisões na Seção 10 e atualizar o `AGENTS.md`.

**Conhecimentos:** definição de problema, ciclo de vida, requisitos, escopo.

**Dependências:** materiais fornecidos, disponibilidade do grupo e orientação do professor.

**Resultado esperado:** escopo aprovado, decisões registradas, pendências críticas resolvidas.

**Concluída quando:**

- a base escolhida está registrada com link e versão;
- o público e o MVP estão definidos;
- a ferramenta do dashboard está escolhida;
- o formato de entrega e o calendário estão confirmados;
- não há requisito conflitante sem decisão.

### E2 — Aquisição e governança dos dados

**Objetivo:** obter a base correta, garantir proveniência, licença e armazenamento adequado.

**Atividades:**

- baixar a base pela fonte oficial;
- registrar URL, data de acesso, versão e licença;
- calcular hash ou checksum e conferir linhas/colunas;
- criar dicionário de dados;
- armazenar o arquivo bruto em `data/raw/` sem edições, seguindo `data/README.md`;
- conferir se a área de dados continua protegida pelo `.gitignore`;
- verificar se há dados pessoais ou restrições de uso.

**Conhecimentos:** coleta, governança, licenciamento, privacidade.

**Dependências:** E1 (base escolhida).

**Resultado esperado:** base bruta, dicionário de dados e registro de proveniência.

**Concluída quando:**

- a base pode ser obtida novamente de forma documentada;
- o checksum e as dimensões conferem;
- a licença é conhecida;
- não há PII no repositório.

### E3 — Auditoria e pré-processamento

**Objetivo:** diagnosticar a qualidade dos dados e produzir uma base limpa, com decisões documentadas.

**Atividades:**

- inspecionar tipos, faixas, nulos, duplicatas e valores inválidos;
- medir a distribuição do alvo;
- investigar duplicatas (na MLG/ULB, 1.081 linhas, incluindo 19 fraudes);
- investigar valores zero, outliers e inconsistências;
- decidir o que remover, corrigir ou manter;
- documentar cada regra e o impacto no número de fraudes;
- criar relatório de qualidade.

**Conhecimentos:** limpeza, validação, estatística descritiva, análise de dados faltantes.

**Dependências:** E2.

**Resultado esperado:** base limpa, relatório de qualidade e log de transformações.

**Concluída quando:**

- todos os problemas de qualidade estão quantificados;
- cada remoção tem justificativa;
- o impacto na classe positiva está registrado;
- a base limpa é reproduzível por script ou notebook.

### E4 — Análise exploratória dos dados

**Objetivo:** entender distribuições, padrões, relações e anomalias, e responder aos três objetivos específicos.

**Atividades:**

- estatísticas descritivas gerais e por classe;
- histogramas, boxplots e dispersão para valor e atributos relevantes;
- comparação fraudes × não fraudes;
- análise por faixa de valor;
- análise temporal (com a limitação da base);
- heatmap e correlação;
- identificação de outliers e comportamentos inesperados;
- registro de insights e de hipóteses para a modelagem;
- rascunho das visualizações do dashboard.

**Conhecimentos:** EDA, estatística descritiva, visualização, correlação.

**Dependências:** E3.

**Resultado esperado:** notebook/relatório de EDA, figuras e lista de achados.

**Concluída quando:**

- cada objetivo específico tem evidência visual ou estatística;
- as limitações da base estão explícitas;
- não há afirmações causais sem suporte;
- os achados orientam a seleção de variáveis e o tipo de modelo.

### E5 — Preparação de atributos e separação treino/teste

**Objetivo:** construir o conjunto de modelagem de forma reproduzível e sem vazamento.

**Atividades:**

- definir variáveis de entrada e alvo;
- criar indicadores justificados pela EDA (faixas de valor, hora/bloco de hora, categorias);
- codificar variáveis categóricas;
- padronizar variáveis numéricas quando necessário;
- separar treino e teste de forma estratificada;
- definir a estratégia de desbalanceamento (pesos, undersampling ou oversampling) apenas no treino;
- montar um pipeline de pré-processamento;
- documentar o dicionário de features.

**Conhecimentos:** transformação, codificação, escalonamento, amostragem, prevenção de vazamento.

**Dependências:** E3 e E4; decisão sobre o modelo.

**Resultado esperado:** pipeline de features, conjuntos de treino/teste e dicionário de atributos.

**Concluída quando:**

- o pipeline é executável do início ao fim;
- o teste não participa de nenhum ajuste;
- a estratégia de desbalanceamento está justificada;
- as features são compreensíveis e documentadas.

### E6 — Modelagem

**Objetivo:** treinar e comparar modelos de classificação para estimar a probabilidade de fraude.

**Atividades:**

- treinar um baseline simples;
- treinar pelo menos um modelo candidato (árvore, Random Forest ou boosting);
- usar validação cruzada no treino;
- ajustar hiperparâmetros com critério definido;
- testar pesos de classe ou reamostragem;
- registrar sementes, versões e resultados;
- escolher o modelo final com justificativa.

**Conhecimentos:** aprendizado supervisionado, classificação, árvores, ensembles, validação cruzada.

**Dependências:** E5.

**Resultado esperado:** modelos treinados, experimentos registrados e modelo selecionado.

**Concluída quando:**

- há pelo menos dois modelos comparáveis;
- o baseline foi superado ou há justificativa para não superá-lo;
- os experimentos são reproduzíveis;
- o modelo final está salvo com o pipeline correspondente.

### E7 — Avaliação e interpretação

**Objetivo:** medir o desempenho no holdout, escolher o limite de decisão e interpretar os fatores associados.

**Atividades:**

- calcular AUPRC, precision, recall, F1, ROC-AUC e matriz de confusão;
- analisar o efeito do limite de decisão;
- comparar com o baseline;
- analisar erros (falsos positivos e falsos negativos);
- extrair importância de variáveis ou permutação;
- discutir o significado dos fatores, respeitando a limitação de PCA na MLG/ULB;
- registrar limitações e riscos de generalização;
- produzir um “model card” resumido.

**Conhecimentos:** avaliação de classificação desbalanceada, interpretabilidade, análise de erro.

**Dependências:** E6.

**Resultado esperado:** relatório de avaliação, figuras e modelo documentado.

**Concluída quando:**

- as métricas são reproduzíveis;
- a acurácia não é a métrica principal;
- o limite é justificado;
- os fatores são apresentados com o devido cuidado;
- as limitações estão no relatório.

### E8 — Dashboard

**Objetivo:** entregar um dashboard interativo com as quatro visualizações mínimas e dados consistentes.

**Atividades:**

- definir layout e navegação;
- implementar indicadores gerais;
- implementar distribuição de fraudes;
- implementar fraudes por faixa de valor;
- implementar evolução temporal;
- adicionar filtros viáveis (período, faixa, status);
- decidir e implementar a aba de predição, se aprovada;
- corrigir rótulos de moeda e tempo;
- validar números contra a EDA e o modelo;
- testar usabilidade e desempenho;
- documentar como executar.

**Conhecimentos:** visualização, design de dashboard, storytelling com dados.

**Dependências:** E4 (visualizações) e E7 (resultados do modelo); ferramenta escolhida em E1.

**Resultado esperado:** aplicação executável, com as quatro visualizações e documentação de uso.

**Concluída quando:**

- as quatro visualizações obrigatórias estão presentes e funcionam;
- os filtros não quebram;
- os números batem com os artefatos;
- não há valores fictícios do mockup;
- a execução local está documentada;
- o dashboard é compreensível para o público definido.

### E9 — Integração e validação final

**Objetivo:** garantir que o conjunto funciona de ponta a ponta e é reproduzível.

**Atividades:**

- executar o pipeline em ambiente limpo;
- conferir dependências e instruções do README;
- rodar testes e lint;
- validar a matriz de rastreabilidade dos requisitos;
- revisar dados, métricas e figuras;
- revisar segurança e privacidade;
- corrigir inconsistências;
- congelar a versão para a entrega.

**Conhecimentos:** testes, reprodutibilidade, documentação, revisão.

**Dependências:** E2 a E8.

**Resultado esperado:** repositório integrado, validado e documentado.

**Concluída quando:**

- uma pessoa consegue reproduzir os resultados principais;
- o dashboard abre e responde;
- testes e lint passam;
- não há vazamento de dados;
- a documentação está atualizada.

### E10 — Entregáveis acadêmicos

**Objetivo:** produzir relatório, apresentação e demonstração alinhados à rubrica.

**Atividades:**

- escrever introdução, problema, objetivos e justificativa;
- documentar metodologia, dados, tratamento, EDA, modelagem e avaliação;
- apresentar resultados com figuras e tabelas;
- discutir limitações e trabalhos futuros;
- revisar citações e referências;
- preparar slides e roteiro de demonstração;
- ensaiar a defesa;
- revisar a aderência à rubrica.

**Conhecimentos:** escrita científica, comunicação, ética, apresentação.

**Dependências:** E9 e rubrica oficial.

**Resultado esperado:** relatório, slides, demo e código organizado.

**Concluída quando:**

- a rubrica está atendida;
- as fontes estão citadas;
- os resultados são rastreáveis;
- a demo funciona;
- o grupo consegue explicar e defender as decisões.

---

## 8. Dependências e riscos

### 8.1 Dependências críticas

| Dependência | Bloqueia | Observação |
|---|---|---|
| Definição da base (MLG/ULB ou sintética) | E2–E8 | Sem isso, o escopo do dashboard e as métricas mudam. |
| Ferramenta do dashboard | E8 | Streamlit, Dash ou outra; decidir antes de implementar. |
| Público-alvo | E8 e E10 | Afeta linguagem, filtros e narrativa. |
| Rubrica e prazos | E10 | Não fornecidos nos materiais. |
| Estratégia de métricas e limite | E7 | Depende do objetivo de negócio. |
| Definição sobre a aba de predição | E8 | Pode ser MVP ou incremento. |
| Política de uso de IA e citações | E10 | Confirmar com a instituição. |

### 8.2 Riscos

| ID | Risco | Impacto | Mitigação |
|---|---|---|---|
| **R-01** | Ambiguidade entre a base MLG/ULB e a sintética. | Alto: muda escopo, números e conclusões. | Decidir em E1; registrar versão e link; não misturar. |
| **R-02** | Usar números do mockup como se fossem resultados. | Alto: integridade acadêmica. | Recalcular tudo; remover ou rotular o mockup como ilustrativo. |
| **R-03** | Desbalanceamento extremo. | Alto: modelo aparentemente bom e inútil. | AUPRC, precision, recall, F1, matriz de confusão; pesos/reamostragem. |
| **R-04** | Vazamento de dados. | Alto: métricas infladas. | Split antes de transformar; pipeline; validar. |
| **R-05** | Duplicatas na MLG/ULB. | Médio: 19 fraudes removidas se descartadas. | Decidir e documentar; comparar cenários. |
| **R-06** | Atributos anonimizados por PCA. | Médio: explicação de negócio limitada. | Explicar a limitação; usar importância de componentes e `Amount`. |
| **R-07** | Tempo relativo e janela de dois dias. | Médio: análise temporal limitada. | Analisar blocos relativos; não afirmar sazonalidade; declarar premissas. |
| **R-08** | Moeda do esboço (R$) diferente da base. | Médio: gráficos e conclusões errados. | Definir unidade; corrigir rótulos; não converter sem fonte. |
| **R-09** | Filtro “tipo de transação” inexistente na MLG/ULB. | Médio: requisito impossível. | Remover, redefinir ou migrar para a base sintética. |
| **R-10** | Poucos positivos (492 ou 151). | Médio: métricas instáveis. | Validação cruzada; evitar subgrupos pequenos; reportar incerteza. |
| **R-11** | Escopo do dashboard crescer demais. | Médio: atraso. | MVP com quatro visualizações; incrementos depois. |
| **R-12** | Ferramenta do dashboard mal escolhida. | Médio: retrabalho. | Protótipo rápido; decidir em E1. |
| **R-13** | Acesso ao Kaggle ou mudança de versão. | Baixo/médio: retrabalho. | Baixar cedo; guardar cópia bruta; registrar checksum. |
| **R-14** | Base sintética ser apresentada como real. | Alto: credibilidade. | Rotular claramente; não usar para conclusões sobre fraude real. |
| **R-15** | Falta de rubrica/prazos. | Médio: entrega fora do esperado. | Confirmar com o professor antes de E10. |
| **R-16** | Reprodução inconsistente entre notebook e dashboard. | Médio: números divergentes. | Fonte única de artefatos; testes de conferência. |

### 8.3 Premissas explícitas

Estas premissas **não** são fatos dos materiais; devem ser validadas:

- o dashboard será usado por analistas/gestores, e não por clientes finais;
- a execução local é suficiente; não há exigência de deploy;
- o projeto será entregue como relatório + apresentação + código;
- a instituição aceita o uso de dados públicos anonimizados;
- o grupo pode usar Python e bibliotecas open source;
- não há restrição de tamanho ou de ferramenta que impeça a base MLG/ULB.

### 8.4 Restrições

- tempo acadêmico limitado;
- equipe de cinco integrantes, com disponibilidade não documentada;
- base pública, sem possibilidade de coletar dados próprios;
- a base real tem apenas dois dias e atributos anonimizados;
- o projeto é acadêmico, não um produto comercial.

---

## 9. Critérios de validação

### 9.1 Portões por etapa

| Etapa | Critério de aceite |
|---|---|
| E1 | Base, público, MVP, ferramenta e formato de entrega definidos e registrados. |
| E2 | Base obtida, checksum/dimensões conferidos, licença e dicionário documentados. |
| E3 | Qualidade auditada; remoções justificadas; impacto na classe positiva registrado. |
| E4 | Cada objetivo específico respondido com estatística/figura; limitações explícitas. |
| E5 | Pipeline reproduzível; split estratificado; sem vazamento; desbalanceamento tratado no treino. |
| E6 | Pelo menos dois modelos comparados; validação cruzada; sementes registradas. |
| E7 | Métricas completas; limite justificado; erros analisados; limitações registradas. |
| E8 | Quatro visualizações obrigatórias; filtros funcionando; números conferidos. |
| E9 | Execução em ambiente limpo; testes/lint; documentação; rastreabilidade. |
| E10 | Rubrica atendida; fontes citadas; demo; defesa ensaiada. |

### 9.2 Checklist de dados

- [ ] URL, data de acesso, versão e licença registradas.
- [ ] Dimensões e checksum conferidos.
- [ ] Dicionário de dados criado.
- [ ] Nulos, duplicatas, outliers e valores inválidos avaliados.
- [ ] Distribuição do alvo registrada.
- [ ] Decisão sobre duplicatas documentada.
- [ ] Nenhum dado pessoal identificável no repositório.
- [ ] Dados brutos imutáveis e não versionados.

### 9.3 Checklist de modelagem

- [ ] Split treino/teste definido antes das transformações.
- [ ] Baseline treinado.
- [ ] Pelo menos dois modelos comparados.
- [ ] Validação cruzada usada no treino.
- [ ] Desbalanceamento tratado apenas no treino.
- [ ] AUPRC, precision, recall, F1, ROC-AUC e matriz de confusão calculados.
- [ ] Limite de decisão justificado.
- [ ] Importância das variáveis interpretada com cautela.
- [ ] Sementes e versões registradas.
- [ ] Modelo e pipeline salvos juntos.

### 9.4 Checklist do dashboard

- [ ] Indicadores gerais.
- [ ] Distribuição de fraudes.
- [ ] Fraudes por faixa de valor.
- [ ] Evolução temporal.
- [ ] Filtros viáveis e coerentes.
- [ ] Rótulos de moeda e tempo corretos.
- [ ] Números conferidos contra a EDA/modelo.
- [ ] Nenhum valor fictício do mockup.
- [ ] Execução local documentada.
- [ ] Usabilidade testada com alguém do grupo.

### 9.5 Checklist acadêmico

- [ ] Problema, objetivos e justificativa claros.
- [ ] Metodologia alinhada ao ciclo de vida da AULA2.
- [ ] Bases descritas e escolha justificada.
- [ ] Tratamento e limitações documentados.
- [ ] Resultados reproduzíveis.
- [ ] Figuras e tabelas com fonte.
- [ ] Citações no padrão exigido.
- [ ] Limitações e trabalhos futuros.
- [ ] Uso de IA declarado, se exigido.
- [ ] Apresentação e demo ensaiadas.

### 9.6 Critério de “pronto” do projeto

O projeto estará pronto para entrega quando:

1. a base escolhida estiver documentada e tratada;
2. a EDA responder aos três objetivos específicos;
3. o modelo estiver treinado, avaliado e interpretado com métricas adequadas;
4. o dashboard tiver as quatro visualizações obrigatórias e números consistentes;
5. o pipeline puder ser reproduzido em ambiente limpo;
6. o relatório e a apresentação estiverem alinhados à rubrica;
7. todas as pendências críticas da Seção 10 estiverem resolvidas ou explicitamente assumidas como limitações.

---

## 10. Pendências, decisões e recomendações

### 10.1 Decisões registradas

| ID | Decisão | Evidência | Status |
|---|---|---|---|
| **D-01** | O tema é detecção de fraudes em transações com cartão de crédito. | APRES, p. 2–3 | Tomada |
| **D-02** | O objetivo geral é um dashboard interativo com análise preditiva. | APRES, p. 4 | Tomada |
| **D-03** | Os objetivos específicos são padrões, valor e tempo. | APRES, p. 5–6 | Tomada |
| **D-04** | O dashboard deve ter, no mínimo, quatro visualizações. | APRES, p. 10 | Tomada |
| **D-05** | Haverá aplicação de Machine Learning. | APRES, p. 11 | Tomada |
| **D-06** | A base será uma base de fraude do Kaggle. | APRES, p. 9 | Parcial: falta definir qual |
| **D-07** | O projeto seguirá o ciclo de vida da AULA2. | AULA2, p. 3–16 | Diretriz acadêmica |
| **D-08** | A documentação e o código serão em português. | Convenção do repositório e materiais | Tomada |
| **D-09** | A interface do dashboard será em inglês; documentação e código permanecem em português. Aplica-se a rótulos de UI (KPIs, filtros, títulos/eixos/legendas, cabeçalhos, navegação, mensagens e estados vazios/de erro), não a valores brutos da base. | Decisão confirmada pelo grupo | Tomada; substitui apenas a parte de idioma da interface de RNF-03 e mantém D-08 integralmente para documentação e código. |
| **D-10 a D-25** | Decisões da implementação: base MLG/ULB, remoção de duplicatas, tempo relativo, moeda original, Streamlit, filtros viáveis, modelos, limite de decisão, métricas, split e organização em camadas. | [`DECISIONS.md`](DECISIONS.md) | Tomadas e aplicadas no pipeline. |

> **Nota:** o registro detalhado com justificativa e impacto de D-10 a D-25 está em [`DECISIONS.md`](DECISIONS.md).

### 10.2 Recomendações

| ID | Recomendação | Justificativa |
|---|---|---|
| **R-01** | Usar a MLG/ULB como base principal. | É a base citada na apresentação, compatível com o esboço e com dado real anonimizado. |
| **R-02** | Não usar *Give Me Some Credit*. | É de inadimplência, não de fraude, e não tem variáveis transacionais. |
| **R-03** | Não misturar bases. | Misturar fontes diferentes confunde resultados e conclusões. |
| **R-04** | Se usar a sintética, rotulá-la como sintética em todo lugar. | Evita afirmações sobre fraude real. |
| **R-05** | Adotar AUPRC como métrica principal. | Recomendação oficial da base e adequada ao desbalanceamento. |
| **R-06** | Comparar baseline e modelo candidato. | Evita escolher Random Forest apenas porque aparece no mockup. |
| **R-07** | Usar Streamlit, salvo exigência contrária. | Menor complexidade para um dashboard acadêmico em Python. |
| **R-08** | Fazer o MVP com as quatro visualizações antes de incrementos. | Controla escopo e prazo. |
| **R-09** | Remover ou redefinir “tipo de transação” na MLG/ULB. | O campo não existe; não inventar dados. |
| **R-10** | Corrigir moeda e eixo temporal no dashboard. | O esboço usa R$ e datas que não existem na base. |
| **R-11** | Criar um dicionário de dados e um registro de decisões. | Rastreabilidade acadêmica. |
| **R-12** | Confirmar rubrica, prazos e formato com o professor. | Não estão nos materiais fornecidos. |

### 10.3 Pendências

| ID | Pendência | Por que importa | Encaminhamento sugerido |
|---|---|---|---|
| **P-01** | Qual base será usada: MLG/ULB ou sintética 10K? | Muda escopo, métricas, dashboard e narrativa. | **Resolvida (D-10):** MLG/ULB adotada como base única. |
| **P-02** | O link fornecido para a “Base 2” aponta para um notebook da base sintética, não para a MLG/ULB. Qual era a intenção? | Evita usar a base errada. | **Resolvida (D-10):** o pipeline usa `data/raw/creditcard.csv` (MLG/ULB). |
| **P-03** | Qual é o público-alvo? | Afeta linguagem, filtros e storytelling. | Definir com o grupo/professor. |
| **P-04** | Qual ferramenta de dashboard? | Define dependências e arquitetura. | **Resolvida (D-15):** Streamlit. |
| **P-05** | Haverá aba de predição? | Pode aumentar muito o escopo. | **Fora do MVP (D-18):** incremento futuro; decidir antes de implementar. |
| **P-06** | Quais modelos e qual limite de decisão? | Define métricas e resultados. | **Definido no pipeline (D-19/D-20):** baseline + RF + boosting; limite por F1 out-of-fold. |
| **P-07** | Qual estratégia de desbalanceamento? | Afeta desempenho e interpretação. | **Definido no pipeline (D-19):** `class_weight="balanced"`, sem reamostragem. |
| **P-08** | O filtro “tipo de transação” será removido ou a base será trocada? | Na MLG/ULB o campo não existe. | **Resolvida (D-16):** filtro removido; a interface informa a ausência. |
| **P-09** | Como tratar o tempo na MLG/ULB? | `Time` é relativo e cobre dois dias. | **Resolvida (D-13):** blocos relativos (hora decorrida, dia relativo). |
| **P-10** | Qual moeda usar nos gráficos? | O esboço usa R$, a base é europeia. | **Resolvida (D-14):** unidade original, sem símbolo de moeda. |
| **P-11** | Quais são o formato, os prazos e a rubrica? | Não estão nos materiais; o slide “CALENDÁRIO” da AULA2 não tem datas legíveis. | Confirmar com o professor. |
| **P-12** | Quais são os papéis e a disponibilidade do grupo? | Afeta cronograma. | Definir responsáveis por etapa. |
| **P-13** | Como armazenar e versionar os dados? | Evita commits acidentais e perda de versão. | `data/` no `.gitignore`; checksum; cópia bruta. |
| **P-14** | Qual política de citação e de uso de IA? | Pode afetar a avaliação. | Consultar o professor/orientador. |
| **P-15** | Os números do mockup podem ser usados como referência visual? | Risco de confundir com resultados. | Usar apenas como layout; nunca como resultado. |

### 10.4 Decisões que não devem ser tomadas por suposição

- não assumir que a base sintética é a base da apresentação;
- não assumir que “R$” é a moeda correta;
- não assumir que `Time = 0` é meia-noite;
- não assumir que o modelo será Random Forest;
- não assumir que a acurácia é a métrica principal;
- não assumir que a aba de predição faz parte do MVP;
- não assumir que o dashboard será publicado na internet;
- não assumir que o público são clientes finais;
- não assumir que as métricas do mockup são reais.

---

## 11. Entregáveis previstos

| ID | Entregável | Tipo | Etapa | Status |
|---|---|---|---|---|
| **EN-01** | `AGENTS.md` com contexto do projeto | Documentação | E1 | **Criado nesta etapa** |
| **EN-02** | `docs/PROJECT_PLAN.md` | Documentação | E1 | **Criado nesta etapa** |
| **EN-03** | Dicionário de dados e registro de proveniência | Dados | E2 | **Concluído** — `docs/DATA_DICTIONARY.md` |
| **EN-04** | Base tratada + relatório de qualidade | Dados | E3 | **Concluído** — `data/processed/` e `reports/RELATORIO_QUALIDADE.md` |
| **EN-05** | Notebook/relatório de EDA com figuras | Análise | E4 | **Concluído** — `reports/RELATORIO_EDA.md` e figuras |
| **EN-06** | Pipeline de features e conjuntos treino/teste | Modelagem | E5 | **Concluído** — `features/` |
| **EN-07** | Modelo treinado + experimentos | Modelagem | E6 | **Concluído** — `models/` e `reports/model/metricas.json` |
| **EN-08** | Relatório de avaliação + modelo documentado | Modelagem | E7 | **Concluído** — `reports/RELATORIO_MODELO.md` |
| **EN-09** | Dashboard interativo | Produto | E8 | **Concluído** — `presentation/app.py` |
| **EN-10** | Repositório integrado + validação | Engenharia | E9 | Parcial — testes e lint executados; revisão final pendente |
| **EN-11** | Relatório final | Acadêmico | E10 | Planejado; formato pendente |
| **EN-12** | Apresentação/slides + demo | Acadêmico | E10 | Planejado; formato pendente |
| **EN-13** | Código, testes e instruções de execução | Engenharia | E9–E10 | **Concluído** — README e suíte pytest |
| **EN-14** | `docs/ARCHITECTURE.md` | Documentação | E1 | **Criado nesta etapa** |

**Entregáveis acadêmicos não especificados:** não há, nos materiais fornecidos, indicação de formato (PDF, artigo, vídeo, banca), número de páginas, prazo ou rubrica. Confirmar antes de iniciar E10.

---

## 12. Matriz de rastreabilidade

| Requisito | Fonte | Etapa | Evidência de conclusão |
|---|---|---|---|
| RF-01 | APRES, p. 9 | E1–E2 | Link, versão e checksum registrados |
| RF-02 | APRES, p. 9 | E3 | Base tratada + log de transformações |
| RF-03 | APRES, p. 5 | E4 | EDA com comparação fraudes × não fraudes |
| RF-04 | APRES, p. 5 e 10 | E4/E8 | Gráfico de fraudes por faixa de valor |
| RF-05 | APRES, p. 6 | E4/E8 | Série temporal + heatmap |
| RF-06 | APRES, p. 10 | E8 | Quatro visualizações no dashboard |
| RF-07 | Mockup | E1/E8 | Filtros implementados ou redefinidos |
| RF-08 | APRES, p. 11 | E6 | Modelo treinado |
| RF-09 | APRES, p. 11 | E7 | Métricas + interpretação |
| RF-10 | Mockup | E8 | KPIs no dashboard |
| RF-11 | Mockup | E1/E8 | Aba de predição decidida e implementada ou descartada |
| RF-12 | Recomendado | E9 | README + pipeline reproduzível |
| RNF-01 a RNF-10 | Ver Seção 3.2 | E1–E10 | Checklist da Seção 9 |

---

## 13. Próximos passos imediatos

1. **Revisar os resultados com o grupo:** conferir os relatórios gerados
   (`reports/RELATORIO_QUALIDADE.md`, `reports/RELATORIO_EDA.md` e
   `reports/RELATORIO_MODELO.md`) e validar as decisões D-10 a D-25.
2. **Confirmar com o professor (P-11/P-14):** rubrica, prazos, formato de entrega
   e política de uso de IA.
3. **Definir o público-alvo (P-03):** a hipótese de analistas/gestores de risco
   ainda não foi validada.
4. **Decidir sobre a aba de predição (P-05):** está fora do MVP (D-18); se for
   incluída, deve reutilizar o pipeline salvo em `reports/model/`.
5. **Preparar os entregáveis acadêmicos (E10):** relatório, slides e roteiro da
   demo, com os números recalculados pelo pipeline.
6. **Revisão final (E9):** executar o pipeline em ambiente limpo, rodar testes e
   lint e revisar os checklists da Seção 9.

---

## Apêndice A — Conferência do esboço do dashboard

O esboço da APRES, p. 10, é útil como referência visual, mas **não é um resultado validado**. A tabela abaixo compara os valores do esboço com a base MLG/ULB real, conferida em 17/11/2026.

| Item do esboço | Valor no esboço | Conferência na MLG/ULB | Conclusão |
|---|---|---|---|
| Total de transações | 284.907 | 284.807 | Provável erro de digitação; o donut soma 284.315 + 492 = 284.807. |
| Fraudes | 492 | 492 | Correto. |
| Taxa de fraude | 0,17% | 0,1727% | Correto com arredondamento. |
| Não fraudes | 284.315 | 284.315 | Correto. |
| Valor movimentado | R$ 25 Mi | 25.162.590,01 na unidade original | Ordem de grandeza compatível, mas a moeda “R$” não está justificada; a base é de cartões europeus. |
| Fraudes por faixa (0–10, 10–50, 50–100, 100–250, 250–500, 500–1k, 1k+) | 130, 70, 45, 100, 70, 55, 22 | 249, 56, 57, 56, 39, 26, 9 | O gráfico do esboço não corresponde à distribuição real. Não reutilizar. |
| Insight “70% das fraudes abaixo de R$ 250” | 70% | 85,0% das fraudes têm valor ≤ 250; 73,6% ≤ 100 | O insight do esboço não confere. Recalcular. |
| Métricas do modelo | 99% acurácia, 94% precisão, 92% recall, 93% F1 | Não validadas | O esboço diz “ilustrativo”; recalcular e documentar. |
| Matriz de confusão | TN 56.858; FP 6; FN 8; TP 90 | Consistente com um teste de ~20% (56.962 linhas, 98 fraudes), mas não validada | Pode ser um resultado real ou uma simulação; não usar sem reprodução. |
| Importância das variáveis | V14 18%, V10 11%, V17 10% | Não validada | O esboço diz “valores ilustrativos”; recalcular. |
| Filtro “Tipo de transação” | Presente | Inexistente na MLG/ULB | Remover ou trocar de base. |
| Período “01/01/2024–30/04/2025” | Presente | Inexistente; a base cobre dois dias de 2013 em tempo relativo | Corrigir o eixo; não inventar datas. |
| Heatmap “0h–21h” | Presente | `Time` é relativo; a origem não é meia-noite documentada | Declarar a premissa ou usar blocos relativos. |
| “Risco alto: bloquear e investigar” | Presente | Não é uma política do projeto | Tratar como ilustração de UI, não como funcionalidade. |

**Regra:** o esboço serve para layout e inspiração. Nenhum número, insight ou métrica dele deve ir para o relatório sem recálculo e documentação.

---

## Apêndice B — Dicionário de dados resumido

### B.1 MLG/ULB — *Credit Card Fraud Detection*

| Coluna | Tipo | Descrição |
|---|---|---|
| `Time` | numérico | Segundos desde a primeira transação da base; não é data/hora absoluta. |
| `V1`–`V28` | numérico | Componentes principais (PCA) anonimizados; significado original não divulgado. |
| `Amount` | numérico | Valor da transação. |
| `Class` | binário | 1 = fraude; 0 = legítima. |

### B.2 Sintética — *Credit Card Fraud Detection Dataset* (10K)

| Coluna | Tipo | Descrição |
|---|---|---|
| `transaction_id` | inteiro | Identificador da transação. |
| `amount` | numérico | Valor da transação. |
| `transaction_hour` | inteiro | Hora da transação (0–23). |
| `merchant_category` | categórico | Categoria do comerciante (5 valores). |
| `foreign_transaction` | binário | Transação internacional (0/1). |
| `location_mismatch` | binário | Incompatibilidade entre localização de cobrança e transação (0/1). |
| `device_trust_score` | numérico | Confiança do dispositivo (0–100). |
| `velocity_last_24h` | inteiro | Número de transações nas últimas 24 horas. |
| `cardholder_age` | inteiro | Idade do portador. |
| `is_fraud` | binário | 1 = fraude; 0 = legítima. |

### B.3 Give Me Some Credit

| Coluna | Descrição |
|---|---|
| `SeriousDlqin2yrs` | Pessoa ficou 90 dias ou mais em atraso em dois anos (alvo). |
| `RevolvingUtilizationOfUnsecuredLines` | Uso de crédito rotativo não garantido. |
| `age` | Idade. |
| `NumberOfTime30-59DaysPastDueNotWorse` | Número de atrasos de 30–59 dias. |
| `DebtRatio` | Dívida mensal dividida pela renda bruta. |
| `MonthlyIncome` | Renda mensal. |
| `NumberOfOpenCreditLinesAndLoans` | Linhas de crédito e empréstimos abertos. |
| `NumberOfTimes90DaysLate` | Número de atrasos de 90 dias ou mais. |
| `NumberRealEstateLoansOrLines` | Empréstimos imobiliários e linhas de crédito. |
| `NumberOfTime60-89DaysPastDueNotWorse` | Número de atrasos de 60–89 dias. |
| `NumberOfDependents` | Número de dependentes. |

---

## Apêndice C — Referências

- DAL POZZOLO, Andrea; CAELEN, Olivier; JOHNSON, Reid A.; BONTEMPI, Gianluca. **Credit Card Fraud Detection Dataset**. Kaggle, 2015. Disponível em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Acesso em: 17 nov. 2026.
- GÉRON, Aurélien. **Mãos à obra: aprendizado de máquina com Scikit-Learn, Keras e TensorFlow**. 2. ed. Rio de Janeiro: Alta Books, 2021.
- KAGGLE. **Give Me Some Credit**. Disponível em: https://www.kaggle.com/c/GiveMeSomeCredit. Acesso em: 17 nov. 2026.
- KAGGLE. **Credit Card Fraud Detection Dataset** (`miadul/credit-card-fraud-detection-dataset`). Disponível em: https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset. Acesso em: 17 nov. 2026.
- NILSON REPORT. **Payment Card Fraud Worldwide**. Edição 1298, jan. 2026. Disponível em: https://nilsonreport.com/newsletters/1298/. Acesso em: 17 nov. 2026.
- OLIVEIRA, Vinicius Dias. Métodos de machine learning na detecção de fraude em cartão de crédito: um estudo comparado. **Revista Científica ACERTTE**, v. 5, n. 9, 2025. Disponível em: https://acertte.org/acertte/article/view/265. Acesso em: 17 nov. 2026.
- PROVOST, Foster; FAWCETT, Tom. **Data Science para negócios: o que você precisa saber sobre mineração de dados e pensamento analítico de dados**. Rio de Janeiro: Alta Books, 2016.
- SANTOS, Aline Bertolazo dos; SANTOS, Isadora Disposti Bueno dos; TOMA, Karina Mitiko. **Detecção de transações fraudulentas com cartão de crédito**. Disponível em: https://ojs.focopublicacoes.com.br/foco/article/view/8328. Acesso em: 17 nov. 2026.
- SENAC. **Introdução à Ciência de Dados — Webconferência 2** (`aula2.pdf`). Material da disciplina, 2026.
- SENAC. **Códigos da Aula 2** (`Codigos_Aula2.zip`). Material da disciplina, 2026.
