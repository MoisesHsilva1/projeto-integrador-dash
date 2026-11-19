# AGENTS.md

Contexto operacional do **Projeto Integrador do Grupo 54 — Centro Universitário SENAC**.

> **Leia primeiro:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). Ele é o plano técnico e funcional detalhado, com requisitos, análise das bases, etapas, riscos e pendências. Este arquivo resume o que um agente precisa saber para trabalhar sem reconstruir o contexto.

---

## Status atual

- **Fase:** implementação do MVP concluída (E2–E8) sobre a base MLG/ULB. Análise, documentação e pipeline existem no repositório; as pendências acadêmicas (rubrica, prazos, formato de entrega e política de IA) continuam abertas.
- **Última atualização:** 18/11/2026.
- **Código existente:** pipeline completo em `src/projeto_integrador/` (Data Access, Analytics, Application, Features/Models e Presentation), relatórios em `reports/` e dashboard Streamlit. O modelo avaliação atingiu AUPRC 0,7992 no holdout; esses números vêm do pipeline, não do mockup.
- **Decisões da implementação:** registradas em [`docs/DECISIONS.md`](docs/DECISIONS.md) (D-10 a D-25). A base MLG/ULB foi adotada, duplicatas exatas removidas, tempo tratado em blocos relativos e moeda mantida na unidade original.
- **Regra:** se uma decisão nova não estiver registrada nos materiais, no plano ou em `docs/DECISIONS.md`, trate como **pendência**, não como requisito.

---

## Contexto do projeto

### Tema

Detecção de **fraudes em transações financeiras realizadas com cartão de crédito**, usando análise de dados e Machine Learning para identificar padrões e comportamentos suspeitos.

### Problema

Identificar quais transações podem ser fraudulentas, diferenciando-as das legítimas e buscando padrões que indiquem possíveis operações fraudulentas.

### Objetivo geral

Criar um **dashboard interativo** para analisar transações com cartão de crédito, compreender características e comportamentos relacionados a fraudes e apoiar uma análise preditiva e eficiente.

### Objetivos específicos

1. Identificar padrões e características das transações fraudulentas.
2. Analisar a ocorrência de fraudes de acordo com o **valor** das transações.
3. Analisar a ocorrência de fraudes ao longo do **tempo** (horários, períodos, heatmap).

### Entregas mínimas identificadas

- tratamento e preparação dos dados;
- análise exploratória;
- dashboard interativo com, no mínimo, **quatro visualizações**: indicadores gerais, distribuição de fraudes, fraudes por faixa de valor e evolução temporal;
- aplicação de Machine Learning para prever possíveis fraudes;
- avaliação do modelo e identificação dos fatores associados;
- relatório/apresentação acadêmica.

### Contexto de negócio

Perdas globais com fraude de cartões chegaram a **US$ 33,41 bilhões em 2024** (Nilson Report, edição 1298). O dashboard é uma ferramenta de **apoio à decisão**, não um sistema antifraude de produção com bloqueio automático.

### Escopo

**Dentro:** base histórica de transações, EDA, classificação supervisionada, avaliação, dashboard, documentação.

**Fora (presumido; confirmar):** integração com sistemas de pagamento, scoring em tempo real, bloqueio automático, deploy em produção, dados pessoais identificáveis, deep learning/reforço sem justificativa.

### Público-alvo

**Não definido nos materiais.** Não assuma que são clientes finais. A hipótese de trabalho é equipes de análise de risco/fraude e gestores, mas isso precisa ser confirmado pelo grupo.

### Arquitetura

A arquitetura **alvo proposta** — ainda em planejamento, sem código — organiza o caminho de serviço em cinco blocos: dataset preparado, `Data Access`, `Analytics/Domain`, `Application` e `Presentation`, culminando no `Interactive Dashboard`. As dependências descem apenas nessa ordem; a Presentation não lê CSVs nem acessa Data Access diretamente, e Analytics/Domain é a fonte única de métricas, agregações e regras analíticas para notebooks e dashboard. O treino/avaliação de modelos (E5–E7) é trilha offline: seus artefatos são consumidos por Analytics/Domain, sem criar uma camada “ML” no caminho de serviço. A base (P-01) e a ferramenta (P-04) seguem pendentes, então o desenho é agnóstico a ambas. Referência completa: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Referências acadêmicas

| Fonte | Conteúdo relevante |
|---|---|
| `aula2.pdf` | Ciclo de vida de ciência de dados: entender o problema → coleta → pré-processamento → EDA → construção de modelos. Pré-processamento: limpeza, amostragem, desbalanceamento, transformação. EDA: estatísticas descritivas, histogramas, boxplots, dispersão, heatmaps, correlação. Modelos: supervisionado (classificação/regressão, árvore de decisão), não supervisionado e reforço. |
| `Codigos_Aula2.zip` | `Aula2.ipynb` (pandas, limpeza, datas, `groupby`, `merge`, `explode`, matplotlib/seaborn) e `Extra_Aula2.ipynb` (`value_counts`, gráficos de barras/pizza). Usar como referência de nível técnico, não como molde. |
| `grupo-54.pptx.pdf` | Fonte principal do escopo: problema, objetivos, próximos passos, esboço do dashboard e referências. |

**Conceitos que devem aparecer no projeto:** ciclo de vida, qualidade de dados, desbalanceamento, transformação, EDA, classificação supervisionada, avaliação por métricas adequadas, reprodutibilidade.

**Conceitos que não devem ser adicionados sem necessidade:** aprendizado por reforço, deep learning, clusters/segmentação, sistemas em tempo real, arquitetura de microsserviços.

---

## Dados

### Situação da decisão

A base **MLG/ULB** foi adotada como base única do projeto (D-10). O arquivo bruto
está em `data/raw/creditcard.csv`. A base sintética 10K e a *Give Me Some Credit*
não são usadas no pipeline.

| Base | Situação | Adequação |
|---|---|---|
| **MLG/ULB — Credit Card Fraud Detection** | **Adotada**; pipeline executado. | Base principal, com ressalvas documentadas. |
| **Sintética 10K — `miadul/credit-card-fraud-detection-dataset`** | Não usada. | Alternativa descartada nesta implementação. |
| **Give Me Some Credit** | Não usada. | Base de inadimplência, inadequada ao problema. |

**Não misture as bases.** A decisão está registrada em
[`docs/DECISIONS.md`](docs/DECISIONS.md) (D-10).

### MLG/ULB — fatos verificados

- 284.807 transações; 31 colunas: `Time`, `V1`–`V28`, `Amount`, `Class`.
- 492 fraudes (**0,1727%**).
- Dois dias de transações de cartões europeus (setembro de 2013).
- `Time` = segundos desde a primeira transação; **não é data/hora absoluta**.
- `V1`–`V28` = componentes PCA anonimizados; o significado de negócio não foi divulgado.
- Sem valores ausentes.
- 1.081 duplicatas exatas, incluindo **19 fraudes**; decidir e documentar antes de remover.
- `Amount` total = 25.162.590,01; 1.825 transações com valor zero (27 fraudes).
- A recomendação oficial é avaliar com **AUPRC**, não com acurácia.

### Sintética 10K — fatos verificados

- 10.000 transações; 10 colunas: `transaction_id`, `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, `is_fraud`.
- 151 fraudes (**1,51%**) no arquivo, embora a página da base declare 4–5%.
- Sem nulos e sem duplicatas exatas na inspeção.
- 24 horas, 5 categorias de comerciante, atributos interpretáveis.
- Base **sintética** (Apache 2.0, criada em dezembro de 2025). Não sustenta afirmações sobre fraude real.

### Give Me Some Credit — por que não usar

- 150.000 registros de treino + 101.503 de teste; alvo `SeriousDlqin2yrs` (inadimplência grave em dois anos).
- 10 variáveis de crédito; sem transação, valor de compra, comerciante, horário ou rótulo de fraude.
- É um problema de **risco de crédito**, não de fraude. Não há chave para cruzar com as outras bases.

### Onde inserir os arquivos

A área `data/` já existe e está protegida pelo `.gitignore`. Consulte `data/README.md`.

| Base | Caminho esperado |
|---|---|
| MLG/ULB | `data/raw/creditcard.csv` |
| Sintética 10K | `data/raw/credit_card_fraud_10k.csv` |
| Give Me Some Credit | `data/raw/cs-training.csv` (e `cs-test.csv`, se necessário) |

Coloque apenas a base escolhida. Não edite o arquivo original; gere cópias tratadas em `data/interim/` ou `data/processed/`.

### Cuidados obrigatórios

- Não versionar dados brutos ou processados no Git. A área `data/` já está no `.gitignore`; mantenha assim.
- Registrar URL, data de acesso, versão, licença e checksum.
- Separar treino/teste **antes** de qualquer transformação que aprenda parâmetros.
- Tratar desbalanceamento apenas no treino.
- Não remover outliers automaticamente: fraudes podem estar nos extremos.
- Não inventar hora do dia, data absoluta ou moeda. Na MLG/ULB, `Time` é relativo e a moeda não está fixada como “R$”.
- Não usar os números do mockup como resultados.

---

## Decisões, recomendações e pendências

### Decisões já registradas

- Tema, problema e objetivo geral (APRES).
- Três objetivos específicos (APRES), com correção pendente da redação duplicada do objetivo 2.
- Dashboard com no mínimo quatro visualizações (APRES).
- Aplicação de Machine Learning (APRES).
- Base de fraude do Kaggle (APRES) — **qual base ainda é pendência**.
- Ciclo de vida da AULA2 como diretriz metodológica.
- Documentação e código em português; interface do dashboard em português brasileiro (D-26).

### Recomendações

- Usar **MLG/ULB** como base principal.
- Não usar **Give Me Some Credit**.
- Não misturar bases.
- Se usar a sintética, rotulá-la como sintética em todos os lugares.
- Adotar **AUPRC** como métrica principal; reportar precision, recall, F1, ROC-AUC e matriz de confusão.
- Comparar um baseline com pelo menos um modelo candidato; Random Forest não é decisão final.
- Usar **Streamlit** como candidato padrão para o dashboard, salvo exigência contrária.
- Fazer o MVP com as quatro visualizações antes de incrementos.
- Remover ou redefinir o filtro “tipo de transação” na MLG/ULB.
- Corrigir moeda e eixo temporal no dashboard.
- Confirmar rubrica, prazos e formato de entrega com o professor.

### Pendências críticas (não implementar sem resolver)

1. **P-01/P-02:** MLG/ULB ou sintética? O link fornecido aponta para um notebook da sintética, não para a MLG/ULB. Confirmar a intenção.
2. **P-03:** qual é o público-alvo?
3. **P-04:** qual ferramenta de dashboard?
4. **P-05:** a aba de predição faz parte do MVP?
5. **P-06/P-07:** quais modelos, limite de decisão e estratégia de desbalanceamento?
6. **P-08:** o filtro “tipo de transação” será removido ou a base será trocada?
7. **P-09:** como tratar o tempo na MLG/ULB?
8. **P-10:** qual moeda usar nos gráficos?
9. **P-11:** formato, prazos e rubrica acadêmica.
10. **P-12/P-13:** papéis do grupo e armazenamento/versionamento dos dados.
11. **P-14:** política de citação e uso de IA.
12. **P-15:** os números do mockup são apenas ilustrativos.

Consulte a Seção 10 do plano para o encaminhamento de cada item.

---

## Diretrizes de desenvolvimento

### Abordagem

Seguir o ciclo da AULA2 e a sequência do plano:

```text
E1 Escopo → E2 Dados → E3 Auditoria/limpeza → E4 EDA → E5 Features/split
→ E6 Modelagem → E7 Avaliação → E8 Dashboard → E9 Integração/validação
→ E10 Entregáveis acadêmicos
```

A EDA pode voltar para a limpeza; a avaliação pode voltar para o treino; o dashboard pode exigir novos cortes na EDA.

### Organização esperada

- `data/raw/` — base bruta, imutável e não versionada; instruções em `data/README.md`.
- `data/interim/` — dados intermediários, não versionados.
- `data/processed/` — base tratada, não versionada.
- `notebooks/` — EDA e experimentos.
- `src/projeto_integrador/` — código reutilizável: dados, features, modelos, visualização e dashboard; organização em camadas proposta em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- `reports/figures/` — figuras geradas.
- `docs/` — plano, dicionário de dados e decisões.
- `tests/` — testes.

Não criar estrutura vazia nem módulos sem necessidade. O projeto é acadêmico: simplicidade e explicabilidade valem mais que arquitetura de produção.

### Boas práticas

- Manter o código em Python ≥3.11, com `uv`.
- Usar `pandas`, `numpy`, `matplotlib`, `seaborn` e `scikit-learn` quando necessário; não adicionar dependências antes da decisão.
- Sementes fixas; pipeline reproduzível; versões registradas.
- Separar treino/teste antes de ajustar transformações.
- Evitar lógica duplicada entre notebook e dashboard.
- Reportar métricas adequadas ao desbalanceamento; acurácia não é suficiente.
- Documentar cada transformação e cada linha removida.
- Não apresentar dados sintéticos, simulações ou valores do mockup como resultados reais.
- Manter documentação, identificadores e comentários de código em português; a interface do dashboard (KPIs, filtros, gráficos, tabelas, navegação, mensagens e estados vazios/de erro) em português brasileiro, conforme D-26. Não traduzir valores brutos da base.

### Critérios de qualidade

- **Dados:** proveniência, dicionário, qualidade, impacto das remoções.
- **Modelo:** baseline, comparação, validação cruzada, AUPRC, precision/recall/F1, matriz de confusão, limite justificado.
- **Dashboard:** quatro visualizações obrigatórias, filtros viáveis, números consistentes, rótulos corretos, execução local documentada.
- **Acadêmico:** rastreabilidade, fontes, limitações, reprodutibilidade e aderência à rubrica.

### Documentação

- `docs/PROJECT_PLAN.md` é o roadmap; mantenha-o atualizado quando uma decisão for tomada.
- Registre decisões novas na Seção 10 do plano.
- Crie `docs/DATA_DICTIONARY.md` na etapa de dados e um registro de decisões se o grupo achar útil.
- Atualize o README com instruções de execução quando o pipeline existir.

### Controle de escopo

O MVP implementado inclui dashboard, treino de modelos e pipeline de dados.
Ainda **não** implemente, sem necessidade identificada e justificada:

- download automatizado de dados;
- notebooks de análise (os relatórios são gerados por código);
- deploy;
- aba de predição individual (D-18/P-05);
- funcionalidades de produção.

Também não adicione deep learning, reforço, microsserviços ou banco de dados sem uma necessidade identificada e justificada.

---

## Orientações para o agente

1. **Leia o plano antes de propor alterações relevantes.** Consulte `docs/PROJECT_PLAN.md`, especialmente as Seções 3, 4, 7 e 10.
2. **Respeite os requisitos acadêmicos.** O ciclo de vida da AULA2 e a apresentação do grupo são as fontes de escopo.
3. **Não invente requisitos, dados, métricas, prazos ou resultados.** Se a informação não estiver nos materiais, registre como pendência.
4. **Diferencie** requisito explícito, intenção do mockup, recomendação técnica e decisão do grupo.
5. **Justifique decisões técnicas importantes** com o problema, os dados e a literatura.
6. **Evite complexidade desnecessária.** Priorize soluções que o grupo consiga explicar e defender academicamente.
7. **Não use números do mockup como resultados.** O Apêndice A do plano mostra as inconsistências.
8. **Não misture bases.** Se a base sintética for usada, ela deve ser claramente rotulada.
9. **Consulte `docs/DECISIONS.md`** antes de reabrir decisões já tomadas (D-10 a D-25); pergunte antes de implementar algo que dependa de P-03, P-05, P-11, P-12, P-14 ou P-15.
10. **Mantenha a documentação atualizada** quando uma decisão for tomada.
11. **Faça mudanças pequenas, com testes e lint**, preservando o comportamento do entrypoint (`main.py`) e do scaffold.
12. **Não use números do mockup nem resultados de outra execução** sem recalcular e registrar a semente e a base.

---

## Comandos e scaffold

- `uv run pytest` — suíte completa; teste único: `uv run pytest tests/test_main.py::test_main`
- `uv run ruff check --fix .` — lint + autofix
- `uv run ruff format .` — formatação
- `uv run projeto-integrador` — executa o entrypoint (`src/projeto_integrador/main.py:main`)
- `uv run projeto-integrador-pipeline preparar|eda|treinar|tudo` — pipeline de dados, EDA e modelagem
- `uv run projeto-integrador-dashboard` — abre o dashboard Streamlit
- Primeira configuração: `uv venv && uv pip install -e ".[dev]"`

Sempre use `uv run` (mantém `.venv` sincronizado com `pyproject.toml`). Não há typecheck, pre-commit ou CI configurado.

### Gotchas

- Nome da distribuição: `projeto-integrador`; pacote de importação: `projeto_integrador` (underscore, em `src/`). O pytest adiciona `src` ao `sys.path` via `[tool.pytest.ini_options].pythonpath`.
- A versão está duplicada em `pyproject.toml` (`[project].version`) e `src/projeto_integrador/__init__.py` (`__version__`). Atualize as duas; `tests/test_main.py::test_versao` verifica o valor.
- A saída exata de `main()` é verificada em `tests/test_main.py::test_main`; altere o teste junto com a mensagem.
- Ruff é o único portão de qualidade: linha 88, alvo `py311`, regras `E,F,I,UP,B,SIM`. O VS Code formata/corrige ao salvar via `.vscode/settings.json`.

---

## Resumo em uma frase

O Grupo 54 implementou uma análise e um dashboard de fraude em cartão de crédito
com Machine Learning sobre a base MLG/ULB, seguindo o ciclo de vida da AULA2; o
MVP cobre as quatro visualizações obrigatórias, o modelo avaliado por AUPRC e a
documentação reprodutível, e as pendências restantes são acadêmicas (rubrica,
prazos, formato de entrega e uso de IA).
