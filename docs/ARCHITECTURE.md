# Arquitetura proposta do dashboard

**Grupo 54 — Centro Universitário SENAC**

- **Status:** proposta de arquitetura **implementada no MVP**. O caminho em camadas, as regras de dependência e a trilha offline de ML estão refletidos no código em `src/projeto_integrador/`, nos testes e nos relatórios gerados.
- **Objeto:** descrever o caminho pelo qual o dataset preparado alimenta o dashboard, sem antecipar decisões ainda pendentes.
- **Referência canônica:** este é o documento de referência do desenho em camadas. O resumo está na Seção 6.2.1 de [`PROJECT_PLAN.md`](PROJECT_PLAN.md); o contexto operacional está em [`../AGENTS.md`](../AGENTS.md).
- **Idioma:** a interface do dashboard é em inglês; documentação, identificadores e comentários de código permanecem em português, conforme **D-09** (Seção 10.1 do plano).

> **Atualização:** **P-01** foi resolvida pela **D-10** (base MLG/ULB) e **P-04** pela **D-15** (Streamlit). O desenho continua válido e foi implementado; a base sintética não é usada. Ver [`DECISIONS.md`](DECISIONS.md).

## 1. Fluxo canônico

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

O fluxo representa o **caminho de serviço do dashboard**. `Prepared Dataset (CSV)` é o artefato tratado e documentado na etapa E3, normalmente em `data/processed/`; nomes de colunas, unidades, moeda e eixo temporal dependem da base escolhida em P-01 e não são fixados aqui. O fluxo também não pressupõe nenhuma API de interface: a ferramenta será decidida em P-04.

## 2. Camadas

| Camada | Responsabilidade | Mapeamento proposto no repositório | Dependência a jusante permitida | Não deve |
|---|---|---|---|---|
| **Presentation Layer** | Renderizar páginas, filtros, KPIs, gráficos e tabelas e converter interações do usuário em solicitações para a Application Layer. | `src/projeto_integrador/presentation/` — implementado; absorve `viz/` e `dashboard/` da proposta anterior. | `Application Layer` | Não ler CSVs; não acessar `Data Access` diretamente; não calcular métricas ou agregações próprias; não conter regras analíticas; não depender de `models/` nem de biblioteca de ML. |
| **Application Layer** | Orquestrar filtros, expor serviços do dashboard e coordenar o estado da sessão; transformar interações em parâmetros para Analytics/Domain. | `src/projeto_integrador/application/` — implementado. | `Analytics / Domain` | Não calcular métricas nem agregações; não carregar, validar ou cachear dados; não implementar gráficos; não duplicar regras analíticas; não manipular arquivos. |
| **Analytics / Domain** | Ser a fonte única de verdade de métricas, agregações e regras analíticas, respondendo pelas mesmas definições em notebooks e no dashboard. | `src/projeto_integrador/analytics/` — implementado. | `Data Access` | Não ler arquivos diretamente; não implementar UI nem estado de sessão; não treinar modelos; não duplicar cálculo entre notebook e dashboard; não depender de framework de dashboard. |
| **Data Access** | Carregar o dataset preparado, validar o esquema esperado e gerenciar cache, isolando detalhes de entrada e saída. | `src/projeto_integrador/data/` — implementado com carga, validação de esquema, cache e auditoria. | `Prepared Dataset (CSV)` | Não calcular indicadores de negócio; não criar filtros de UI; não decidir regras analíticas; não editar `data/raw/`; não implementar regras de outra base. |
| **Prepared Dataset (CSV)** | Entrada imutável do fluxo: base tratada, com proveniência e dicionário registrados. | `data/processed/` — convenção existente, não versionada. | Nenhuma. | Não ser lido diretamente por Presentation, Application ou Analytics; não ser alterado pelo caminho de leitura; não conter lógica ou regras de negócio. |

## 3. Regras obrigatórias

1. As dependências fluem **apenas para baixo**: `Presentation → Application → Analytics/Domain → Data Access → Prepared Dataset`. Nenhuma camada depende de uma camada superior; os dados retornam para cima, mas as chamadas não invertem.
2. A **Presentation nunca lê CSVs** e **nunca acessa Data Access diretamente**. Toda entrada de dados passa por `Application → Analytics/Domain → Data Access`.
3. A **Data Access** é a única responsável por `loading`, `schema validation` e `caching`. O esquema validado deve ser parametrizável conforme a base decidida em P-01.
4. A **Application** é a única responsável por `filter orchestration`, `dashboard services` e `session state coordination`.
5. A **Analytics/Domain** é a única fonte de verdade de `metrics`, `aggregations` e `analytical rules`. Notebooks e dashboard devem consumir as mesmas funções ou artefatos, para que os números permaneçam consistentes (RNF-08 e risco R-16 do plano).
6. Nenhuma regra analítica pode ser duplicada em outra camada: todo número exibido no dashboard deve nascer em Analytics/Domain e ser rastreável até o dataset preparado.
7. O desenho não fixa biblioteca de UI, mecanismo de estado de sessão nem formato de cache; essas escolhas dependem de P-04 e de decisões posteriores.

## 4. Onde o Machine Learning se encaixa

O diagrama de cinco camadas é o **caminho de serviço do dashboard**. Ele não inclui uma camada “ML” no runtime, nem deve ganhar uma sexta camada com esse nome.

O treino e a avaliação de modelos (etapas **E5–E7**) formam uma **trilha offline**, mantida em `features/` e `models/`. Essa trilha produz artefatos — modelo ajustado, pipeline de features e métricas de avaliação — que são **consumidos pela camada Analytics/Domain**, que os publica ao restante do fluxo como métricas e regras analíticas. A aba de predição, se aprovada em P-05, deve reutilizar o mesmo pipeline de transformação do treino, sem reimplementação paralela.

Assim, a trilha offline alimenta a arquitetura, mas não altera a direção das dependências do caminho de serviço.

## 5. Mapeamento proposto para as pastas

O mapeamento abaixo refere-se à proposta de estrutura da Seção 6.3 de `PROJECT_PLAN.md` e foi implementado no MVP. O scaffold original (`src/projeto_integrador/main.py`, testes e configuração de Ruff/pytest) permanece funcionando.

| Camada / trilha | Pasta proposta | Situação em relação à proposta anterior (6.3) |
|---|---|---|
| **Prepared Dataset** | `data/processed/` | Já prevista como área de dados não versionada; permanece como entrada do fluxo. |
| **Data Access** | `src/projeto_integrador/data/` | Já proposta para “carga e validação”; passa a incluir explicitamente validação de esquema e cache. |
| **Analytics / Domain** | `src/projeto_integrador/analytics/` | **Diretório novo proposto** para centralizar métricas, agregações e regras analíticas. |
| **Application** | `src/projeto_integrador/application/` | **Diretório novo proposto** para filtros, serviços do dashboard e estado da sessão. |
| **Presentation** | `src/projeto_integrador/presentation/` | **Mudança proposta:** absorve `viz/` (gráficos reutilizáveis) e `dashboard/` (aplicação) da proposta anterior. |
| **Trilha offline de ML** | `src/projeto_integrador/features/` e `src/projeto_integrador/models/` | Mantidas com seus papéis anteriores (transformações, treino e avaliação). `features/` **não** é renomeada nem absorvida pela Presentation. |
| **Suporte** | `src/projeto_integrador/config.py` | Mantido para caminhos, sementes e constantes, sem introduzir dependência circular entre camadas. |

**Estrutura implementada:** `analytics/` e `application/` foram criados e `viz/` e `dashboard/` ficaram absorvidos por `presentation/`. A implementação segue as regras de dependência da Seção 3 e está coberta por testes.

## 6. Idioma e pendências

### 6.1 Idioma da interface

Após a decisão **D-26**, todo texto voltado ao usuário do dashboard está em **português brasileiro (pt-BR)**: rótulos de KPIs e filtros, títulos/eixos/legendas de gráficos, cabeçalhos de tabelas, navegação, mensagens e estados vazios ou de erro. A regra anterior de interface em inglês (D-09/D-17) foi revogada. Documentação, identificadores e comentários de código permanecem em português. A regra vale para rótulos de interface, não para valores brutos provenientes da base escolhida.

### 6.2 Pendências que limitam o desenho

**P-01** e **P-04** foram resolvidas por D-10 (MLG/ULB) e D-15 (Streamlit). O
esquema fixo da base está em `src/projeto_integrador/data/schema.py`, a unidade de
tempo relativa está documentada em `docs/DATA_DICTIONARY.md` e os componentes de
Presentation seguem os serviços de Application.
