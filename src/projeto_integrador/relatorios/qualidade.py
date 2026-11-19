"""Relatório de qualidade da base (etapa E3)."""

from __future__ import annotations

from pathlib import Path

from projeto_integrador.relatorios.texto import (
    formatar_inteiro,
    formatar_numero,
    formatar_percentual,
)


def gerar_relatorio_qualidade(
    auditoria: dict[str, object],
    registro: dict[str, object],
    *,
    checksum: str,
    caminho: Path,
) -> Path:
    """Escreve o relatório de qualidade em Markdown e devolve o caminho."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    remocoes = registro["remocoes"]
    linhas_remocao = "\n".join(
        f"| {item['regra']} | {formatar_inteiro(item['linhas_removidas'])} | "
        f"{formatar_inteiro(item['fraudes_removidas'])} |"
        for item in remocoes
    )
    mantidos = "\n".join(f"- {item}" for item in registro["mantidos"])
    tempo_intervalo = (
        f"{formatar_numero(auditoria['inicio_segundos'])} / "
        f"{formatar_numero(auditoria['fim_segundos'])}"
    )
    valor_intervalo = (
        f"{formatar_numero(auditoria['valor_minimo'])} / "
        f"{formatar_numero(auditoria['valor_maximo'])}"
    )
    valor_zero = (
        f"{formatar_inteiro(auditoria['valor_zero_transacoes'])} "
        f"(fraudes: {formatar_inteiro(auditoria['valor_zero_fraudes'])})"
    )

    conteudo = f"""# Relatório de qualidade dos dados

**Base:** MLG/ULB — *Credit Card Fraud Detection* (`data/raw/creditcard.csv`)

**MD5 do arquivo bruto:** `{checksum}`

## 1. Dimensões e alvo

| Verificação | Resultado |
|---|---|
| Linhas | {formatar_inteiro(auditoria["linhas"])} |
| Colunas | {formatar_inteiro(auditoria["colunas"])} |
| Valores ausentes | {formatar_inteiro(auditoria["valores_ausentes"])} |
| Transações legítimas | {formatar_inteiro(auditoria["legitimas"])} |
| Transações fraudulentas | {formatar_inteiro(auditoria["fraudes"])} |
| Taxa de fraude | {formatar_percentual(auditoria["taxa_fraude"])} |

## 2. Qualidade

| Verificação | Resultado |
|---|---|
| Duplicatas exatas | {formatar_inteiro(auditoria["duplicatas_exatas"])} |
| Fraudes entre as duplicatas | {formatar_inteiro(auditoria["duplicatas_fraudes"])} |
| `Time` mínimo / máximo (s) | {tempo_intervalo} |
| Duração coberta | {formatar_numero(auditoria["duracao_horas"], 1)} horas |
| `Amount` mínimo / máximo | {valor_intervalo} |
| `Amount` mediano | {formatar_numero(auditoria["valor_mediano"])} |
| `Amount` total | {formatar_numero(auditoria["valor_total"])} (unidade original) |
| Transações com `Amount` = 0 | {valor_zero} |

## 3. Tratamento aplicado

| Regra | Linhas removidas | Fraudes removidas |
|---|---|---|
{linhas_remocao}

**Resultado após a limpeza:** {formatar_inteiro(registro["linhas_finais"])} linhas,
{formatar_inteiro(registro["fraudes_finais"])} fraudes
({formatar_percentual(registro["taxa_fraude_final"])}).

**Itens mantidos deliberadamente:**

{mantidos}

A base tratada é gravada em `data/processed/transacoes.parquet` e não substitui o
arquivo bruto em `data/raw/`.

## 4. Limitações

- A base cobre apenas dois dias de transações de cartões europeus (setembro de 2013).
- `Time` é relativo à primeira transação; não há data/hora absoluta.
- `V1`–`V28` são componentes PCA anonimizados, sem significado de negócio divulgado.
- A moeda não está fixada no dicionário; os valores são exibidos na unidade original.
- A remoção de duplicatas também remove fraudes; por isso o número de positivos
  cai em relação ao arquivo bruto. O impacto está registrado acima.
"""

    caminho.write_text(conteudo, encoding="utf-8")
    return caminho
