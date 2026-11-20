"""Data quality report (step E3)."""

from __future__ import annotations

from pathlib import Path

from credit_card_fraud.reports.text import (
    format_integer,
    format_number,
    format_percentage,
)


def generate_quality_report(
    audit: dict[str, object],
    record: dict[str, object],
    *,
    checksum: str,
    path: Path,
) -> Path:
    """Write the quality report in Markdown and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    removals = record["removals"]
    removal_rows = "\n".join(
        f"| {item['rule']} | {format_integer(item['rows_removed'])} | "
        f"{format_integer(item['frauds_removed'])} |"
        for item in removals
    )
    kept = "\n".join(f"- {item}" for item in record["kept"])
    time_interval = (
        f"{format_number(audit['start_seconds'])} / "
        f"{format_number(audit['end_seconds'])}"
    )
    amount_interval = (
        f"{format_number(audit['min_amount'])} / {format_number(audit['max_amount'])}"
    )
    zero_amount = (
        f"{format_integer(audit['zero_amount_transactions'])} "
        f"(fraudes: {format_integer(audit['zero_amount_frauds'])})"
    )

    content = f"""# Relatório de qualidade dos dados

**Base:** MLG/ULB — *Credit Card Fraud Detection* (`data/raw/creditcard.csv`)

**MD5 do arquivo bruto:** `{checksum}`

## 1. Dimensões e alvo

| Verificação | Resultado |
|---|---|
| Linhas | {format_integer(audit["rows"])} |
| Colunas | {format_integer(audit["columns"])} |
| Valores ausentes | {format_integer(audit["missing_values"])} |
| Transações legítimas | {format_integer(audit["legitimate"])} |
| Transações fraudulentas | {format_integer(audit["frauds"])} |
| Taxa de fraude | {format_percentage(audit["fraud_rate"])} |

## 2. Qualidade

| Verificação | Resultado |
|---|---|
| Duplicatas exatas | {format_integer(audit["exact_duplicates"])} |
| Fraudes entre as duplicatas | {format_integer(audit["duplicate_frauds"])} |
| `Time` mínimo / máximo (s) | {time_interval} |
| Duração coberta | {format_number(audit["duration_hours"], 1)} horas |
| `Amount` mínimo / máximo | {amount_interval} |
| `Amount` mediano | {format_number(audit["median_amount"])} |
| `Amount` total | {format_number(audit["total_amount"])} (unidade original) |
| Transações com `Amount` = 0 | {zero_amount} |

## 3. Tratamento aplicado

| Regra | Linhas removidas | Fraudes removidas |
|---|---|---|
{removal_rows}

**Resultado após a limpeza:** {format_integer(record["final_rows"])} linhas,
{format_integer(record["final_frauds"])} fraudes
({format_percentage(record["final_fraud_rate"])}).

**Itens mantidos deliberadamente:**

{kept}

A base tratada é gravada em `data/processed/transactions.parquet` e não substitui o
arquivo bruto em `data/raw/`.

## 4. Limitações

- A base cobre apenas dois dias de transações de cartões europeus (setembro de 2013).
- `Time` é relativo à primeira transação; não há data/hora absoluta.
- `V1`–`V28` são componentes PCA anonimizados, sem significado de negócio divulgado.
- A moeda não está fixada no dicionário; os valores são exibidos na unidade original.
- A remoção de duplicatas também remove fraudes; por isso o número de positivos
  cai em relação ao arquivo bruto. O impacto está registrado acima.
"""

    path.write_text(content, encoding="utf-8")
    return path
