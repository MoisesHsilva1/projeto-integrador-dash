"""Linha de comando do pipeline (E2–E8)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from projeto_integrador.config import (
    CAMINHO_BRUTO,
    CAMINHO_DESBALANCEAMENTO,
    CAMINHO_METRICAS,
    CAMINHO_MODELO,
    CAMINHO_PROCESSADO,
    DIRETORIO_FIGURAS,
    DIRETORIO_RELATORIOS,
    NUMERO_FOLDS,
    PROPORCAO_TESTE,
    SEMENTE,
)
from projeto_integrador.data.auditoria import (
    auditar,
    preparar_transacoes,
    salvar_preparadas,
)
from projeto_integrador.data.loader import carregar_brutas, carregar_preparadas
from projeto_integrador.features.dataset import construir_matriz, dividir_treino_teste
from projeto_integrador.models.desbalanceamento import comparar_estrategias
from projeto_integrador.models.treino import salvar_modelo, treinar_e_avaliar
from projeto_integrador.relatorios.eda import gerar_relatorio_eda
from projeto_integrador.relatorios.modelo import gerar_relatorio_modelo
from projeto_integrador.relatorios.qualidade import gerar_relatorio_qualidade
from projeto_integrador.relatorios.texto import (
    formatar_inteiro,
    formatar_numero,
    formatar_percentual,
)


def calcular_md5(caminho: Path) -> str:
    """Calcula o MD5 do arquivo bruto em blocos."""
    resumo = hashlib.md5()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def preparar_dados() -> dict[str, object]:
    """E2/E3: audita, limpa, grava a base tratada e o relatório de qualidade."""
    brutas = carregar_brutas()
    auditoria = auditar(brutas)
    preparadas, registro = preparar_transacoes(brutas)
    salvar_preparadas(preparadas, CAMINHO_PROCESSADO)

    checksum = calcular_md5(CAMINHO_BRUTO)
    DIRETORIO_RELATORIOS.mkdir(parents=True, exist_ok=True)
    relatorio = gerar_relatorio_qualidade(
        auditoria,
        registro,
        checksum=checksum,
        caminho=DIRETORIO_RELATORIOS / "RELATORIO_QUALIDADE.md",
    )
    (DIRETORIO_RELATORIOS / "qualidade.json").write_text(
        json.dumps(
            {"checksum_md5": checksum, "auditoria": auditoria, "registro": registro},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Base preparada: "
        f"{formatar_inteiro(auditoria['linhas'])} -> "
        f"{formatar_inteiro(registro['linhas_finais'])} linhas; "
        f"fraudes: {formatar_inteiro(auditoria['fraudes'])} -> "
        f"{formatar_inteiro(registro['fraudes_finais'])} "
        f"({formatar_percentual(registro['taxa_fraude_final'])})."
    )
    print(f"Arquivo: {CAMINHO_PROCESSADO}")
    print(f"Relatório: {relatorio}")
    return auditoria


def gerar_eda() -> Path:
    """E4: gera figuras e relatório da análise exploratória."""
    transacoes = carregar_preparadas()
    relatorio = gerar_relatorio_eda(
        transacoes,
        diretorio_figuras=DIRETORIO_FIGURAS,
        caminho=DIRETORIO_RELATORIOS / "RELATORIO_EDA.md",
    )
    print(f"Relatório de EDA: {relatorio}")
    return relatorio


def treinar_modelos() -> dict[str, object]:
    """E5–E7: prepara atributos, compara modelos, avalia e salva artefatos."""
    transacoes = carregar_preparadas()
    atributos, alvo = construir_matriz(transacoes)
    atributos_treino, atributos_teste, alvo_treino, alvo_teste = dividir_treino_teste(
        atributos, alvo
    )

    resultado = treinar_e_avaliar(
        atributos_treino, alvo_treino, atributos_teste, alvo_teste
    )
    metadados: dict[str, object] = {
        "semente": SEMENTE,
        "folds": NUMERO_FOLDS,
        "proporcao_teste": PROPORCAO_TESTE,
        "colunas": list(atributos.columns),
        "linhas_treino": int(len(atributos_treino)),
        "linhas_teste": int(len(atributos_teste)),
        "fraudes_treino": int(alvo_treino.sum()),
        "fraudes_teste": int(alvo_teste.sum()),
        "modelo_selecionado": resultado["melhor_modelo"],
        "limite_decisao": resultado["limite_decisao"],
        "f1_out_of_fold_treino": resultado["f1_out_of_fold_treino"],
        "metricas_holdout": resultado["metricas_holdout"],
        "validacao_cruzada": resultado["validacao_cruzada"],
    }

    salvar_modelo(resultado["modelo"], metadados, CAMINHO_MODELO)
    CAMINHO_METRICAS.write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    relatorio = gerar_relatorio_modelo(
        resultado,
        atributos_teste,
        alvo_teste,
        metadados=metadados,
        diretorio_figuras=DIRETORIO_FIGURAS,
        caminho=DIRETORIO_RELATORIOS / "RELATORIO_MODELO.md",
    )

    metricas = resultado["metricas_holdout"]
    print(f"Modelo selecionado: {resultado['melhor_modelo']}")
    print(
        "Holdout: "
        f"AUPRC={formatar_numero(metricas['auprc'], 4)}; "
        f"F1={formatar_numero(metricas['f1'], 4)}; "
        f"recall={formatar_numero(metricas['recall'], 4)}; "
        f"precisão={formatar_numero(metricas['precisao'], 4)}."
    )
    print(f"Modelo: {CAMINHO_MODELO}")
    print(f"Métricas: {CAMINHO_METRICAS}")
    print(f"Relatório: {relatorio}")
    return metadados


def comparar_desbalanceamento() -> dict[str, dict[str, float]]:
    transacoes = carregar_preparadas()
    atributos, alvo = construir_matriz(transacoes)
    atributos_treino, _, alvo_treino, _ = dividir_treino_teste(atributos, alvo)

    resultados = comparar_estrategias(
        atributos_treino, alvo_treino, folds=NUMERO_FOLDS, semente=SEMENTE
    )

    CAMINHO_DESBALANCEAMENTO.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Estratégias de desbalanceamento (validação cruzada no treino):")
    for nome, valores in resultados.items():
        print(
            f"  {nome}: AUPRC={formatar_numero(valores['auprc_media'], 4)} "
            f"± {formatar_numero(valores['auprc_desvio'], 4)} "
            f"({formatar_numero(valores['tempo_treino_segundos'], 1)} s por ajuste)."
        )
    print(f"Arquivo: {CAMINHO_DESBALANCEAMENTO}")
    return resultados


def abrir_dashboard() -> None:
    """E8: inicia o dashboard Streamlit."""
    aplicativo = Path(__file__).parent / "presentation" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(aplicativo)],
        check=True,
    )


def executar_tudo() -> None:
    """Executa preparação, EDA e modelagem em sequência."""
    preparar_dados()
    gerar_eda()
    treinar_modelos()
    print(
        "Pipeline completo. Inicie o dashboard com: uv run projeto-integrador-dashboard"
    )


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada da linha de comando."""
    parser = argparse.ArgumentParser(
        prog="projeto-integrador-pipeline",
        description="Pipeline do Projeto Integrador (base MLG/ULB).",
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    subcomandos.add_parser(
        "preparar", help="auditar a base bruta e gerar a base tratada (E2/E3)"
    )
    subcomandos.add_parser("eda", help="gerar figuras e relatório da EDA (E4)")
    subcomandos.add_parser(
        "treinar", help="treinar, avaliar e salvar modelo e artefatos (E5–E7)"
    )
    subcomandos.add_parser("tudo", help="executar preparar, eda e treinar")
    subcomandos.add_parser(
        "desbalanceamento",
        help="comparar estratégias de desbalanceamento no treino",
    )
    subcomandos.add_parser("dashboard", help="abrir o dashboard Streamlit (E8)")

    argumentos = parser.parse_args(argv)

    comandos = {
        "preparar": preparar_dados,
        "eda": gerar_eda,
        "treinar": treinar_modelos,
        "tudo": executar_tudo,
        "desbalanceamento": comparar_desbalanceamento,
        "dashboard": abrir_dashboard,
    }

    try:
        comandos[argumentos.comando]()
    except FileNotFoundError as erro:
        raise SystemExit(f"Erro: {erro}") from erro


if __name__ == "__main__":
    main()
