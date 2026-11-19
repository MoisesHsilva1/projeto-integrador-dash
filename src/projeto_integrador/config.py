"""Configurações centrais: caminhos, sementes e constantes do projeto."""

from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

DIRETORIO_DADOS = RAIZ_PROJETO / "data"
CAMINHO_BRUTO = DIRETORIO_DADOS / "raw" / "creditcard.csv"
CAMINHO_PROCESSADO = DIRETORIO_DADOS / "processed" / "transacoes.parquet"

DIRETORIO_RELATORIOS = RAIZ_PROJETO / "reports"
DIRETORIO_FIGURAS = DIRETORIO_RELATORIOS / "figures"
DIRETORIO_MODELO = DIRETORIO_RELATORIOS / "model"
CAMINHO_MODELO = DIRETORIO_MODELO / "modelo_final.joblib"
CAMINHO_METRICAS = DIRETORIO_MODELO / "metricas.json"

SEMENTE = 42
COLUNA_ALVO = "Class"
COLUNA_VALOR = "Amount"
COLUNA_TEMPO = "Time"
COLUNAS_PCA = [f"V{indice}" for indice in range(1, 29)]
COLUNAS_MODELO = [COLUNA_TEMPO, *COLUNAS_PCA, COLUNA_VALOR]

PROPORCAO_TESTE = 0.2
NUMERO_FOLDS = 3
