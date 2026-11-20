"""Central settings: paths, seeds, and project constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "transactions.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_DIR = REPORTS_DIR / "model"
MODEL_PATH = MODEL_DIR / "model_final.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
IMBALANCE_PATH = MODEL_DIR / "imbalance.json"
EXPERIMENTS_PATH = MODEL_DIR / "experiments.json"
IMPORTANCES_PATH = MODEL_DIR / "importances_permutation.json"
ERRORS_PATH = MODEL_DIR / "errors_holdout.json"

SEED = 42
TARGET_COLUMN = "Class"
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"
PCA_COLUMNS = [f"V{index}" for index in range(1, 29)]
MODEL_COLUMNS = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]

TEST_SIZE = 0.2
N_FOLDS = 3

# Evaluation and experiments (offline track).
DEFAULT_THRESHOLD = 0.5
TARGET_RECALLS = (0.80, 0.90)
ROBUSTNESS_SEEDS = (0, 1, 42)
PERMUTATION_SAMPLE = 10_000
PERMUTATION_REPEATS = 10
BOOTSTRAP_SAMPLES = 1_000
N_EXPERIMENT_FOLDS = 2
