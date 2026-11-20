"""Command-line pipeline (steps E2–E8)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from credit_card_fraud import __version__
from credit_card_fraud.config import (
    AMOUNT_COLUMN,
    ERRORS_PATH,
    EXPERIMENTS_PATH,
    FIGURES_DIR,
    IMBALANCE_PATH,
    IMPORTANCES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    N_EXPERIMENT_FOLDS,
    N_FOLDS,
    PERMUTATION_REPEATS,
    PERMUTATION_SAMPLE,
    PROCESSED_PATH,
    RAW_PATH,
    REPORTS_DIR,
    ROBUSTNESS_SEEDS,
    SEED,
    TEST_SIZE,
)
from credit_card_fraud.data.audit import audit, prepare_transactions, save_prepared
from credit_card_fraud.data.loader import (
    load_metrics,
    load_model_artifact,
    load_prepared,
    load_raw,
)
from credit_card_fraud.data.schema import SchemaError
from credit_card_fraud.features.dataset import build_matrix, split_train_test
from credit_card_fraud.models.evaluation import (
    confidence_intervals,
    error_summary,
    permutation_importance,
    threshold_table,
    value_impact,
)
from credit_card_fraud.models.experiments import (
    evaluate_resampling_holdout,
    hyperparameter_tuning,
    seed_stability,
    temporal_validation,
)
from credit_card_fraud.models.imbalance import compare_strategies
from credit_card_fraud.models.persistence import collect_environment
from credit_card_fraud.models.training import (
    MODEL_HYPERPARAMETERS,
    create_models,
    save_model,
    train_and_evaluate,
)
from credit_card_fraud.reports.eda import generate_eda_report
from credit_card_fraud.reports.model import generate_model_report
from credit_card_fraud.reports.quality import generate_quality_report
from credit_card_fraud.reports.text import (
    format_integer,
    format_number,
    format_percentage,
)


def calculate_md5(path: Path) -> str:
    """Calculate the MD5 of the raw file in blocks."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data() -> dict[str, object]:
    """E2/E3: audit, clean, write the prepared base and quality report."""
    raw = load_raw()
    audit_result = audit(raw)
    prepared, record = prepare_transactions(raw)
    save_prepared(prepared, PROCESSED_PATH)

    checksum = calculate_md5(RAW_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_quality_report(
        audit_result,
        record,
        checksum=checksum,
        path=REPORTS_DIR / "RELATORIO_QUALIDADE.md",
    )
    (REPORTS_DIR / "quality.json").write_text(
        json.dumps(
            {"checksum_md5": checksum, "audit": audit_result, "record": record},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Prepared base: "
        f"{format_integer(audit_result['rows'])} -> "
        f"{format_integer(record['final_rows'])} rows; "
        f"frauds: {format_integer(audit_result['frauds'])} -> "
        f"{format_integer(record['final_frauds'])} "
        f"({format_percentage(record['final_fraud_rate'])})."
    )
    print(f"File: {PROCESSED_PATH}")
    print(f"Report: {report}")
    return audit_result


def generate_eda() -> Path:
    """E4: generate figures and exploratory analysis report."""
    transactions = load_prepared()
    report = generate_eda_report(
        transactions,
        figures_directory=FIGURES_DIR,
        path=REPORTS_DIR / "RELATORIO_EDA.md",
    )
    print(f"EDA report: {report}")
    return report


def train_models() -> dict[str, object]:
    """E5–E7: build features, compare models, evaluate, and save artifacts."""
    transactions = load_prepared()
    features, target = build_matrix(transactions)
    features_train, features_test, target_train, target_test = split_train_test(
        features, target
    )

    result = train_and_evaluate(
        features_train, target_train, features_test, target_test
    )

    test_amounts = features_test[AMOUNT_COLUMN].reset_index(drop=True)
    threshold_table_result = threshold_table(
        target_test,
        result["test_probabilities"],
        result["decision_threshold"],
    )
    value_impact_result = value_impact(
        target_test,
        result["test_probabilities"],
        test_amounts,
        result["decision_threshold"],
    )
    baseline = result["baseline_holdout"]
    baseline_public = {
        key: value for key, value in baseline.items() if key != "test_probabilities"
    }

    metadata: dict[str, object] = {
        "package_version": __version__,
        "execution_date": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "folds": N_FOLDS,
        "test_proportion": TEST_SIZE,
        "columns": list(features.columns),
        "train_rows": int(len(features_train)),
        "test_rows": int(len(features_test)),
        "train_frauds": int(target_train.sum()),
        "test_frauds": int(target_test.sum()),
        "selected_model": result["best_model"],
        "decision_threshold": result["decision_threshold"],
        "f1_out_of_fold_train": result["f1_out_of_fold_train"],
        "holdout_metrics": result["holdout_metrics"],
        "cross_validation": result["cross_validation"],
        "baseline_holdout": baseline_public,
        "threshold_table": threshold_table_result,
        "value_impact": value_impact_result,
        "hyperparameters": MODEL_HYPERPARAMETERS,
        "environment": collect_environment(),
        "raw_base_checksum": calculate_md5(RAW_PATH),
        "prepared_base_checksum": calculate_md5(PROCESSED_PATH),
    }

    save_model(result["model"], metadata, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = generate_model_report(
        result,
        features_test,
        target_test,
        metadata=metadata,
        figures_directory=FIGURES_DIR,
        path=REPORTS_DIR / "RELATORIO_MODELO.md",
    )

    metrics = result["holdout_metrics"]
    print(f"Selected model: {result['best_model']}")
    print(
        "Holdout: "
        f"AUPRC={format_number(metrics['auprc'], 4)}; "
        f"F1={format_number(metrics['f1'], 4)}; "
        f"recall={format_number(metrics['recall'], 4)}; "
        f"precision={format_number(metrics['precision'], 4)}."
    )
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Report: {report}")
    return metadata


def compare_imbalance() -> dict[str, dict[str, float]]:
    """Compare imbalance strategies on training and record the result.

    Resampling is applied only within the training folds; the holdout does not
    take part. The pipeline strategy (class weights) is kept as reference (D-19).
    """
    transactions = load_prepared()
    features, target = build_matrix(transactions)
    features_train, _, target_train, _ = split_train_test(features, target)

    results = compare_strategies(features_train, target_train, folds=N_FOLDS, seed=SEED)

    IMBALANCE_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Imbalance strategies (cross-validation on training):")
    for name, values in results.items():
        print(
            f"  {name}: AUPRC={format_number(values['auprc_mean'], 4)} "
            f"± {format_number(values['auprc_std'], 4)} "
            f"({format_number(values['train_time_seconds'], 1)} s per fit)."
        )
    print(f"File: {IMBALANCE_PATH}")
    return results


RANDOM_FOREST_GRID: dict[str, list[object]] = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [12, 16, 20],
    "classifier__min_samples_leaf": [5, 10, 20],
}


def _importances_to_json(importances: object) -> dict[str, object]:
    """Serialize the permutation importance DataFrame."""
    return {
        "method": "permutation_importance",
        "sample": PERMUTATION_SAMPLE,
        "repeats": PERMUTATION_REPEATS,
        "attributes": [
            {
                "attribute": str(name),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
            for name, row in importances.iterrows()  # type: ignore[attr-defined]
        ],
    }


def run_experiments() -> dict[str, object]:
    """Run robustness, explainability, and sensitivity evaluations.

    The holdout is used only to describe the final model and sensitivities;
    selection remains on training cross-validation.
    """
    transactions = load_prepared()
    features, target = build_matrix(transactions)
    features_train, features_test, target_train, target_test = split_train_test(
        features, target
    )

    artifact = load_model_artifact()
    if artifact is None:
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Run the 'train' step before 'experiments'."
        )
    model = artifact["model"]
    artifact_metadata = artifact["metadata"]
    threshold = float(artifact_metadata["decision_threshold"])
    model_name = str(artifact_metadata.get("selected_model", "Random Forest"))

    test_probabilities = model.predict_proba(features_test)[:, 1]
    test_amounts = features_test[AMOUNT_COLUMN].to_numpy()

    importances = permutation_importance(
        model,
        features_test,
        target_test,
        n_repeats=PERMUTATION_REPEATS,
        sample=PERMUTATION_SAMPLE,
        seed=SEED,
    )
    importances_json = _importances_to_json(importances)
    errors = error_summary(
        target_test,
        test_probabilities,
        test_amounts,
        threshold,
        index=features_test.index,
    )
    confidence_intervals_result = confidence_intervals(
        target_test, test_probabilities, threshold
    )

    current_model = create_models(SEED)[model_name]
    temporal_validation_result = temporal_validation(
        features, target, current_model, folds=N_FOLDS, seed=SEED
    )
    seed_stability_result = seed_stability(
        features_train,
        target_train,
        lambda seed: create_models(seed)[model_name],
        seeds=ROBUSTNESS_SEEDS,
        folds=N_EXPERIMENT_FOLDS,
    )
    tuning = hyperparameter_tuning(
        features_train,
        target_train,
        create_models(SEED)[model_name],
        RANDOM_FOREST_GRID,
        n_iter=4,
        folds=N_EXPERIMENT_FOLDS,
        seed=SEED,
    )
    tuning.pop("fitted_model", None)

    oversampling = evaluate_resampling_holdout(
        features_train,
        target_train,
        features_test,
        target_test,
        create_models(SEED)[model_name],
        mode="over",
        folds=N_EXPERIMENT_FOLDS,
        seed=SEED,
    )
    undersampling = evaluate_resampling_holdout(
        features_train,
        target_train,
        features_test,
        target_test,
        create_models(SEED)[model_name],
        mode="under",
        folds=N_EXPERIMENT_FOLDS,
        seed=SEED,
    )

    experiments: dict[str, object] = {
        "confidence_intervals": confidence_intervals_result,
        "temporal_validation": temporal_validation_result,
        "seed_stability": seed_stability_result,
        "hyperparameter_tuning": tuning,
        "imbalance_holdout": {
            "oversampling": oversampling,
            "undersampling": undersampling,
        },
    }

    EXPERIMENTS_PATH.write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    IMPORTANCES_PATH.write_text(
        json.dumps(importances_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ERRORS_PATH.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = load_metrics() or {}
    report_metadata = dict(metrics)
    report_metadata.update(
        {
            "experiments": experiments,
            "permutation_importances": importances_json,
            "holdout_errors": errors,
        }
    )
    outcome: dict[str, object] = {
        "cross_validation": metrics.get("cross_validation", {}),
        "selected_model": metrics.get("selected_model", model_name),
        "decision_threshold": metrics.get("decision_threshold", threshold),
        "holdout_metrics": metrics.get("holdout_metrics", {}),
        "baseline_holdout": metrics.get("baseline_holdout", {}),
        "model": model,
        "test_probabilities": test_probabilities,
    }
    report = generate_model_report(
        outcome,  # type: ignore[arg-type]
        features_test,
        target_test,
        metadata=report_metadata,
        figures_directory=FIGURES_DIR,
        path=REPORTS_DIR / "RELATORIO_MODELO.md",
    )

    print("Experiments concluded:")
    print(f"  AUPRC (current model, CV): {format_number(tuning['auprc_mean'], 4)}")
    print(f"  Search time: {format_number(tuning['search_time_seconds'], 1)} s")
    temporal_auprc = temporal_validation_result["metrics"]["auprc"]
    print(f"  Temporal (day 1 -> day 2): AUPRC={format_number(temporal_auprc, 4)}")
    oversampling_auprc = oversampling["metrics"]["auprc"]
    print(f"  Oversampling on holdout: AUPRC={format_number(oversampling_auprc, 4)}")
    print(f"  Files: {EXPERIMENTS_PATH}")
    print(f"         {IMPORTANCES_PATH}")
    print(f"         {ERRORS_PATH}")
    print(f"  Report: {report}")
    return experiments


def open_dashboard() -> None:
    """E8: start the Streamlit dashboard."""
    application = Path(__file__).parent / "presentation" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(application)],
        check=True,
    )


def run_all() -> None:
    """Run preparation, EDA, and modeling in sequence, reporting elapsed time."""
    steps = [
        ("Prepare (E2/E3)", prepare_data),
        ("EDA (E4)", generate_eda),
        ("Train (E5–E7)", train_models),
    ]
    total_start = time.perf_counter()
    for label, step in steps:
        step_start = time.perf_counter()
        print(f"\n── {label} ──")
        step()
        elapsed = time.perf_counter() - step_start
        print(f"   ✓ {label} done in {elapsed:.1f}s")

    total = time.perf_counter() - total_start
    print(
        f"\nFull pipeline complete in {total:.1f}s. "
        "Start the dashboard with: uv run credit-card-fraud-dashboard"
    )


def _check_prerequisites(command: str) -> None:
    """Validate required files exist before running a pipeline step.

    Raises FileNotFoundError with a clear, actionable message when a required
    file is missing, so users know exactly what to do next.
    """
    needs_raw = {"prepare", "all", "eda", "train", "imbalance", "experiments"}
    needs_processed = {"eda", "train", "imbalance", "experiments"}
    needs_model = {"experiments"}

    if command in needs_raw and not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw file not found: {RAW_PATH}\n"
            "Place 'creditcard.csv' in data/raw/ before running the pipeline."
        )
    if command in needs_processed and not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Prepared base not found: {PROCESSED_PATH}\n"
            "Run the 'prepare' step first: credit-card-fraud-pipeline prepare"
        )
    if command in needs_model and not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}\n"
            "Run the 'train' step first: credit-card-fraud-pipeline train"
        )


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="credit-card-fraud-pipeline",
        description="Credit card fraud detection pipeline (MLG/ULB base).",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser(
        "prepare", help="audit the raw base and generate the prepared base (E2/E3)"
    )
    subcommands.add_parser("eda", help="generate figures and EDA report (E4)")
    subcommands.add_parser(
        "train", help="train, evaluate, and save model and artifacts (E5–E7)"
    )
    subcommands.add_parser("all", help="run prepare, eda, and train")
    subcommands.add_parser(
        "imbalance",
        help="compare imbalance strategies on training",
    )
    subcommands.add_parser(
        "experiments",
        help="robustness, explainability, and sensitivities (offline track)",
    )
    subcommands.add_parser("dashboard", help="open the Streamlit dashboard (E8)")

    args = parser.parse_args(argv)

    commands = {
        "prepare": prepare_data,
        "eda": generate_eda,
        "train": train_models,
        "all": run_all,
        "imbalance": compare_imbalance,
        "experiments": run_experiments,
        "dashboard": open_dashboard,
    }

    try:
        _check_prerequisites(args.command)
        commands[args.command]()
    except (FileNotFoundError, SchemaError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
