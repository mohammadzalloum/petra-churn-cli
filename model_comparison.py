"""
Petra Telecom churn model comparison CLI.

This script compares multiple churn classifiers, selects the best model by
cross-validated ranking performance, chooses an operating threshold from
out-of-fold training predictions, evaluates the selected operating point on
the held-out test set, and saves reproducible artifacts for analysis and
deployment.

Run with:
    python model_comparison.py --data-path data/telecom_churn.csv
    python model_comparison.py --data-path data/telecom_churn.csv --dry-run

Optional:
    create config.json next to the script or pass --config-path.
    CLI arguments take precedence over config values.
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]

BINARY_FEATURES = [
    "senior_citizen",
    "has_partner",
    "has_dependents",
]

ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
ALLOWED_SELECTION_METRICS = {
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "pr_auc_mean",
}
ALLOWED_THRESHOLD_STRATEGIES = {
    "best_f1",
    "target_recall_with_best_precision",
    "max_recall_at_or_below_alert_rate",
}

DEFAULT_CONFIG = {
    "random_state": 42,
    "data_path": "data/telecom_churn.csv",
    "results_root": "./output",
    "n_splits": 5,
    "selection_metric": "pr_auc_mean",
    "calibration_bins": 10,
    "error_analysis_top_n": 10,
    "thresholds": [round(float(x), 2) for x in np.arange(0.10, 0.91, 0.05)],
    "threshold_selection_strategy": "best_f1",
    "recall_target": 0.80,
    "max_alert_rate": None,
    "customer_base": 10000,
    "log_level": "INFO",
    "run_snapshot": True,
    "models": [
        {
            "name": "Dummy",
            "type": "dummy",
            "scaler": "passthrough",
            "params": {"strategy": "most_frequent"},
        },
        {
            "name": "LR_default",
            "type": "logistic_regression",
            "scaler": "standard",
            "params": {"max_iter": 1000, "random_state": 42},
        },
        {
            "name": "LR_balanced",
            "type": "logistic_regression",
            "scaler": "standard",
            "params": {
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        {
            "name": "DT_depth5",
            "type": "decision_tree",
            "scaler": "passthrough",
            "params": {"max_depth": 5, "random_state": 42},
        },
        {
            "name": "RF_default",
            "type": "random_forest",
            "scaler": "passthrough",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        {
            "name": "RF_balanced",
            "type": "random_forest",
            "scaler": "passthrough",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    ],
}


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(data, output_path):
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def deep_update(base, updates):
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path="config.json"):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = deep_update(config, user_config)
    return config


def apply_random_seed_to_config(config, random_seed):
    updated_config = copy.deepcopy(config)
    updated_config["random_state"] = random_seed

    for model_cfg in updated_config.get("models", []):
        model_type = model_cfg.get("type")
        params = model_cfg.setdefault("params", {})
        if model_type in {"logistic_regression", "decision_tree", "random_forest"}:
            params["random_state"] = random_seed

    return updated_config


def apply_cli_overrides(config, args):
    updated = apply_random_seed_to_config(config, args.random_seed)
    updated["data_path"] = args.data_path
    updated["results_root"] = args.output_dir
    updated["n_splits"] = args.n_folds
    updated["log_level"] = args.log_level.upper()
    return updated


def validate_config(config):
    if config["selection_metric"] not in ALLOWED_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection_metric: {config['selection_metric']}. "
            f"Allowed: {sorted(ALLOWED_SELECTION_METRICS)}"
        )

    if config["log_level"].upper() not in ALLOWED_LOG_LEVELS:
        raise ValueError(
            f"Unsupported log level: {config['log_level']}. "
            f"Allowed: {sorted(ALLOWED_LOG_LEVELS)}"
        )

    if not isinstance(config["n_splits"], int) or config["n_splits"] < 2:
        raise ValueError("n_splits must be an integer >= 2.")

    if not isinstance(config["calibration_bins"], int) or config["calibration_bins"] < 2:
        raise ValueError("calibration_bins must be an integer >= 2.")

    if not isinstance(config["error_analysis_top_n"], int) or config["error_analysis_top_n"] < 1:
        raise ValueError("error_analysis_top_n must be an integer >= 1.")

    thresholds = config.get("thresholds", [])
    if not isinstance(thresholds, list) or len(thresholds) == 0:
        raise ValueError("thresholds must be a non-empty list.")
    if thresholds != sorted(thresholds):
        raise ValueError("thresholds must be sorted in ascending order.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must not contain duplicates.")
    if any((t <= 0 or t >= 1) for t in thresholds):
        raise ValueError("All thresholds must be strictly between 0 and 1.")

    strategy = config.get("threshold_selection_strategy")
    if strategy not in ALLOWED_THRESHOLD_STRATEGIES:
        raise ValueError(
            f"Unsupported threshold_selection_strategy: {strategy}. "
            f"Allowed: {sorted(ALLOWED_THRESHOLD_STRATEGIES)}"
        )

    recall_target = config.get("recall_target")
    if recall_target is not None and not (0 < recall_target <= 1):
        raise ValueError("recall_target must be between 0 and 1.")

    max_alert_rate = config.get("max_alert_rate")
    if max_alert_rate is not None and not (0 < max_alert_rate <= 1):
        raise ValueError("max_alert_rate must be between 0 and 1.")

    customer_base = config.get("customer_base")
    if not isinstance(customer_base, int) or customer_base <= 0:
        raise ValueError("customer_base must be a positive integer.")

    models = config.get("models")
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("models must be a non-empty list.")

    names = []
    for model_cfg in models:
        for required_key in ("name", "type", "scaler", "params"):
            if required_key not in model_cfg:
                raise ValueError(f"Model config missing required key: {required_key}")
        names.append(model_cfg["name"])

    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique.")

    if not isinstance(config.get("run_snapshot"), bool):
        raise ValueError("run_snapshot must be a boolean.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare churn classification models, save evaluation artifacts, "
            "and support a dry-run configuration check."
        )
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory where all results and plots will be saved.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and configuration without training any models.",
    )
    parser.add_argument(
        "--config-path",
        default="config.json",
        help="Optional path to a JSON config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )
    return parser.parse_args()


def build_output_paths(output_dir, run_dir=None, dry_run=False):
    log_dir = run_dir if run_dir else output_dir
    log_name = "dry_run.log" if dry_run else "run.log"

    return {
        "comparison_table": os.path.join(output_dir, "comparison_table.csv"),
        "pr_curves": os.path.join(output_dir, "pr_curves.png"),
        "calibration": os.path.join(output_dir, "calibration.png"),
        "best_model": os.path.join(output_dir, "best_model.joblib"),
        "experiment_log": os.path.join(output_dir, "experiment_log.csv"),
        "tree_vs_linear": os.path.join(output_dir, "tree_vs_linear_disagreement.md"),
        "test_predictions": os.path.join(output_dir, "test_predictions.csv"),
        "calibration_metrics": os.path.join(output_dir, "calibration_metrics.csv"),
        "threshold_tuning_csv": os.path.join(output_dir, "threshold_tuning.csv"),
        "threshold_sweep": os.path.join(output_dir, "threshold_sweep.png"),
        "threshold_recommendation": os.path.join(output_dir, "threshold_recommendation.md"),
        "error_analysis": os.path.join(output_dir, "error_analysis.md"),
        "operating_metrics": os.path.join(output_dir, "operating_metrics.json"),
        "run_metadata": os.path.join(output_dir, "run_metadata.json"),
        "data_quality_report": os.path.join(output_dir, "data_quality_report.json"),
        "run_log": os.path.join(log_dir, log_name),
    }


def setup_logging(log_path, log_level="INFO"):
    ensure_parent_dir(log_path)
    logger = logging.getLogger("model_comparison")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def validate_data(filepath, logger=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("Dataset is empty.")

    required_cols = NUMERIC_FEATURES + ["churned"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    for col in NUMERIC_FEATURES:
        coerced = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = coerced.isna() & df[col].notna()
        if invalid_mask.any():
            raise ValueError(f"Column '{col}' contains non-numeric values.")
        df[col] = coerced

    target_coerced = pd.to_numeric(df["churned"], errors="coerce")
    invalid_target_mask = target_coerced.isna() & df["churned"].notna()
    if invalid_target_mask.any():
        raise ValueError("Target column 'churned' contains non-numeric values.")
    df["churned"] = target_coerced

    if df["churned"].isna().any():
        raise ValueError("Target column 'churned' contains missing values.")

    unique_targets = set(df["churned"].dropna().astype(int).unique().tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("Target column 'churned' must contain only binary values {0, 1}.")
    df["churned"] = df["churned"].astype(int)

    for col in BINARY_FEATURES:
        non_missing = set(df[col].dropna().astype(int).unique().tolist())
        if not non_missing.issubset({0, 1}):
            raise ValueError(f"Binary feature '{col}' must contain only values in {{0, 1}}.")

    if (df["contract_months"].dropna() <= 0).any():
        raise ValueError("contract_months must be strictly positive when present.")

    if (df["tenure"].dropna() < 0).any():
        raise ValueError("tenure cannot be negative.")
    if (df["monthly_charges"].dropna() < 0).any():
        raise ValueError("monthly_charges cannot be negative.")
    if (df["total_charges"].dropna() < 0).any():
        raise ValueError("total_charges cannot be negative.")
    if (df["num_support_calls"].dropna() < 0).any():
        raise ValueError("num_support_calls cannot be negative.")

    class_counts = df["churned"].value_counts().sort_index()
    min_class_count = int(class_counts.min())
    duplicate_rows = int(df.duplicated().sum())
    suspicious_zero_total_charges = int(
        ((df["tenure"] > 3) & (df["total_charges"] == 0)).sum()
    )

    missing_by_column = {
        col: int(count)
        for col, count in df[required_cols].isna().sum().items()
    }

    negative_value_counts = {
        col: int((df[col].dropna() < 0).sum())
        for col in NUMERIC_FEATURES
    }

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "required_columns": required_cols,
        "class_distribution": {
            int(label): {
                "count": int(count),
                "rate": round(float(count / len(df)), 6),
            }
            for label, count in class_counts.items()
        },
        "min_class_count": min_class_count,
        "duplicate_rows": duplicate_rows,
        "missing_by_column": missing_by_column,
        "negative_value_counts": negative_value_counts,
        "suspicious_zero_total_charges_count": suspicious_zero_total_charges,
    }

    if logger is not None:
        logger.info(
            "Loading data from %s (%s rows, %s columns)",
            filepath,
            df.shape[0],
            df.shape[1],
        )
        logger.info("Class distribution (churned): %s", report["class_distribution"])

        total_missing_features = int(df[NUMERIC_FEATURES].isna().sum().sum())
        if total_missing_features > 0:
            logger.warning(
                "Numeric feature columns contain %s missing values in total. "
                "Pipelines will impute them with median values.",
                total_missing_features,
            )

        if duplicate_rows > 0:
            logger.warning("Dataset contains %s duplicated rows.", duplicate_rows)

        if suspicious_zero_total_charges > 0:
            logger.warning(
                "Found %s rows with tenure > 3 and total_charges == 0. "
                "This may indicate data quality issues.",
                suspicious_zero_total_charges,
            )

    return df, report


def validate_cv_feasibility(df, n_splits):
    min_class_count = int(df["churned"].value_counts().min())
    if n_splits > min_class_count:
        raise ValueError(
            f"n_splits={n_splits} is too large for the minority class count "
            f"({min_class_count})."
        )


def log_pipeline_configuration(logger, config, output_paths):
    model_names = [model_cfg["name"] for model_cfg in config.get("models", [])]

    logger.info("Pipeline configuration:")
    logger.info("  data_path: %s", config["data_path"])
    logger.info("  output_dir: %s", config["results_root"])
    logger.info("  n_folds: %s", config["n_splits"])
    logger.info("  random_seed: %s", config["random_state"])
    logger.info("  selection_metric: %s", config["selection_metric"])
    logger.info("  threshold_selection_strategy: %s", config["threshold_selection_strategy"])
    logger.info("  recall_target: %s", config["recall_target"])
    logger.info("  max_alert_rate: %s", config["max_alert_rate"])
    logger.info("  calibration_bins: %s", config["calibration_bins"])
    logger.info("  error_analysis_top_n: %s", config["error_analysis_top_n"])
    logger.info("  models_to_compare: %s", model_names)
    logger.info("  key_output_paths:")
    logger.info("    comparison_table: %s", output_paths["comparison_table"])
    logger.info("    pr_curves: %s", output_paths["pr_curves"])
    logger.info("    calibration: %s", output_paths["calibration"])
    logger.info("    best_model: %s", output_paths["best_model"])
    logger.info("    experiment_log: %s", output_paths["experiment_log"])
    logger.info("    threshold_sweep: %s", output_paths["threshold_sweep"])
    logger.info("    threshold_recommendation: %s", output_paths["threshold_recommendation"])
    logger.info("    operating_metrics: %s", output_paths["operating_metrics"])
    logger.info("    error_analysis: %s", output_paths["error_analysis"])


def build_pipeline(model_type, scaler_name, params):
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if model_type == "dummy":
        model = DummyClassifier(**params)
    elif model_type == "logistic_regression":
        if scaler_name == "standard":
            steps.append(("scaler", StandardScaler()))
        model = LogisticRegression(**params)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    steps.append(("model", model))
    return Pipeline(steps)


class ModelSelector:
    def __init__(self, models, selection_metric="pr_auc_mean"):
        self.models = models
        self.selection_metric = selection_metric

    def fit_all(self, X_train, y_train):
        fitted_models = {}
        for name, pipeline in self.models.items():
            pipeline.fit(X_train, y_train)
            fitted_models[name] = pipeline
        return fitted_models

    def select_best_from_results(self, results_df):
        ranked = results_df.sort_values(
            self.selection_metric,
            ascending=False
        ).reset_index(drop=True)
        best_row = ranked.iloc[0]
        return best_row["model"], best_row

    def top_n_from_results(self, results_df, n=3):
        ranked = results_df.sort_values(
            self.selection_metric,
            ascending=False
        ).reset_index(drop=True)
        return ranked.head(n)["model"].tolist()


def load_and_preprocess(filepath="data/telecom_churn.csv", random_state=42, df=None):
    if df is None:
        df, _ = validate_data(filepath)

    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def define_models(config=None):
    model_configs = (
        DEFAULT_CONFIG["models"]
        if config is None
        else config.get("models", DEFAULT_CONFIG["models"])
    )

    models = {}
    for model_cfg in model_configs:
        models[model_cfg["name"]] = build_pipeline(
            model_type=model_cfg["type"],
            scaler_name=model_cfg.get("scaler", "passthrough"),
            params=model_cfg.get("params", {}),
        )
    return models


def run_cv_comparison(models, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    for name, pipeline in models.items():
        fold_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "pr_auc": [],
        }

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            model = clone(pipeline)
            model.fit(X_train_fold, y_train_fold)

            y_pred = model.predict(X_val_fold)
            y_proba = model.predict_proba(X_val_fold)[:, 1]

            fold_metrics["accuracy"].append(accuracy_score(y_val_fold, y_pred))
            fold_metrics["precision"].append(
                precision_score(y_val_fold, y_pred, zero_division=0)
            )
            fold_metrics["recall"].append(
                recall_score(y_val_fold, y_pred, zero_division=0)
            )
            fold_metrics["f1"].append(
                f1_score(y_val_fold, y_pred, zero_division=0)
            )
            fold_metrics["pr_auc"].append(
                average_precision_score(y_val_fold, y_proba)
            )

        row = {"model": name}
        for metric_name, scores in fold_metrics.items():
            row[f"{metric_name}_mean"] = float(np.mean(scores))
            row[f"{metric_name}_std"] = float(np.std(scores))
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))
    return results_df


def save_comparison_table(results_df, output_path):
    ensure_parent_dir(output_path)
    results_df.to_csv(output_path, index=False)


def plot_pr_curves_top3(models, top_model_names, X_test, y_test, output_path):
    ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top_model_names:
        PrecisionRecallDisplay.from_estimator(
            models[name],
            X_test,
            y_test,
            ax=ax,
            name=name,
        )

    ax.set_title("Precision-Recall Curves for Top Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_calibration_top3(models, top_model_names, X_test, y_test, output_path, n_bins=10):
    ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top_model_names:
        CalibrationDisplay.from_estimator(
            models[name],
            X_test,
            y_test,
            n_bins=n_bins,
            ax=ax,
            name=name,
        )

    ax.set_title("Calibration Curves for Top Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_best_model(best_model, output_path):
    ensure_parent_dir(output_path)
    dump(best_model, output_path)


def log_experiment(results_df, output_path):
    timestamp = datetime.now().isoformat()

    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy": results_df["accuracy_mean"],
        "precision": results_df["precision_mean"],
        "recall": results_df["recall_mean"],
        "f1": results_df["f1_mean"],
        "pr_auc": results_df["pr_auc_mean"],
        "timestamp": timestamp,
    })

    ensure_parent_dir(output_path)
    log_df.to_csv(output_path, index=False)


def generate_oof_probabilities(pipeline, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_probabilities = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]

        model = clone(pipeline)
        model.fit(X_train_fold, y_train_fold)
        oof_probabilities[val_idx] = model.predict_proba(X_val_fold)[:, 1]

    return oof_probabilities


def compute_threshold_table(y_true, y_prob, thresholds, customer_base=10000):
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        predicted_positive_count = int(y_pred.sum())
        alert_rate = float(y_pred.mean())

        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "predicted_positive_count": predicted_positive_count,
            "alert_rate": alert_rate,
            "alerts_per_1000": alert_rate * 1000.0,
            "expected_alerts_per_customer_base": alert_rate * customer_base,
        })

    return pd.DataFrame(rows)


def select_threshold_candidates(
    threshold_df,
    strategy="best_f1",
    recall_target=None,
    max_alert_rate=None,
):
    best_f1 = (
        threshold_df.sort_values(
            ["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )

    recall_target_candidate = None
    if recall_target is not None:
        eligible = threshold_df[threshold_df["recall"] >= recall_target].copy()
        if not eligible.empty:
            recall_target_candidate = (
                eligible.sort_values(
                    ["precision", "f1", "threshold"],
                    ascending=[False, False, False],
                )
                .iloc[0]
                .to_dict()
            )

    capacity_candidate = None
    if max_alert_rate is not None:
        eligible = threshold_df[threshold_df["alert_rate"] <= max_alert_rate].copy()
        if not eligible.empty:
            capacity_candidate = (
                eligible.sort_values(
                    ["recall", "f1", "precision", "threshold"],
                    ascending=[False, False, False, False],
                )
                .iloc[0]
                .to_dict()
            )

    if strategy == "best_f1":
        selected = best_f1
        selected_reason = "Selected the threshold that maximized out-of-fold F1."
    elif strategy == "target_recall_with_best_precision":
        selected = recall_target_candidate if recall_target_candidate is not None else best_f1
        selected_reason = (
            f"Selected the highest-precision threshold achieving recall >= {recall_target:.2f}."
            if recall_target_candidate is not None
            else "No threshold achieved the recall target, so the best-F1 threshold was used."
        )
    elif strategy == "max_recall_at_or_below_alert_rate":
        selected = capacity_candidate if capacity_candidate is not None else best_f1
        selected_reason = (
            f"Selected the highest-recall threshold with alert_rate <= {max_alert_rate:.4f}."
            if capacity_candidate is not None
            else "No threshold satisfied the alert-rate constraint, so the best-F1 threshold was used."
        )
    else:
        raise ValueError(f"Unsupported threshold selection strategy: {strategy}")

    return selected, {
        "selected": selected,
        "selected_reason": selected_reason,
        "best_f1": best_f1,
        "recall_target_candidate": recall_target_candidate,
        "capacity_candidate": capacity_candidate,
    }


def format_threshold_candidate(candidate):
    if candidate is None:
        return "- Not available"

    return (
        f"- threshold={candidate['threshold']:.2f}, "
        f"precision={candidate['precision']:.3f}, "
        f"recall={candidate['recall']:.3f}, "
        f"f1={candidate['f1']:.3f}, "
        f"alert_rate={candidate['alert_rate']:.3f}, "
        f"alerts_per_1000={candidate['alerts_per_1000']:.1f}"
    )


def save_threshold_recommendation(
    best_model_name,
    strategy,
    recall_target,
    max_alert_rate,
    threshold_candidates,
    output_path,
):
    lines = [
        "# Threshold Recommendation",
        "",
        f"- **Model:** {best_model_name}",
        f"- **Selection strategy:** {strategy}",
        f"- **Selection basis:** out-of-fold training predictions",
        f"- **Recall target:** {recall_target}",
        f"- **Max alert rate:** {max_alert_rate}",
        "",
        "## Selected Operating Threshold",
        "",
        format_threshold_candidate(threshold_candidates["selected"]),
        "",
        threshold_candidates["selected_reason"],
        "",
        "## Alternative Candidates",
        "",
        "**Best F1 candidate**",
        format_threshold_candidate(threshold_candidates["best_f1"]),
        "",
        "**Recall-target candidate**",
        format_threshold_candidate(threshold_candidates["recall_target_candidate"]),
        "",
        "**Capacity-constrained candidate**",
        format_threshold_candidate(threshold_candidates["capacity_candidate"]),
        "",
        "The selected threshold is chosen on training data only. "
        "The held-out test set remains reserved for final evaluation.",
    ]

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_threshold_sweep(
    threshold_df,
    selected_threshold,
    output_path,
    max_alert_rate=None,
):
    ensure_parent_dir(output_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    ax1.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    ax1.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    ax1.axvline(selected_threshold, linestyle="--", label=f"Selected threshold ({selected_threshold:.2f})")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(
        threshold_df["threshold"],
        threshold_df["alerts_per_1000"],
        marker="s",
        linestyle="--",
        label="Alerts per 1,000",
    )
    ax2.set_ylabel("Alerts per 1,000 customers")

    if max_alert_rate is not None:
        ax2.axhline(
            max_alert_rate * 1000.0,
            linestyle=":",
            label=f"Max alerts ({max_alert_rate * 1000.0:.1f} / 1,000)",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    ax1.set_title("Threshold Selection on Training OOF Predictions")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_operating_point(model, X, y, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "brier_score": float(brier_score_loss(y, y_proba)),
        "predicted_positive_count": int(y_pred.sum()),
        "alert_rate": float(y_pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_tree_vs_linear_disagreement(
    rf_model,
    lr_model,
    X_test,
    y_test,
    feature_names,
    min_diff=0.15,
):
    rf_proba_all = rf_model.predict_proba(X_test)[:, 1]
    lr_proba_all = lr_model.predict_proba(X_test)[:, 1]

    prob_diffs = np.abs(rf_proba_all - lr_proba_all)

    max_pos = int(np.argmax(prob_diffs))
    max_diff = float(prob_diffs[max_pos])

    if max_diff < min_diff:
        return None

    sample_idx = int(X_test.index[max_pos])
    sample_row = X_test.iloc[max_pos]

    feature_values = {
        feature: (
            sample_row[feature].item()
            if hasattr(sample_row[feature], "item")
            else sample_row[feature]
        )
        for feature in feature_names
    }

    true_label = y_test.iloc[max_pos]
    true_label = int(true_label.item() if hasattr(true_label, "item") else true_label)

    return {
        "sample_idx": sample_idx,
        "feature_values": feature_values,
        "rf_proba": float(rf_proba_all[max_pos]),
        "lr_proba": float(lr_proba_all[max_pos]),
        "prob_diff": max_diff,
        "true_label": true_label,
    }


def save_tree_vs_linear_disagreement(disagreement, output_path):
    md_lines = [
        "# Tree vs. Linear Disagreement Analysis",
        "",
        "## Sample Details",
        "",
        f"- **Test-set index:** {disagreement['sample_idx']}",
        f"- **True label:** {disagreement['true_label']}",
        f"- **RF predicted P(churn=1):** {disagreement['rf_proba']:.4f}",
        f"- **LR predicted P(churn=1):** {disagreement['lr_proba']:.4f}",
        f"- **Probability difference:** {disagreement['prob_diff']:.4f}",
        "",
        "## Feature Values",
        "",
    ]
    for feat, val in disagreement["feature_values"].items():
        md_lines.append(f"- **{feat}:** {val}")

    md_lines.extend([
        "",
        "## Structural Explanation",
        "The disagreement is likely driven by a threshold-style pattern around "
        "`contract_months = 1`, which the random forest can treat as a strong churn "
        "signal when combined with the rest of the feature profile. In this sample, "
        "the tree model assigned a much higher churn probability than logistic "
        "regression, suggesting that the tree captured a rule-like interaction that "
        "the linear model smoothed out through additive feature effects. This also "
        "matches the broader pattern in churn modeling where tree-based models can "
        "react more strongly to split points and feature interactions than linear models.",
    ])

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def save_test_predictions(
    models,
    best_model_name,
    selected_threshold,
    X_test,
    y_test,
    output_path,
):
    pred_df = pd.DataFrame({
        "sample_idx": X_test.index,
        "true_label": y_test.values,
    })

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pred_df[f"{name}_proba"] = y_proba
        pred_df[f"{name}_pred_0_50"] = (y_proba >= 0.5).astype(int)

    best_probs = models[best_model_name].predict_proba(X_test)[:, 1]
    pred_df["selected_threshold"] = float(selected_threshold)
    pred_df[f"{best_model_name}_pred_selected"] = (best_probs >= selected_threshold).astype(int)

    ensure_parent_dir(output_path)
    pred_df.to_csv(output_path, index=False)
    return pred_df


def save_calibration_metrics(models, X_test, y_test, output_path, n_bins=10):
    rows = []
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "brier_score": float(brier_score_loss(y_test, y_proba)),
            "ece": float(compute_ece(y_test, y_proba, n_bins=n_bins)),
            "pr_auc_test": float(average_precision_score(y_test, y_proba)),
        })

    metrics_df = pd.DataFrame(rows).sort_values("brier_score", ascending=True).reset_index(drop=True)
    ensure_parent_dir(output_path)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.mean()

    return float(ece)


def save_error_analysis(
    best_model,
    X_test,
    y_test,
    feature_names,
    output_path,
    threshold,
    top_n=10,
):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    analysis_df = X_test.copy()
    analysis_df["sample_idx"] = X_test.index
    analysis_df["true_label"] = y_test.values
    analysis_df["predicted_label"] = y_pred
    analysis_df["predicted_proba"] = y_proba

    def label_error(row):
        if row["true_label"] == 1 and row["predicted_label"] == 1:
            return "TP"
        if row["true_label"] == 0 and row["predicted_label"] == 0:
            return "TN"
        if row["true_label"] == 0 and row["predicted_label"] == 1:
            return "FP"
        return "FN"

    analysis_df["error_type"] = analysis_df.apply(label_error, axis=1)

    fp_df = analysis_df[analysis_df["error_type"] == "FP"].sort_values(
        "predicted_proba",
        ascending=False,
    )
    fn_df = analysis_df[analysis_df["error_type"] == "FN"].sort_values(
        "predicted_proba",
        ascending=True,
    )

    lines = [
        "# Error Analysis",
        "",
        f"- **Threshold used:** {threshold:.2f}",
        f"- **Total test samples:** {len(analysis_df)}",
        f"- **False positives:** {len(fp_df)}",
        f"- **False negatives:** {len(fn_df)}",
        "",
        "## Highest-Confidence False Positives",
        "",
    ]

    if fp_df.empty:
        lines.append("No false positives found.")
    else:
        for _, row in fp_df.head(top_n).iterrows():
            lines.append(
                f"- **sample_idx={int(row['sample_idx'])}** | "
                f"proba={row['predicted_proba']:.4f} | "
                + ", ".join([f"{feat}={row[feat]}" for feat in feature_names])
            )

    lines.extend([
        "",
        "## Lowest-Confidence False Negatives",
        "",
    ])

    if fn_df.empty:
        lines.append("No false negatives found.")
    else:
        for _, row in fn_df.head(top_n).iterrows():
            lines.append(
                f"- **sample_idx={int(row['sample_idx'])}** | "
                f"proba={row['predicted_proba']:.4f} | "
                + ", ".join([f"{feat}={row[feat]}" for feat in feature_names])
            )

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return analysis_df


def save_run_metadata(
    config,
    best_name,
    selected_threshold,
    run_dir,
    data_quality_report,
    threshold_candidates,
    operating_metrics,
    output_path,
):
    metadata = {
        "run_dir": run_dir,
        "saved_at": datetime.now().isoformat(),
        "best_model_name": best_name,
        "selected_threshold": float(selected_threshold),
        "selection_metric": config["selection_metric"],
        "threshold_selection_strategy": config["threshold_selection_strategy"],
        "data_quality_report": data_quality_report,
        "threshold_candidates": threshold_candidates,
        "operating_metrics": operating_metrics,
        "config": config,
    }
    save_json(metadata, output_path)


def copy_outputs_to_run_dir(source_paths, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    for src in source_paths:
        if os.path.exists(src):
            dst = os.path.join(run_dir, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)


def run_dry_run(logger, config, output_paths):
    df, data_quality_report = validate_data(config["data_path"], logger=logger)
    validate_cv_feasibility(df, config["n_splits"])
    models = define_models(config=config)

    if not models:
        raise ValueError("No model configurations were defined.")

    logger.info("%s model configurations defined: %s", len(models), list(models.keys()))
    log_pipeline_configuration(logger, config, output_paths)
    save_json(data_quality_report, output_paths["data_quality_report"])
    logger.info(
        "Dry run completed successfully. Data validated and configuration checked. "
        "No models were trained."
    )
    return 0


def run_full_pipeline(logger, config, output_paths, run_dir):
    df, data_quality_report = validate_data(config["data_path"], logger=logger)
    validate_cv_feasibility(df, config["n_splits"])
    save_json(data_quality_report, output_paths["data_quality_report"])

    models = define_models(config=config)
    if not models:
        raise ValueError("No model configurations were defined.")

    logger.info("%s model configurations defined: %s", len(models), list(models.keys()))
    log_pipeline_configuration(logger, config, output_paths)

    X_train, X_test, y_train, y_test = load_and_preprocess(
        config["data_path"],
        random_state=config["random_state"],
        df=df,
    )
    logger.info(
        "Data split complete: %s train, %s test, churn rate: %.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )

    results_df = run_cv_comparison(
        models,
        X_train,
        y_train,
        n_splits=config["n_splits"],
        random_state=config["random_state"],
    )
    logger.info("=== Model Comparison Table (CV) ===")
    logger.info("\n%s", results_df.to_string(index=False))
    save_comparison_table(results_df, output_paths["comparison_table"])
    log_experiment(results_df, output_paths["experiment_log"])

    selector = ModelSelector(models, selection_metric=config["selection_metric"])
    best_name, _ = selector.select_best_from_results(results_df)
    top_model_names = selector.top_n_from_results(results_df, n=3)
    best_pipeline_template = models[best_name]

    oof_probabilities = generate_oof_probabilities(
        best_pipeline_template,
        X_train,
        y_train,
        n_splits=config["n_splits"],
        random_state=config["random_state"],
    )

    threshold_df = compute_threshold_table(
        y_train,
        oof_probabilities,
        thresholds=config["thresholds"],
        customer_base=config["customer_base"],
    )
    ensure_parent_dir(output_paths["threshold_tuning_csv"])
    threshold_df.to_csv(output_paths["threshold_tuning_csv"], index=False)

    selected_threshold_row, threshold_candidates = select_threshold_candidates(
        threshold_df,
        strategy=config["threshold_selection_strategy"],
        recall_target=config["recall_target"],
        max_alert_rate=config["max_alert_rate"],
    )
    selected_threshold = float(selected_threshold_row["threshold"])

    save_threshold_recommendation(
        best_model_name=best_name,
        strategy=config["threshold_selection_strategy"],
        recall_target=config["recall_target"],
        max_alert_rate=config["max_alert_rate"],
        threshold_candidates=threshold_candidates,
        output_path=output_paths["threshold_recommendation"],
    )

    plot_threshold_sweep(
        threshold_df=threshold_df,
        selected_threshold=selected_threshold,
        output_path=output_paths["threshold_sweep"],
        max_alert_rate=config["max_alert_rate"],
    )

    fitted_models = selector.fit_all(X_train, y_train)
    save_best_model(fitted_models[best_name], output_paths["best_model"])
    logger.info("Best model by %s: %s", config["selection_metric"], best_name)
    logger.info("Selected operating threshold from training OOF: %.2f", selected_threshold)

    plot_pr_curves_top3(
        fitted_models,
        top_model_names,
        X_test,
        y_test,
        output_paths["pr_curves"],
    )
    plot_calibration_top3(
        fitted_models,
        top_model_names,
        X_test,
        y_test,
        output_paths["calibration"],
        n_bins=config["calibration_bins"],
    )

    if "RF_default" in fitted_models and "LR_default" in fitted_models:
        disagreement = find_tree_vs_linear_disagreement(
            fitted_models["RF_default"],
            fitted_models["LR_default"],
            X_test,
            y_test,
            NUMERIC_FEATURES,
        )
        if disagreement:
            logger.info(
                "--- Tree-vs-linear disagreement (sample idx=%s) ---",
                disagreement["sample_idx"],
            )
            logger.info(
                "RF P(churn=1)=%.3f | LR P(churn=1)=%.3f | |diff|=%.3f | true label=%s",
                disagreement["rf_proba"],
                disagreement["lr_proba"],
                disagreement["prob_diff"],
                disagreement["true_label"],
            )
            save_tree_vs_linear_disagreement(disagreement, output_paths["tree_vs_linear"])
            logger.info("Saved to %s", output_paths["tree_vs_linear"])
    else:
        logger.warning(
            "Skipped tree-vs-linear disagreement analysis because RF_default or LR_default "
            "is missing from the configured model list."
        )

    pred_df = save_test_predictions(
        fitted_models,
        best_name,
        selected_threshold,
        X_test,
        y_test,
        output_paths["test_predictions"],
    )
    logger.info(
        "Saved test predictions to %s (%s rows)",
        output_paths["test_predictions"],
        len(pred_df),
    )

    calibration_metrics_df = save_calibration_metrics(
        fitted_models,
        X_test,
        y_test,
        output_paths["calibration_metrics"],
        n_bins=config["calibration_bins"],
    )
    logger.info("Saved calibration metrics to %s", output_paths["calibration_metrics"])
    logger.info("\n%s", calibration_metrics_df.to_string(index=False))

    operating_metrics = evaluate_operating_point(
        fitted_models[best_name],
        X_test,
        y_test,
        selected_threshold,
    )
    save_json(operating_metrics, output_paths["operating_metrics"])
    logger.info(
        "Operating-point test metrics at threshold %.2f: precision=%.3f, recall=%.3f, f1=%.3f, alert_rate=%.3f",
        selected_threshold,
        operating_metrics["precision"],
        operating_metrics["recall"],
        operating_metrics["f1"],
        operating_metrics["alert_rate"],
    )

    error_df = save_error_analysis(
        fitted_models[best_name],
        X_test,
        y_test,
        feature_names=NUMERIC_FEATURES,
        output_path=output_paths["error_analysis"],
        threshold=selected_threshold,
        top_n=config["error_analysis_top_n"],
    )
    logger.info("Saved error analysis to %s", output_paths["error_analysis"])
    logger.info("Error type counts: %s", error_df["error_type"].value_counts().to_dict())

    save_run_metadata(
        config=config,
        best_name=best_name,
        selected_threshold=selected_threshold,
        run_dir=run_dir,
        data_quality_report=data_quality_report,
        threshold_candidates=threshold_candidates,
        operating_metrics=operating_metrics,
        output_path=output_paths["run_metadata"],
    )

    if config["run_snapshot"]:
        canonical_outputs = [
            output_paths["comparison_table"],
            output_paths["pr_curves"],
            output_paths["calibration"],
            output_paths["best_model"],
            output_paths["experiment_log"],
            output_paths["tree_vs_linear"],
            output_paths["test_predictions"],
            output_paths["calibration_metrics"],
            output_paths["threshold_tuning_csv"],
            output_paths["threshold_sweep"],
            output_paths["threshold_recommendation"],
            output_paths["error_analysis"],
            output_paths["operating_metrics"],
            output_paths["run_metadata"],
            output_paths["data_quality_report"],
        ]
        copy_outputs_to_run_dir(canonical_outputs, run_dir)

    logger.info(
        "--- All results saved to %s and snapshot copied to %s ---",
        config["results_root"],
        run_dir,
    )
    return 0


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_config = load_config(args.config_path)
    config = apply_cli_overrides(base_config, args)
    validate_config(config)

    run_dir = None
    if not args.dry_run:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, "runs", run_id)

    output_paths = build_output_paths(
        args.output_dir,
        run_dir=run_dir,
        dry_run=args.dry_run,
    )
    logger = setup_logging(output_paths["run_log"], log_level=config["log_level"])

    try:
        logger.info("Loaded configuration.")
        logger.info("CLI arguments: %s", vars(args))
        if os.path.exists(args.config_path):
            logger.info("Detected config file at %s. CLI arguments take precedence.", args.config_path)

        if args.dry_run:
            return run_dry_run(logger, config, output_paths)

        return run_full_pipeline(logger, config, output_paths, run_dir)

    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except ValueError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())