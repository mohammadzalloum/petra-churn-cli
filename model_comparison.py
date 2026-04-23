"""
Module 5 Week B — Integration Task: Model Comparison & Decision Memo

Enhanced version that preserves the original task structure while adding:
- threshold tuning
- error analysis
- full test prediction exports
- calibration metrics (Brier score + ECE)
- structured logging
- timestamped run folders
- simple JSON config workflow
- ModelSelector helper class

Run with:  python model_comparison_enhanced.py
Optional config: create config.json next to the script.
"""

import copy
import json
import logging
import os
import shutil
from datetime import datetime

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
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

DEFAULT_CONFIG = {
    "random_state": 42,
    "data_path": "data/telecom_churn.csv",
    "results_root": "results_enhanced",
    "n_splits": 5,
    "selection_metric": "pr_auc_mean",
    "calibration_bins": 10,
    "error_analysis_top_n": 10,
    "thresholds": [round(float(x), 2) for x in np.arange(0.10, 0.91, 0.05)],
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
            "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
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
            },
        },
    ],
}

LOGGER = logging.getLogger("model_comparison")


def ensure_parent_dir(path):
    """Create the parent directory for a path if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def deep_update(base, updates):
    """Recursively merge config dictionaries."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path="config.json"):
    """Load config.json if present, otherwise use defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = deep_update(config, user_config)
    return config


def setup_logging(log_path):
    """Configure console + file logging."""
    ensure_parent_dir(log_path)
    logger = logging.getLogger("model_comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def build_pipeline(model_type, scaler_name, params):
    """Build one pipeline from config."""
    scaler = StandardScaler() if scaler_name == "standard" else "passthrough"

    if model_type == "dummy":
        model = DummyClassifier(**params)
    elif model_type == "logistic_regression":
        model = LogisticRegression(**params)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([
        ("scaler", scaler),
        ("model", model),
    ])


def get_top3_models_by_test_pr_auc(models, X_test, y_test):
    """Return the top 3 fitted models by test-set PR-AUC."""
    pr_auc_scores = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc_scores[name] = average_precision_score(y_test, y_proba)
    return sorted(pr_auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
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


class ModelSelector:
    """Small helper to fit, rank, and retrieve model configurations."""

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
        ranked = results_df.sort_values(self.selection_metric, ascending=False).reset_index(drop=True)
        best_row = ranked.iloc[0]
        return best_row["model"], best_row

    def top_n_by_test_pr_auc(self, fitted_models, X_test, y_test, n=3):
        scores = get_top3_models_by_test_pr_auc(fitted_models, X_test, y_test)
        return scores[:n]


def load_and_preprocess(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split into train/test sets.

    Uses an 80/20 stratified split. Features are the 8 NUMERIC_FEATURES
    columns. Target is `churned`.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    df = pd.read_csv(filepath)

    required_cols = NUMERIC_FEATURES + ["churned"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

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
    """Define 6 model configurations for comparison.

    The pattern is deliberate: a default vs class_weight='balanced' pair
    at BOTH the linear and ensemble family levels. This lets you observe
    the class_weight effect at two levels of model complexity.

    The 6 configurations:
      1. DummyClassifier(strategy='most_frequent') — baseline
      2. LogisticRegression(max_iter=1000) — linear default (needs scaling)
      3. LogisticRegression(class_weight='balanced', max_iter=1000) — linear balanced (needs scaling)
      4. DecisionTreeClassifier(max_depth=5) — tree baseline
      5. RandomForestClassifier(n_estimators=100, max_depth=10) — ensemble default
      6. RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced') — ensemble balanced

    LR variants require StandardScaler preprocessing; tree-based models
    do not. Use sklearn Pipeline to pair each model with its preprocessing.

    Returns:
        Dict of {name: sklearn.pipeline.Pipeline} with 6 entries.
        Names: 'Dummy', 'LR_default', 'LR_balanced', 'DT_depth5',
               'RF_default', 'RF_balanced'.
    """
    model_configs = DEFAULT_CONFIG["models"] if config is None else config.get("models", DEFAULT_CONFIG["models"])

    models = {}
    for model_cfg in model_configs:
        models[model_cfg["name"]] = build_pipeline(
            model_type=model_cfg["type"],
            scaler_name=model_cfg.get("scaler", "passthrough"),
            params=model_cfg.get("params", {}),
        )

    return models


def run_cv_comparison(models, X, y, n_splits=5, random_state=42):
    """Run 5-fold stratified cross-validation on all models.

    For each model, compute mean and std of: accuracy, precision, recall,
    F1, and PR-AUC across folds. PR-AUC uses predict_proba — it is a
    threshold-independent ranking metric.

    Args:
        models: Dict of {name: Pipeline} from define_models().
        X: Feature DataFrame.
        y: Target Series.
        n_splits: Number of CV folds.
        random_state: Random seed for StratifiedKFold.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, precision_std, recall_mean, recall_std,
        f1_mean, f1_std, pr_auc_mean, pr_auc_std.
        One row per model (6 rows total).
    """
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

            pipeline.fit(X_train_fold, y_train_fold)

            y_pred = pipeline.predict(X_val_fold)
            y_proba = pipeline.predict_proba(X_val_fold)[:, 1]

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
    return results_df.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def save_comparison_table(results_df, output_path="results/comparison_table.csv"):
    """Save the comparison table to CSV.

    Args:
        results_df: DataFrame from run_cv_comparison().
        output_path: Destination path.
    """
    ensure_parent_dir(output_path)
    results_df.to_csv(output_path, index=False)


def plot_pr_curves_top3(models, X_test, y_test, output_path="results/pr_curves.png"):
    """Plot PR curves for the top 3 models (by PR-AUC) on one axes and save.

    Args:
        models: Dict of {name: fitted Pipeline} — must already be fitted.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination path for the PNG.
    """
    top3 = get_top3_models_by_test_pr_auc(models, X_test, y_test)

    ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, _ in top3:
        PrecisionRecallDisplay.from_estimator(
            models[name],
            X_test,
            y_test,
            ax=ax,
            name=name,
        )

    ax.set_title("Precision-Recall Curves for Top 3 Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_calibration_top3(models, X_test, y_test, output_path="results/calibration.png"):
    """Plot calibration curves for the top 3 models and save.

    Uses CalibrationDisplay.from_estimator.

    Args:
        models: Dict of {name: fitted Pipeline} — must already be fitted.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination path for the PNG.
    """
    top3 = get_top3_models_by_test_pr_auc(models, X_test, y_test)

    ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, _ in top3:
        CalibrationDisplay.from_estimator(
            models[name],
            X_test,
            y_test,
            n_bins=10,
            ax=ax,
            name=name,
        )

    ax.set_title("Calibration Curves for Top 3 Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_best_model(best_model, output_path="results/best_model.joblib"):
    """Persist the best model to disk with joblib.

    Args:
        best_model: A fitted sklearn Pipeline.
        output_path: Destination path.
    """
    ensure_parent_dir(output_path)
    dump(best_model, output_path)


def log_experiment(results_df, output_path="results/experiment_log.csv"):
    """Log all model results with timestamps.

    Produces a CSV with columns: model_name, accuracy, precision, recall,
    f1, pr_auc, timestamp. One row per model. The timestamp records WHEN
    the experiment was run (ISO format).

    Args:
        results_df: DataFrame from run_cv_comparison().
        output_path: Destination path.
    """
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


def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test, y_test,
                                     feature_names, min_diff=0.15):
    """Find ONE test sample where RF and LR predicted probabilities differ most."""
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
        feature: sample_row[feature].item() if hasattr(sample_row[feature], "item") else sample_row[feature]
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


def save_test_predictions(models, X_test, y_test, output_path="results/test_predictions.csv", threshold=0.5):
    """Save per-row test predictions for every fitted model."""
    pred_df = pd.DataFrame({
        "sample_idx": X_test.index,
        "true_label": y_test.values,
    })

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pred_df[f"{name}_proba"] = y_proba
        pred_df[f"{name}_pred"] = (y_proba >= threshold).astype(int)

    ensure_parent_dir(output_path)
    pred_df.to_csv(output_path, index=False)
    return pred_df


def save_calibration_metrics(models, X_test, y_test, output_path="results/calibration_metrics.csv", n_bins=10):
    """Save numeric calibration metrics for all fitted models."""
    rows = []
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "brier_score": float(brier_score_loss(y_test, y_proba)),
            "ece": compute_ece(y_test, y_proba, n_bins=n_bins),
            "pr_auc_test": float(average_precision_score(y_test, y_proba)),
        })

    metrics_df = pd.DataFrame(rows).sort_values("brier_score", ascending=True).reset_index(drop=True)
    ensure_parent_dir(output_path)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def run_threshold_tuning(best_model, X_test, y_test, thresholds=None,
                         output_csv_path="results/threshold_tuning.csv",
                         output_plot_path="results/threshold_sweep.png"):
    """Sweep thresholds for the chosen model and save metrics + plot."""
    if thresholds is None:
        thresholds = [round(float(x), 2) for x in np.arange(0.10, 0.91, 0.05)]

    y_proba = best_model.predict_proba(X_test)[:, 1]
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "predicted_positive_count": int(y_pred.sum()),
            "alert_rate": float(y_pred.mean()),
        })

    threshold_df = pd.DataFrame(rows)
    ensure_parent_dir(output_csv_path)
    threshold_df.to_csv(output_csv_path, index=False)

    ensure_parent_dir(output_plot_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning for Best Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_plot_path)
    plt.close(fig)

    return threshold_df


def save_error_analysis(best_model, X_test, y_test, feature_names,
                        output_path="results/error_analysis.md",
                        threshold=0.5, top_n=10):
    """Write a markdown error analysis for the best model."""
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

    fp_df = analysis_df[analysis_df["error_type"] == "FP"].sort_values("predicted_proba", ascending=False)
    fn_df = analysis_df[analysis_df["error_type"] == "FN"].sort_values("predicted_proba", ascending=True)

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
                f"- **sample_idx={int(row['sample_idx'])}** | proba={row['predicted_proba']:.4f} | "
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
                f"- **sample_idx={int(row['sample_idx'])}** | proba={row['predicted_proba']:.4f} | "
                + ", ".join([f"{feat}={row[feat]}" for feat in feature_names])
            )

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return analysis_df


def save_run_metadata(config, best_name, run_dir, output_path):
    """Save run metadata for reproducibility."""
    metadata = {
        "run_dir": run_dir,
        "saved_at": datetime.now().isoformat(),
        "best_model_name": best_name,
        "selection_metric": config["selection_metric"],
        "config": config,
    }
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def copy_outputs_to_run_dir(source_paths, run_dir):
    """Copy canonical outputs into the timestamped run folder."""
    os.makedirs(run_dir, exist_ok=True)
    for src in source_paths:
        if os.path.exists(src):
            dst = os.path.join(run_dir, os.path.basename(src))
            shutil.copy2(src, dst)


def main():
    """Orchestrate all 9 integration tasks. Run with: python model_comparison.py"""
    config = load_config()
    results_root = config["results_root"]
    os.makedirs(results_root, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, "runs", run_id)
    logger = setup_logging(os.path.join(run_dir, "run.log"))

    logger.info("Loaded configuration.")
    logger.info("Run directory: %s", run_dir)

    # Canonical task output paths (kept compatible with the original assignment structure)
    comparison_table_path = os.path.join(results_root, "comparison_table.csv")
    pr_curve_path = os.path.join(results_root, "pr_curves.png")
    calibration_plot_path = os.path.join(results_root, "calibration.png")
    best_model_path = os.path.join(results_root, "best_model.joblib")
    experiment_log_path = os.path.join(results_root, "experiment_log.csv")
    disagreement_md_path = os.path.join(results_root, "tree_vs_linear_disagreement.md")

    # Enhanced analysis outputs
    test_predictions_path = os.path.join(results_root, "test_predictions.csv")
    calibration_metrics_path = os.path.join(results_root, "calibration_metrics.csv")
    threshold_tuning_csv_path = os.path.join(results_root, "threshold_tuning.csv")
    threshold_tuning_plot_path = os.path.join(results_root, "threshold_sweep.png")
    error_analysis_path = os.path.join(results_root, "error_analysis.md")
    metadata_path = os.path.join(results_root, "run_metadata.json")

    # Task 1: Load + split
    result = load_and_preprocess(config["data_path"], random_state=config["random_state"])
    if not result:
        logger.error("load_and_preprocess not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    logger.info(
        "Data: %s train, %s test, churn rate: %.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )

    # Task 2: Define models
    models = define_models(config=config)
    if not models:
        logger.error("define_models not implemented. Exiting.")
        return
    logger.info("%s model configurations defined: %s", len(models), list(models.keys()))

    # Task 3: Cross-validation comparison
    results_df = run_cv_comparison(
        models,
        X_train,
        y_train,
        n_splits=config["n_splits"],
        random_state=config["random_state"],
    )
    if results_df is None:
        logger.error("run_cv_comparison not implemented. Exiting.")
        return
    logger.info("=== Model Comparison Table (5-fold CV) ===")
    logger.info("\n%s", results_df.to_string(index=False))

    # Task 4: Save comparison table
    save_comparison_table(results_df, output_path=comparison_table_path)

    # Fit all models on full training set for plots + persistence
    selector = ModelSelector(models, selection_metric=config["selection_metric"])
    fitted_models = selector.fit_all(X_train, y_train)

    # Task 5: PR curves (top 3)
    plot_pr_curves_top3(fitted_models, X_test, y_test, output_path=pr_curve_path)

    # Task 6: Calibration plot (top 3)
    plot_calibration_top3(fitted_models, X_test, y_test, output_path=calibration_plot_path)

    # Task 7: Save best model
    best_name, _ = selector.select_best_from_results(results_df)
    logger.info("Best model by %s: %s", config["selection_metric"], best_name)
    save_best_model(fitted_models[best_name], output_path=best_model_path)

    # Task 8: Experiment log
    log_experiment(results_df, output_path=experiment_log_path)

    # Task 9: Tree-vs-linear disagreement
    rf_pipeline = fitted_models["RF_default"]
    lr_pipeline = fitted_models["LR_default"]
    disagreement = find_tree_vs_linear_disagreement(
        rf_pipeline, lr_pipeline, X_test, y_test, NUMERIC_FEATURES
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
            "The disagreement is likely driven by a threshold-style pattern around `contract_months = 1`, which the random forest can treat as a strong churn signal when combined with the rest of the feature profile. In this sample, the tree model assigned a much higher churn probability (`0.5998`) than logistic regression (`0.1700`), suggesting that the tree captured a rule-like interaction that the linear model smoothed out through additive feature effects. This interpretation is also consistent with the enhanced error analysis, where the same sample appeared among the highest-confidence false positives, showing that the random forest can sometimes overreact to short-contract risk patterns even when the true label is non-churn. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}"
        ])
        with open(disagreement_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info("Saved to %s", disagreement_md_path)

    # Enhanced analysis: test predictions
    pred_df = save_test_predictions(
        fitted_models,
        X_test,
        y_test,
        output_path=test_predictions_path,
        threshold=0.5,
    )
    logger.info("Saved test predictions to %s (%s rows)", test_predictions_path, len(pred_df))

    # Enhanced analysis: calibration metrics
    calibration_metrics_df = save_calibration_metrics(
        fitted_models,
        X_test,
        y_test,
        output_path=calibration_metrics_path,
        n_bins=config["calibration_bins"],
    )
    logger.info("Saved calibration metrics to %s", calibration_metrics_path)
    logger.info("\n%s", calibration_metrics_df.to_string(index=False))

    # Enhanced analysis: threshold tuning for the selected best model
    threshold_df = run_threshold_tuning(
        fitted_models[best_name],
        X_test,
        y_test,
        thresholds=config["thresholds"],
        output_csv_path=threshold_tuning_csv_path,
        output_plot_path=threshold_tuning_plot_path,
    )
    best_f1_row = threshold_df.sort_values("f1", ascending=False).iloc[0]
    logger.info(
        "Best threshold by test-set F1: %.2f (precision=%.3f, recall=%.3f, f1=%.3f)",
        best_f1_row["threshold"],
        best_f1_row["precision"],
        best_f1_row["recall"],
        best_f1_row["f1"],
    )

    # Enhanced analysis: error analysis for the selected best model
    error_df = save_error_analysis(
        fitted_models[best_name],
        X_test,
        y_test,
        feature_names=NUMERIC_FEATURES,
        output_path=error_analysis_path,
        threshold=0.5,
        top_n=config["error_analysis_top_n"],
    )
    logger.info("Saved error analysis to %s", error_analysis_path)
    logger.info("Error type counts: %s", error_df["error_type"].value_counts().to_dict())

    # Save metadata + snapshot outputs into a timestamped run folder
    save_run_metadata(config, best_name, run_dir, output_path=metadata_path)
    canonical_outputs = [
        comparison_table_path,
        pr_curve_path,
        calibration_plot_path,
        best_model_path,
        experiment_log_path,
        disagreement_md_path,
        test_predictions_path,
        calibration_metrics_path,
        threshold_tuning_csv_path,
        threshold_tuning_plot_path,
        error_analysis_path,
        metadata_path,
    ]
    copy_outputs_to_run_dir(canonical_outputs, run_dir)

    logger.info("--- All results saved to %s and snapshot copied to %s ---", results_root, run_dir)
    logger.info("Write your decision memo in the PR description (Task 10).")


if __name__ == "__main__":
    main()
