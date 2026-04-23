# Petra Telecom Churn Prediction — Model Comparison Workflow

A reproducible machine learning workflow for comparing baseline, linear, and tree-based classifiers on Petra Telecom churn data.

This repository contains two implementations:

- **`model_comparison.py`** — the original assignment solution that completes the required 9 tasks and writes the required assignment artifacts to **`results/`**.
- **`model_comparison_enhanced.py`** — an extended version that preserves the original structure while adding stronger analysis and engineering features such as threshold tuning, error analysis, calibration metrics, run logging, timestamped run folders, and JSON-based configuration. Its outputs are written to **`results_enhanced/`**.

## Project Goals

This project is designed to answer a practical business question:

> **Which model should Petra Telecom use to rank customers by churn risk, and what tradeoffs come with that decision?**

The workflow compares six classification pipelines, evaluates them with 5-fold stratified cross-validation, generates diagnostic plots, saves the best model, and produces analysis artifacts that support a decision memo.

## Model Families Compared

The base workflow compares these six model configurations:

1. **Dummy** — majority-class baseline
2. **LR_default** — logistic regression with standard scaling
3. **LR_balanced** — class-balanced logistic regression with standard scaling
4. **DT_depth5** — decision tree with max depth 5
5. **RF_default** — random forest with 100 trees and max depth 10
6. **RF_balanced** — class-balanced random forest with 100 trees and max depth 10

## Key Features

### Base workflow (`model_comparison.py`)

- Loads and splits the telecom churn dataset
- Defines 6 sklearn pipelines
- Runs 5-fold stratified cross-validation
- Reports mean and standard deviation for:
  - Accuracy
  - Precision
  - Recall
  - F1
  - PR-AUC
- Saves a model comparison table
- Plots precision-recall curves for the top 3 models
- Plots calibration curves for the top 3 models
- Saves the best model with `joblib`
- Logs experiment results to CSV
- Finds a sample where random forest and logistic regression disagree most

### Enhanced workflow (`model_comparison_enhanced.py`)

In addition to everything above, the enhanced version adds:

- **Threshold tuning** for the selected best model
- **Error analysis** for false positives and false negatives
- **Full test-set prediction export** for every model
- **Calibration metrics** (`Brier score` and `ECE`)
- **Structured logging** to console and file
- **Timestamped run folders** for experiment snapshots
- **Config-driven execution** with optional `config.json`
- **`ModelSelector` helper** for cleaner model orchestration

## Repository Structure

```text
.
├── data/
│   └── telecom_churn.csv
├── model_comparison.py
├── model_comparison_enhanced.py
├── config.example.json
├── setup.sh
├── results/
│   ├── comparison_table.csv
│   ├── pr_curves.png
│   ├── calibration.png
│   ├── best_model.joblib
│   ├── experiment_log.csv
│   └── tree_vs_linear_disagreement.md
├── results_enhanced/
│   ├── comparison_table.csv
│   ├── pr_curves.png
│   ├── calibration.png
│   ├── best_model.joblib
│   ├── experiment_log.csv
│   ├── tree_vs_linear_disagreement.md
│   ├── test_predictions.csv
│   ├── calibration_metrics.csv
│   ├── threshold_tuning.csv
│   ├── threshold_sweep.png
│   ├── error_analysis.md
│   ├── run_metadata.json
│   └── runs/
│       └── <timestamp>/
│           ├── run.log
│           └── ...snapshot of outputs...
└── README.md
```

## Installation

You can set up the project in either of these ways.

### Option 1 — Manual setup

Create and activate a virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2 — Setup script

A convenience script is included to automate environment setup.

```bash
bash setup.sh
```

If the script needs execute permission first:

```bash
chmod +x setup.sh
./setup.sh
```

After setup finishes, activate the virtual environment if needed:

```bash
source .venv/bin/activate
```

## How to Run

### Run the base version

```bash
python model_comparison.py
```

This writes the required assignment outputs to:

```text
results/
```

### Run the enhanced version

```bash
python model_comparison_enhanced.py
```

This writes the enhanced outputs to:

```text
results_enhanced/
```

## Optional Configuration

The enhanced workflow supports an optional `config.json` file placed next to the script.

Start from the provided example:

```bash
cp config.example.json config.json
```

Then edit values such as:

- `random_state`
- `data_path`
- `results_root`
- `n_splits`
- `selection_metric`
- `calibration_bins`
- `error_analysis_top_n`
- `thresholds`
- model definitions and hyperparameters

### Example

```json
{
  "random_state": 42,
  "results_root": "results_enhanced",
  "n_splits": 5,
  "selection_metric": "pr_auc_mean"
}
```

## Methodology

### 1. Data preparation

The workflow uses the following numeric features:

- `tenure`
- `monthly_charges`
- `total_charges`
- `num_support_calls`
- `senior_citizen`
- `has_partner`
- `has_dependents`
- `contract_months`

The target is:

- `churned`

The data is split into train and test sets using an **80/20 stratified split**.

### 2. Cross-validation

All six pipelines are evaluated with **5-fold stratified cross-validation**.

The workflow records:

- mean score across folds
- standard deviation across folds

This helps compare both **performance** and **stability**.

### 3. Ranking metric

The main model-selection metric is **PR-AUC** because churn prediction is an imbalanced classification problem and PR-AUC focuses on positive-class ranking quality.

### 4. Diagnostics

The workflow includes multiple diagnostic layers:

- **PR curves** to inspect precision–recall tradeoffs
- **Calibration curves** to inspect probability reliability
- **Threshold tuning** to inspect decision-threshold tradeoffs
- **Error analysis** to inspect false positives and false negatives
- **Tree-vs-linear disagreement analysis** to show structural differences between model families

## Generated Outputs

### Required assignment outputs

These are produced by `model_comparison.py` and saved in `results/`:

- `results/comparison_table.csv`
- `results/pr_curves.png`
- `results/calibration.png`
- `results/best_model.joblib`
- `results/experiment_log.csv`
- `results/tree_vs_linear_disagreement.md`

### Enhanced outputs

These are produced by `model_comparison_enhanced.py` and saved in `results_enhanced/`:

- `results_enhanced/comparison_table.csv`
- `results_enhanced/pr_curves.png`
- `results_enhanced/calibration.png`
- `results_enhanced/best_model.joblib`
- `results_enhanced/experiment_log.csv`
- `results_enhanced/tree_vs_linear_disagreement.md`
- `results_enhanced/test_predictions.csv`
- `results_enhanced/calibration_metrics.csv`
- `results_enhanced/threshold_tuning.csv`
- `results_enhanced/threshold_sweep.png`
- `results_enhanced/error_analysis.md`
- `results_enhanced/run_metadata.json`
- `results_enhanced/runs/<timestamp>/run.log`

## Results vs. Results Enhanced

The `results/` directory contains the required base-task outputs produced by `model_comparison.py`, including the comparison table, PR curves, calibration plot, saved best model, experiment log, and tree-vs-linear disagreement analysis. The `results_enhanced/` directory contains the outputs of `model_comparison_enhanced.py`, which preserves the same core workflow and model selection outcome while adding extra analysis layers such as threshold tuning, calibration metrics, full test-set prediction exports, error analysis, run metadata, and timestamped run folders. In other words, `results/` is the assignment submission output, while `results_enhanced/` is an extended analysis and engineering version built on top of the same underlying comparison pipeline.

## Example Interpretation

A typical run of the project leads to conclusions like these:

- **Random forest** can achieve the strongest **PR-AUC**, making it the best model for ranking customers by churn risk.
- **Balanced models** often increase **recall** at the default 0.5 threshold, but this does not always improve **threshold-independent ranking quality**.
- **Calibration analysis** helps determine whether model probabilities can support prioritization and early warning.
- **Threshold tuning** is useful when business capacity is limited and the retention team cannot contact every predicted churner.

## Suggested Structural Explanation for `results_enhanced/tree_vs_linear_disagreement.md`

The disagreement is likely driven by a threshold-style pattern around `contract_months = 1`, which the random forest can treat as a strong churn signal when combined with the rest of the feature profile. In this sample, the tree model assigned a much higher churn probability than logistic regression, suggesting that the tree captured a rule-like interaction that the linear model smoothed out through additive feature effects. This interpretation is also consistent with the enhanced error analysis, where the same sample appeared among the highest-confidence false positives, showing that the random forest can sometimes overreact to short-contract risk patterns even when the true label is non-churn.

## Why the Enhanced Version Exists

The original script solves the assignment requirements and remains the main deliverable.

The enhanced script was added to extend the project in a more production-like direction without replacing the required submission. It keeps the original workflow structure but makes the project more reusable, traceable, and analytically rich.

## Suggested Next Improvements

If you want to extend the project further, good next steps include:

- permutation importance
- nested cross-validation
- hyperparameter search
- probability calibration methods such as Platt scaling or isotonic regression
- richer model registry support in the config workflow
- deployment packaging for inference

## Notes

- Generated result files are typically ignored by Git through `.gitignore`.
- The enhanced workflow writes its canonical outputs to `results_enhanced/`, with timestamped snapshots saved under `results_enhanced/runs/`.
- The original `model_comparison.py` file remains part of the repository and is not replaced by the enhanced version.
- `setup.sh` is provided as a convenience helper for local environment setup.
