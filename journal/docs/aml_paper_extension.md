# AML Extension: Dataset + Baseline + Ensemble (AMLSim)

This guide extends the article workflow to an AML scenario using synthetic data generated in this repository.

## 1) Generate an AML-focused dataset

A dedicated preset is provided:

- Config: `journal/conf_aml_paper.json`
- Parameters: `journal/paramFiles/aml_paper/`

It includes:

- 12,043 accounts across 4 banks (`bank_a`..`bank_d`)
- mixed normal behaviors (`single`, `fan_in`, `fan_out`, `forward`, `mutual`, `periodical`)
- multiple AML typologies (`fan_in`, `fan_out`, `cycle`, `stack`, `bipartite`, `gather_scatter`, `scatter_gather`)
- both SAR and non-SAR alerts for realism

Run:

```bash
sh scripts/run_batch.sh journal/conf_aml_paper.json
```

Outputs are created under:

- `journal/outputs/aml_paper/transactions.csv`
- `journal/outputs/aml_paper/alert_transactions.csv`
- `journal/outputs/aml_paper/sar_accounts.csv`
- `journal/outputs/aml_paper/accounts.csv`

## 2) Train a baseline classifier

Script:

- `journal/scripts/ml/train_aml_brf_baseline.py`

Run:

```bash
python3 journal/scripts/ml/train_aml_brf_baseline.py journal/conf_aml_paper.json
```

Generated artifacts:

- `journal/outputs/aml_paper/ml_baseline_metrics.json`
- `journal/outputs/aml_paper/ml_feature_importance.csv`
- `journal/outputs/aml_paper/ml_account_features.csv`

### Baseline model

- Primary: `BalancedRandomForestClassifier` (if `imbalanced-learn` is installed)
- Fallback: `RandomForestClassifier(class_weight='balanced_subsample')`
- 34 hand-crafted features (structural, monetary, temporal, cash behavior)
- Baseline F1 (SAR class): ~0.74, Macro-F1: ~0.83

## 3) Train the top-tier ensemble

Script:

- `journal/scripts/ml/train_aml_ensemble.py`

Run:

```bash
python3 journal/scripts/ml/train_aml_ensemble.py journal/conf_aml_paper.json
```

Options:

```bash
# Quick run (no Optuna tuning)
python3 journal/scripts/ml/train_aml_ensemble.py journal/conf_aml_paper.json --n-trials 0

# Full tuning (100 trials per model, default)
python3 journal/scripts/ml/train_aml_ensemble.py journal/conf_aml_paper.json --n-trials 100
```

Generated artifacts:

- `journal/outputs/aml_paper/ml_ensemble_metrics.json`
- `journal/outputs/aml_paper/ml_ensemble_feature_importance.csv`
- `journal/outputs/aml_paper/ml_ensemble_predictions.csv`
- `journal/outputs/aml_paper/ml_ensemble_account_features.csv`

### Ensemble architecture

**Feature engineering** (~80+ features):

- Baseline (34): structural, monetary, temporal, cash behavior
- Graph/network (12): PageRank, betweenness/degree centrality, clustering coefficient, Louvain community, k-core number, reciprocity, 2-hop reach
- Temporal windows (18): 7/30/90-day volume and counts, trend ratios (early vs late period), day-of-week entropy, temporal entropy
- Transaction patterns (18): round amount ratios, Gini coefficient, Herfindahl index, amount CV/skewness, max/avg ratio, transaction type entropy

**Models** (Optuna-tuned):

- XGBoost with scale_pos_weight
- LightGBM with is_unbalance
- Balanced Random Forest

**Stacking ensemble:**

- Cross-validated predictions from all base models (avoids data leakage)
- LogisticRegression meta-learner with balanced class weights
- Per-model and ensemble threshold optimization for F1

### Dependencies

```bash
pip install xgboost lightgbm optuna scikit-learn imbalanced-learn pandas scipy
```

## 4) Multi-seed robustness evaluation

Script:

- `journal/scripts/ml/evaluate_robustness.py`

Run:

```bash
# Default: 5 seeds (42, 1337, 2025, 7, 123)
python3 journal/scripts/ml/evaluate_robustness.py journal/conf_aml_paper.json

# Custom seeds, reduced tuning budget
python3 journal/scripts/ml/evaluate_robustness.py journal/conf_aml_paper.json --seeds 42 1337 2025 --n-trials 30
```

Generated artifacts:

- `journal/outputs/aml_paper/ml_robustness_report.json` — mean/std metrics across seeds, paired t-test vs baseline
- `journal/outputs/aml_paper/ml_typology_analysis.csv` — per-typology detection rates

### What it evaluates

- Baseline (BRF) and ensemble across N seeds
- Statistical comparison (paired t-test, p-values)
- Detection breakdown by AML typology (fan_in, fan_out, cycle, stack, etc.)
- Identifies which AML patterns are hardest to detect

## 5) Module structure

```
journal/scripts/ml/
├── feature_engineering.py       # Shared feature builder (80+ features)
├── train_aml_brf_baseline.py    # Baseline BRF (backward compatible)
├── train_aml_ensemble.py        # Multi-model ensemble + Optuna
└── evaluate_robustness.py       # Multi-seed evaluation + typology
```

All scripts share the `feature_engineering.py` module, which provides:

- `build_all_features(output_dir)` — full 80+ feature set
- `build_baseline_features_only(output_dir)` — original 34 features
- `load_conf()`, `resolve_sim_name()`, `resolve_output_dir()` — config helpers

## 6) Suggested paper protocol

Use 3+ reproducible scenarios by changing only `--seed`:

```bash
for SEED in 42 1337 2025; do
    python3 journal/scripts/ml/train_aml_ensemble.py journal/conf_aml_paper.json --seed $SEED
done

# Or use the robustness evaluator for automated comparison:
python3 journal/scripts/ml/evaluate_robustness.py journal/conf_aml_paper.json --seeds 42 1337 2025
```

Report format: mean +/- std across seeds, with significance tests.
