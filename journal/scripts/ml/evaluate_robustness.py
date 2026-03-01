#!/usr/bin/env python3
"""Multi-seed robustness evaluation and AML typology analysis.

Runs the ensemble pipeline across multiple seeds, computes statistical
summaries, and breaks down detection performance by AML typology.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import build_all_features, build_baseline_features_only, load_conf, resolve_output_dir, resolve_sim_name

try:
    from train_aml_ensemble import run_ensemble_pipeline
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def _extract_best_metrics(summary: dict) -> dict:
    """Extract key metrics from ensemble summary for the best model."""
    best_name = summary.get("best_model", "")
    best = summary.get("models", {}).get(best_name, {})
    return {
        "model": best_name,
        "f1": best.get("f1", 0.0),
        "f1_macro": best.get("f1_macro", 0.0),
        "precision": best.get("precision", 0.0),
        "recall": best.get("recall", 0.0),
        "balanced_accuracy": best.get("balanced_accuracy", 0.0),
        "roc_auc": best.get("roc_auc", 0.0),
        "auc_pr": best.get("auc_pr", 0.0),
    }


def _run_baseline_seed(output_dir: Path, seed: int) -> dict:
    """Run baseline (BRF only) for a given seed and return metrics."""
    features, y = build_baseline_features_only(output_dir)
    X = features.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=0.2, random_state=seed, stratify=y_arr,
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=20, random_state=seed,
        class_weight="balanced_subsample", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": "baseline_brf",
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "auc_pr": float(average_precision_score(y_test, y_prob)),
    }


def evaluate_multi_seed(
    output_dir: Path,
    seeds: list[int],
    n_trials: int = 50,
) -> dict:
    """Run ensemble + baseline across multiple seeds and compute statistics."""

    ensemble_results = []
    baseline_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[robustness] Seed {seed}")
        print(f"{'='*60}")

        # Baseline
        print(f"  [baseline] Training seed={seed}...")
        bl = _run_baseline_seed(output_dir, seed)
        baseline_results.append(bl)
        print(f"  [baseline] F1={bl['f1']:.4f} ROC-AUC={bl['roc_auc']:.4f}")

        # Ensemble
        if HAS_ENSEMBLE:
            print(f"  [ensemble] Training seed={seed}...")
            summary, _, _, _, _ = run_ensemble_pipeline(
                output_dir, seed=seed, n_trials=n_trials,
            )
            ens = _extract_best_metrics(summary)
            ensemble_results.append(ens)
            print(f"  [ensemble] F1={ens['f1']:.4f} ROC-AUC={ens['roc_auc']:.4f}")

    # Aggregate statistics
    metric_keys = ["f1", "f1_macro", "precision", "recall", "balanced_accuracy", "roc_auc", "auc_pr"]

    def _aggregate(results):
        agg = {}
        for key in metric_keys:
            values = [r[key] for r in results]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
            agg[f"{key}_values"] = values
        return agg

    report = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "baseline": _aggregate(baseline_results),
        "baseline_per_seed": baseline_results,
    }

    if ensemble_results:
        report["ensemble"] = _aggregate(ensemble_results)
        report["ensemble_per_seed"] = ensemble_results

        # Paired t-test: ensemble vs baseline
        comparisons = {}
        for key in metric_keys:
            bl_vals = [r[key] for r in baseline_results]
            ens_vals = [r[key] for r in ensemble_results]
            if len(bl_vals) >= 3 and len(ens_vals) >= 3:
                t_stat, p_val = scipy_stats.ttest_rel(ens_vals, bl_vals)
                comparisons[key] = {
                    "delta_mean": float(np.mean(ens_vals) - np.mean(bl_vals)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "significant_at_005": bool(p_val < 0.05),
                }
            else:
                comparisons[key] = {
                    "delta_mean": float(np.mean(ens_vals) - np.mean(bl_vals)),
                    "note": "Not enough seeds for t-test (need >= 3)",
                }
        report["comparison_ensemble_vs_baseline"] = comparisons

    return report


# ---------------------------------------------------------------------------
# Typology analysis
# ---------------------------------------------------------------------------

def analyze_typologies(output_dir: Path, seed: int = 42, n_trials: int = 50) -> pd.DataFrame:
    """Break down detection performance by AML alert typology."""

    alert_tx_path = output_dir / "alert_transactions.csv"
    sar_path = output_dir / "sar_accounts.csv"

    if not alert_tx_path.exists():
        print("[typology] alert_transactions.csv not found, skipping typology analysis.")
        return pd.DataFrame()

    alert_tx = pd.read_csv(alert_tx_path)
    sar = pd.read_csv(sar_path)

    # Build features and train model
    features, y = build_all_features(output_dir)
    X = features.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = y.values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_arr, np.arange(len(features)),
        test_size=0.2, random_state=seed, stratify=y_arr,
    )

    # Train a simple tuned model for typology analysis
    model = RandomForestClassifier(
        n_estimators=500, max_depth=25, random_state=seed,
        class_weight="balanced_subsample", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Map test accounts to their alert types
    test_acct_ids = features.index[idx_test]

    # Build account → alert_type mapping from alert_transactions
    # An account can appear in multiple alerts; take all unique types
    acct_alert_types = {}
    for _, row in alert_tx.iterrows():
        for acct_col in ["orig_acct", "bene_acct"]:
            acct = str(row[acct_col])
            atype = row.get("alert_type", "unknown")
            is_sar = row.get("is_sar", False)
            if is_sar:
                if acct not in acct_alert_types:
                    acct_alert_types[acct] = set()
                acct_alert_types[acct].add(atype)

    # Build per-typology metrics
    typology_rows = []
    all_types = sorted(set(t for types in acct_alert_types.values() for t in types))

    for alert_type in all_types:
        # Find test accounts involved in this typology
        type_mask = np.array([
            test_acct_ids[i] in acct_alert_types and alert_type in acct_alert_types[test_acct_ids[i]]
            for i in range(len(test_acct_ids))
        ])

        n_type_accounts = type_mask.sum()
        if n_type_accounts == 0:
            continue

        # Among accounts of this type, how many were correctly predicted as SAR?
        tp = int(np.sum(type_mask & (y_pred == 1) & (y_test == 1)))
        fn = int(np.sum(type_mask & (y_pred == 0) & (y_test == 1)))
        total_sar = int(np.sum(type_mask & (y_test == 1)))

        detection_rate = tp / (total_sar + 1e-9) if total_sar > 0 else 0.0
        avg_prob = float(y_prob[type_mask & (y_test == 1)].mean()) if total_sar > 0 else 0.0

        typology_rows.append({
            "alert_type": alert_type,
            "n_test_accounts": int(n_type_accounts),
            "n_sar_in_test": total_sar,
            "true_positives": tp,
            "false_negatives": fn,
            "detection_rate": round(detection_rate, 4),
            "avg_sar_probability": round(avg_prob, 4),
        })

    typology_df = pd.DataFrame(typology_rows)
    if not typology_df.empty:
        typology_df = typology_df.sort_values("detection_rate", ascending=True)

    return typology_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed robustness evaluation for AML detection.")
    parser.add_argument("conf", nargs="?", default="journal/conf_aml_paper.json")
    parser.add_argument("--simulation-name", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025, 7, 123])
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per model per seed.")
    parser.add_argument("--skip-typology", action="store_true")
    args = parser.parse_args()

    conf = load_conf(Path(args.conf))
    sim_name = resolve_sim_name(conf, args.simulation_name)
    output_dir = resolve_output_dir(conf, sim_name)

    print(f"[robustness] Simulation: {sim_name}")
    print(f"[robustness] Seeds: {args.seeds}")
    print(f"[robustness] Ensemble available: {HAS_ENSEMBLE}")

    # Multi-seed evaluation
    report = evaluate_multi_seed(output_dir, seeds=args.seeds, n_trials=args.n_trials)

    # Typology analysis
    if not args.skip_typology:
        print(f"\n{'='*60}")
        print("[robustness] Typology analysis...")
        typology_df = analyze_typologies(output_dir, seed=args.seeds[0], n_trials=args.n_trials)
    else:
        typology_df = pd.DataFrame()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "ml_robustness_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    if not typology_df.empty:
        typology_path = output_dir / "ml_typology_analysis.csv"
        typology_df.to_csv(typology_path, index=False)
        print(f"\n[robustness] Typology analysis saved: {typology_path}")
        print(typology_df.to_string(index=False))

    # Print summary
    print(f"\n{'='*60}")
    print("[robustness] SUMMARY")
    print(f"{'='*60}")

    bl = report["baseline"]
    print(f"\nBaseline (BRF, {len(args.seeds)} seeds):")
    print(f"  F1:       {bl['f1_mean']:.4f} +/- {bl['f1_std']:.4f}")
    print(f"  Macro-F1: {bl['f1_macro_mean']:.4f} +/- {bl['f1_macro_std']:.4f}")
    print(f"  ROC-AUC:  {bl['roc_auc_mean']:.4f} +/- {bl['roc_auc_std']:.4f}")
    print(f"  PR-AUC:   {bl['auc_pr_mean']:.4f} +/- {bl['auc_pr_std']:.4f}")

    if "ensemble" in report:
        ens = report["ensemble"]
        print(f"\nEnsemble ({len(args.seeds)} seeds):")
        print(f"  F1:       {ens['f1_mean']:.4f} +/- {ens['f1_std']:.4f}")
        print(f"  Macro-F1: {ens['f1_macro_mean']:.4f} +/- {ens['f1_macro_std']:.4f}")
        print(f"  ROC-AUC:  {ens['roc_auc_mean']:.4f} +/- {ens['roc_auc_std']:.4f}")
        print(f"  PR-AUC:   {ens['auc_pr_mean']:.4f} +/- {ens['auc_pr_std']:.4f}")

        comp = report.get("comparison_ensemble_vs_baseline", {})
        if comp:
            print(f"\nDelta (ensemble - baseline):")
            for key in ["f1", "roc_auc", "auc_pr"]:
                c = comp.get(key, {})
                delta = c.get("delta_mean", 0)
                sig = c.get("significant_at_005", "N/A")
                p = c.get("p_value", "N/A")
                print(f"  {key}: {delta:+.4f} (p={p}, significant={sig})")

    print(f"\n[robustness] Report saved: {report_path}")


if __name__ == "__main__":
    main()
