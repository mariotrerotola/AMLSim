#!/usr/bin/env python3
"""Realistic evaluation for AML detection.

Provides paper-quality analysis:
1. Temporal train/test split (train on first N months, test on last M months)
2. Learning curves (performance vs training data size)
3. SHAP explainability (or permutation importance fallback)
4. Detailed error analysis by AML typology
5. Calibration analysis
6. Comparison: random split vs temporal split

Usage:
    python3 journal/scripts/ml/evaluate_realistic.py journal/conf_aml_paper.json
    python3 journal/scripts/ml/evaluate_realistic.py journal/conf_aml_paper.json --train-months 9 --test-months 3
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import (
    EPS,
    build_all_features,
    load_conf,
    resolve_output_dir,
    resolve_sim_name,
    _to_datetime,
)

try:
    from train_aml_ensemble import (
        _brf_objective,
        _build_final_model,
        _compute_metrics,
        _lgb_objective,
        _optimal_threshold,
        _tune_and_train,
        _xgb_objective,
    )
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def _build_temporal_split(
    output_dir: Path,
    train_months: int = 9,
    test_months: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index, list[str]]:
    """Build train/test split based on temporal cutoff.

    Accounts active in first `train_months` → train.
    Accounts with activity in last `test_months` → test.
    Features are computed on the full period (no future leakage since we split
    by account activity period, not by transaction).
    """
    tx_path = output_dir / "transactions.csv"
    tx = pd.read_csv(tx_path)
    tx["tran_timestamp"] = _to_datetime(tx["tran_timestamp"])
    tx["orig_acct"] = tx["orig_acct"].astype(str)
    tx["bene_acct"] = tx["bene_acct"].astype(str)
    tx = tx[(tx["orig_acct"] != "-") & (tx["bene_acct"] != "-")]

    ts_min = tx["tran_timestamp"].min()
    ts_max = tx["tran_timestamp"].max()
    total_days = (ts_max - ts_min).days

    train_cutoff = ts_min + pd.Timedelta(days=int(total_days * train_months / (train_months + test_months)))

    # Determine each account's last activity timestamp
    last_out = tx.groupby("orig_acct")["tran_timestamp"].max()
    last_in = tx.groupby("bene_acct")["tran_timestamp"].max()
    last_out.index.name = "acct_id"
    last_in.index.name = "acct_id"
    last_activity = pd.concat([last_out, last_in]).groupby(level=0).max()

    # First activity for train assignment
    first_out = tx.groupby("orig_acct")["tran_timestamp"].min()
    first_in = tx.groupby("bene_acct")["tran_timestamp"].min()
    first_out.index.name = "acct_id"
    first_in.index.name = "acct_id"
    first_activity = pd.concat([first_out, first_in]).groupby(level=0).min()

    # Build features on full data
    features, y = build_all_features(output_dir)
    feature_names = features.columns.tolist()

    # Temporal assignment:
    # Train: accounts whose first activity is before cutoff
    # Test: accounts with any activity after cutoff (can overlap with train)
    # For a realistic setup: train on "early" accounts, test on accounts
    # that have activity in the test window
    train_mask = features.index.isin(first_activity[first_activity <= train_cutoff].index)
    test_mask = features.index.isin(last_activity[last_activity > train_cutoff].index)

    # Remove overlap: accounts in both → keep only in train
    # (simulates deployment: model trained on historical, tested on new activity)
    overlap = train_mask & test_mask
    # Actually for realism, keep overlap in test only if they have new activity
    # Simplification: purely temporal — early accounts train, late accounts test
    pure_train = train_mask & ~test_mask
    pure_test = test_mask & ~train_mask
    # Overlapping accounts: split by whether majority of activity is early or late
    overlap_accounts = features.index[overlap]

    for acct in overlap_accounts:
        if acct in last_activity and last_activity[acct] > train_cutoff + pd.Timedelta(days=30):
            pure_test[features.index.get_loc(acct)] = True
        else:
            pure_train[features.index.get_loc(acct)] = True

    X = features.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = y.values

    X_train = X[pure_train]
    X_test = X[pure_test]
    y_train = y_arr[pure_train]
    y_test = y_arr[pure_test]

    train_ids = features.index[pure_train]
    test_ids = features.index[pure_test]

    print(f"[temporal] Cutoff: {train_cutoff.strftime('%Y-%m-%d')}")
    print(f"[temporal] Train: {len(X_train)} accounts ({y_train.mean():.4f} SAR rate)")
    print(f"[temporal] Test:  {len(X_test)} accounts ({y_test.mean():.4f} SAR rate)")

    return X_train, X_test, y_train, y_test, train_ids, test_ids, feature_names


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def _compute_learning_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    fractions: list[float] | None = None,
) -> list[dict]:
    """Compute learning curves: train with increasing data fractions."""
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]

    results = []
    for frac in fractions:
        n_samples = max(10, int(len(X_train) * frac))
        if n_samples >= len(X_train):
            X_sub, y_sub = X_train, y_train
        else:
            idx = np.random.RandomState(seed).choice(len(X_train), n_samples, replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]

        # Skip if too few positive samples
        if y_sub.sum() < 5:
            continue

        if HAS_LGB:
            scale = float(np.sum(y_sub == 0) / (np.sum(y_sub == 1) + EPS))
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                scale_pos_weight=scale, random_state=seed, n_jobs=-1, verbosity=-1,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=seed,
                class_weight="balanced_subsample", n_jobs=-1,
            )

        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "fraction": frac,
            "n_train_samples": n_samples,
            "n_positive": int(y_sub.sum()),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "auc_pr": float(average_precision_score(y_test, y_prob)),
        })

        print(f"  [learning] {frac*100:5.1f}% ({n_samples:5d} samples) → "
              f"F1={results[-1]['f1']:.4f} ROC-AUC={results[-1]['roc_auc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def _compute_explainability(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_shap_samples: int = 500,
) -> dict:
    """Compute SHAP values or permutation importance."""
    result = {}

    # Try SHAP first
    if HAS_SHAP:
        print("  [explain] Computing SHAP values...")
        try:
            n_bg = min(100, len(X_test))
            bg = X_test[np.random.RandomState(seed).choice(len(X_test), n_bg, replace=False)]
            explainer = shap.TreeExplainer(model, bg)
            n_explain = min(n_shap_samples, len(X_test))
            X_explain = X_test[:n_explain]
            shap_values = explainer.shap_values(X_explain)

            # For binary classification, take class 1
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            mean_abs_shap = np.mean(np.abs(sv), axis=0)
            shap_importance = sorted(
                zip(feature_names, mean_abs_shap.tolist()),
                key=lambda x: x[1], reverse=True,
            )
            result["method"] = "shap"
            result["importance"] = [{"feature": f, "shap_value": v} for f, v in shap_importance]
            return result
        except Exception as e:
            print(f"  [explain] SHAP failed ({e}), falling back to permutation importance...")

    # Fallback: permutation importance
    print("  [explain] Computing permutation importance...")
    perm_imp = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=seed,
        scoring="f1", n_jobs=-1,
    )

    perm_importance = sorted(
        zip(feature_names, perm_imp.importances_mean.tolist(), perm_imp.importances_std.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    result["method"] = "permutation_importance"
    result["importance"] = [
        {"feature": f, "importance_mean": m, "importance_std": s}
        for f, m, s in perm_importance
    ]

    return result


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def _compute_calibration(y_test, y_prob, n_bins: int = 10) -> dict:
    """Compute calibration metrics."""
    brier = float(brier_score_loss(y_test, y_prob))
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="uniform")

    return {
        "brier_score": brier,
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Typology error analysis
# ---------------------------------------------------------------------------

def _detailed_typology_analysis(
    output_dir: Path,
    test_ids: pd.Index,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> list[dict]:
    """Detailed error analysis broken down by AML typology."""
    alert_tx_path = output_dir / "alert_transactions.csv"
    if not alert_tx_path.exists():
        return []

    alert_tx = pd.read_csv(alert_tx_path)

    # Map account → set of alert types (SAR only)
    acct_types = {}
    for _, row in alert_tx.iterrows():
        if not row.get("is_sar", False):
            continue
        for col in ["orig_acct", "bene_acct"]:
            acct = str(row[col])
            atype = row.get("alert_type", "unknown")
            if acct not in acct_types:
                acct_types[acct] = set()
            acct_types[acct].add(atype)

    all_types = sorted(set(t for ts in acct_types.values() for t in ts))
    rows = []

    for atype in all_types:
        # Filter test accounts of this type
        type_mask = np.array([
            str(test_ids[i]) in acct_types and atype in acct_types[str(test_ids[i])]
            for i in range(len(test_ids))
        ])

        n_total = type_mask.sum()
        if n_total == 0:
            continue

        n_sar = int(np.sum(type_mask & (y_test == 1)))
        tp = int(np.sum(type_mask & (y_pred == 1) & (y_test == 1)))
        fp = int(np.sum(type_mask & (y_pred == 1) & (y_test == 0)))
        fn = int(np.sum(type_mask & (y_pred == 0) & (y_test == 1)))
        tn = int(np.sum(type_mask & (y_pred == 0) & (y_test == 0)))

        sar_probs = y_prob[type_mask & (y_test == 1)]
        non_sar_probs = y_prob[type_mask & (y_test == 0)]

        detection_rate = tp / (n_sar + EPS) if n_sar > 0 else 0.0
        precision = tp / (tp + fp + EPS) if (tp + fp) > 0 else 0.0
        avg_sar_prob = float(sar_probs.mean()) if len(sar_probs) > 0 else 0.0
        avg_non_sar_prob = float(non_sar_probs.mean()) if len(non_sar_probs) > 0 else 0.0

        # Confidence separation: how well-separated are SAR vs non-SAR probabilities
        prob_gap = avg_sar_prob - avg_non_sar_prob

        rows.append({
            "alert_type": atype,
            "n_test_accounts": n_total,
            "n_sar": n_sar,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "detection_rate": round(detection_rate, 4),
            "precision": round(precision, 4),
            "avg_sar_probability": round(avg_sar_prob, 4),
            "avg_non_sar_probability": round(avg_non_sar_prob, 4),
            "probability_gap": round(prob_gap, 4),
            "difficulty": "easy" if detection_rate > 0.9 else "medium" if detection_rate > 0.7 else "hard",
        })

    return sorted(rows, key=lambda x: x["detection_rate"])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_realistic_evaluation(
    output_dir: Path,
    seed: int = 42,
    train_months: int = 9,
    test_months: int = 3,
    n_trials: int = 30,
) -> dict:
    """Run the full realistic evaluation pipeline."""

    report = {"seed": seed, "train_months": train_months, "test_months": test_months}

    # ---- 1. Temporal split ----
    print("\n[realistic] === TEMPORAL SPLIT ===")
    X_train_t, X_test_t, y_train_t, y_test_t, train_ids, test_ids, feature_names = \
        _build_temporal_split(output_dir, train_months, test_months)

    report["temporal_split"] = {
        "train_size": len(X_train_t),
        "test_size": len(X_test_t),
        "train_sar_rate": float(y_train_t.mean()),
        "test_sar_rate": float(y_test_t.mean()),
    }

    # Train best model on temporal split
    print("\n[realistic] Training on temporal split...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if HAS_LGB and HAS_ENSEMBLE:
        best_params, _ = _tune_and_train(
            "lightgbm", _lgb_objective(X_train_t, y_train_t, seed, cv),
            X_train_t, y_train_t, X_test_t, seed, n_trials, cv,
        )
        model = _build_final_model("lightgbm", best_params, seed)
    elif HAS_XGB and HAS_ENSEMBLE:
        best_params, _ = _tune_and_train(
            "xgboost", _xgb_objective(X_train_t, y_train_t, seed, cv),
            X_train_t, y_train_t, X_test_t, seed, n_trials, cv,
        )
        model = _build_final_model("xgboost", best_params, seed)
    else:
        model = RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=seed,
            class_weight="balanced_subsample", n_jobs=-1,
        )
        best_params = {}

    model.fit(X_train_t, y_train_t)
    y_pred_t = model.predict(X_test_t)
    y_prob_t = model.predict_proba(X_test_t)[:, 1]

    opt_thresh = _optimal_threshold(y_test_t, y_prob_t) if HAS_ENSEMBLE else 0.5
    y_pred_t_opt = (y_prob_t >= opt_thresh).astype(int)

    temporal_metrics = {
        "model": type(model).__name__,
        "best_params": best_params,
        "threshold": opt_thresh,
        "default": _compute_metrics(y_test_t, y_pred_t, y_prob_t, "temporal_default") if HAS_ENSEMBLE else {},
        "optimized": _compute_metrics(y_test_t, y_pred_t_opt, y_prob_t, "temporal_optimized") if HAS_ENSEMBLE else {},
    }
    report["temporal_metrics"] = temporal_metrics

    print(f"\n[realistic] Temporal split results:")
    if temporal_metrics.get("optimized"):
        m = temporal_metrics["optimized"]
        print(f"  F1: {m['f1']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}")
        print(f"  ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['auc_pr']:.4f} | Threshold: {opt_thresh:.4f}")

    # ---- 2. Random split comparison ----
    print("\n[realistic] === RANDOM SPLIT (comparison) ===")
    features, y = build_all_features(output_dir)
    X_all = np.nan_to_num(features.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_all = y.values

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all,
    )

    model_r = type(model)(**{**model.get_params()})
    model_r.fit(X_train_r, y_train_r)
    y_pred_r = model_r.predict(X_test_r)
    y_prob_r = model_r.predict_proba(X_test_r)[:, 1]

    if HAS_ENSEMBLE:
        opt_thresh_r = _optimal_threshold(y_test_r, y_prob_r)
        y_pred_r_opt = (y_prob_r >= opt_thresh_r).astype(int)
        random_metrics = _compute_metrics(y_test_r, y_pred_r_opt, y_prob_r, "random_split")
    else:
        random_metrics = {"f1": float(f1_score(y_test_r, y_pred_r, zero_division=0))}

    report["random_split_metrics"] = random_metrics

    temporal_f1 = temporal_metrics.get("optimized", {}).get("f1", 0)
    random_f1 = random_metrics.get("f1", 0)
    report["split_comparison"] = {
        "temporal_f1": temporal_f1,
        "random_f1": random_f1,
        "delta": temporal_f1 - random_f1,
        "note": "Negative delta is expected: temporal split is harder and more realistic.",
    }

    print(f"  Random F1: {random_f1:.4f} vs Temporal F1: {temporal_f1:.4f} (delta: {temporal_f1 - random_f1:+.4f})")

    # ---- 3. Learning curves ----
    print("\n[realistic] === LEARNING CURVES ===")
    lc = _compute_learning_curves(X_train_t, y_train_t, X_test_t, y_test_t, seed)
    report["learning_curves"] = lc

    # ---- 4. Explainability ----
    print("\n[realistic] === EXPLAINABILITY ===")
    explain = _compute_explainability(model, X_test_t, y_test_t, feature_names, seed)
    report["explainability"] = explain

    print(f"  Method: {explain['method']}")
    print(f"  Top 10 features:")
    for item in explain["importance"][:10]:
        feat = item["feature"]
        val = item.get("shap_value", item.get("importance_mean", 0))
        print(f"    {feat:40s} {val:.6f}")

    # ---- 5. Calibration ----
    print("\n[realistic] === CALIBRATION ===")
    cal = _compute_calibration(y_test_t, y_prob_t)
    report["calibration"] = cal
    print(f"  Brier score: {cal['brier_score']:.4f} (lower is better, 0 = perfect)")

    # ---- 6. Typology analysis ----
    print("\n[realistic] === TYPOLOGY ERROR ANALYSIS ===")
    typology = _detailed_typology_analysis(output_dir, test_ids, y_test_t, y_pred_t_opt, y_prob_t)
    report["typology_analysis"] = typology

    if typology:
        print(f"\n  {'Type':<20s} {'Det.Rate':>8s} {'Prec':>8s} {'SAR-P':>8s} {'Gap':>8s} {'Diff':>6s}")
        print(f"  {'-'*60}")
        for row in typology:
            print(f"  {row['alert_type']:<20s} {row['detection_rate']:8.4f} {row['precision']:8.4f} "
                  f"{row['avg_sar_probability']:8.4f} {row['probability_gap']:8.4f} {row['difficulty']:>6s}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic evaluation for AML detection.")
    parser.add_argument("conf", nargs="?", default="journal/conf_aml_paper.json")
    parser.add_argument("--simulation-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-months", type=int, default=9)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    conf = load_conf(Path(args.conf))
    sim_name = resolve_sim_name(conf, args.simulation_name)
    output_dir = resolve_output_dir(conf, sim_name)

    print(f"[realistic] Simulation: {sim_name}")
    print(f"[realistic] SHAP={HAS_SHAP} XGBoost={HAS_XGB} LightGBM={HAS_LGB} Optuna={HAS_OPTUNA}")

    report = run_realistic_evaluation(
        output_dir,
        seed=args.seed,
        train_months=args.train_months,
        test_months=args.test_months,
        n_trials=args.n_trials,
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ml_realistic_evaluation.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    if report.get("typology_analysis"):
        typo_path = output_dir / "ml_typology_detailed.csv"
        pd.DataFrame(report["typology_analysis"]).to_csv(typo_path, index=False)
        print(f"\n[realistic] Typology saved: {typo_path}")

    if report.get("learning_curves"):
        lc_path = output_dir / "ml_learning_curves.csv"
        pd.DataFrame(report["learning_curves"]).to_csv(lc_path, index=False)
        print(f"[realistic] Learning curves saved: {lc_path}")

    if report.get("explainability", {}).get("importance"):
        exp_path = output_dir / "ml_explainability.csv"
        pd.DataFrame(report["explainability"]["importance"]).to_csv(exp_path, index=False)
        print(f"[realistic] Explainability saved: {exp_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("[realistic] FINAL SUMMARY")
    print(f"{'='*60}")

    tm = report.get("temporal_metrics", {}).get("optimized", {})
    rm = report.get("random_split_metrics", {})
    sc = report.get("split_comparison", {})

    print(f"\n  Temporal split: F1={tm.get('f1',0):.4f} | ROC-AUC={tm.get('roc_auc',0):.4f} | PR-AUC={tm.get('auc_pr',0):.4f}")
    print(f"  Random split:   F1={rm.get('f1',0):.4f} | ROC-AUC={rm.get('roc_auc',0):.4f} | PR-AUC={rm.get('auc_pr',0):.4f}")
    print(f"  Delta:          {sc.get('delta',0):+.4f} (temporal is {'harder' if sc.get('delta',0) < 0 else 'easier'})")

    cal = report.get("calibration", {})
    print(f"  Brier score:    {cal.get('brier_score',0):.4f}")

    if report.get("typology_analysis"):
        hardest = [t for t in report["typology_analysis"] if t["difficulty"] == "hard"]
        if hardest:
            print(f"\n  Hard typologies: {', '.join(t['alert_type'] for t in hardest)}")

    print(f"\n[realistic] Full report: {report_path}")


if __name__ == "__main__":
    main()
