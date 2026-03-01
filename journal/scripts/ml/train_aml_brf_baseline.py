#!/usr/bin/env python3
"""Train AML baseline classifier (Balanced Random Forest) on AMLSim outputs.

Uses the shared feature_engineering module for baseline features.
This script is kept for backward compatibility and quick baseline runs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import build_baseline_features_only, load_conf, resolve_output_dir, resolve_sim_name


def build_model(seed: int, n_estimators: int, max_depth: int | None):
    if HAS_IMBLEARN:
        model = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            sampling_strategy="all",
            replacement=True,
            n_jobs=-1,
        )
        model_name = "BalancedRandomForestClassifier"
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model_name = "RandomForestClassifier(class_weight=balanced_subsample)"
    return model, model_name


def train_and_evaluate(features: pd.DataFrame, y: pd.Series, seed: int, n_estimators: int, max_depth: int | None):
    model, model_name = build_model(seed=seed, n_estimators=n_estimators, max_depth=max_depth)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "model": model_name,
        "n_samples": int(len(features)),
        "n_features": int(features.shape[1]),
        "positive_rate": float(y.mean()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "auc_pr": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, features, y, scoring="f1_macro", cv=cv, n_jobs=-1)
    metrics["cv_f1_macro_mean"] = float(np.mean(cv_scores))
    metrics["cv_f1_macro_std"] = float(np.std(cv_scores))

    feature_importance = pd.DataFrame({
        "feature": features.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return metrics, feature_importance


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AML baseline classifier on AMLSim outputs.")
    parser.add_argument("conf", nargs="?", default="journal/conf_aml_paper.json", help="Path to AMLSim config JSON.")
    parser.add_argument("--simulation-name", default=None, help="Override simulation name in conf.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees.")
    parser.add_argument("--max-depth", type=int, default=20, help="Maximum tree depth.")
    args = parser.parse_args()

    conf_path = Path(args.conf)
    conf = load_conf(conf_path)
    sim_name = resolve_sim_name(conf, args.simulation_name)
    output_dir = resolve_output_dir(conf, sim_name)

    if not HAS_IMBLEARN:
        print("[aml-ml] imbalanced-learn not available or incompatible, using RandomForest fallback.")

    features, y = build_baseline_features_only(output_dir)
    metrics, feature_importance = train_and_evaluate(
        features,
        y,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    metrics_path = output_dir / "ml_baseline_metrics.json"
    fi_path = output_dir / "ml_feature_importance.csv"
    dataset_path = output_dir / "ml_account_features.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    feature_importance.to_csv(fi_path, index=False)

    dataset = features.copy()
    dataset["is_sar"] = y.values
    dataset.to_csv(dataset_path)

    print(f"[aml-ml] Simulation: {sim_name}")
    print(f"[aml-ml] Model: {metrics['model']}")
    print(f"[aml-ml] Samples: {metrics['n_samples']} | Features: {metrics['n_features']} | Positive rate: {metrics['positive_rate']:.4f}")
    print(f"[aml-ml] F1: {metrics['f1']:.4f} | Macro-F1: {metrics['f1_macro']:.4f} | BA: {metrics['balanced_accuracy']:.4f}")
    print(f"[aml-ml] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['auc_pr']:.4f}")
    print(f"[aml-ml] CV Macro-F1: {metrics['cv_f1_macro_mean']:.4f} +/- {metrics['cv_f1_macro_std']:.4f}")
    print(f"[aml-ml] Saved: {metrics_path}, {fi_path}, {dataset_path}")


if __name__ == "__main__":
    main()
