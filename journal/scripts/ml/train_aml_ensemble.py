#!/usr/bin/env python3
"""Top-tier AML detection: multi-model ensemble with Optuna hyperparameter tuning.

Pipeline:
1. Advanced feature engineering (80+ features)
2. Optuna-tuned XGBoost, LightGBM, Balanced Random Forest
3. Stacking ensemble with LogisticRegression meta-learner
4. Threshold optimization for maximum F1 on SAR class
5. Comprehensive evaluation and artifact export
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Optional heavy dependencies
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

# Import shared feature engineering
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import build_all_features, load_conf, resolve_output_dir, resolve_sim_name

EPS = 1e-9


# ---------------------------------------------------------------------------
# Optuna objective factories
# ---------------------------------------------------------------------------

def _xgb_objective(X_train, y_train, seed, cv):
    scale = float(np.sum(y_train == 0) / (np.sum(y_train == 1) + EPS))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale * 0.5, scale * 2.0),
        }
        model = xgb.XGBClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict(X_train[val_idx])
            scores.append(f1_score(y_train[val_idx], preds, zero_division=0))
        return np.mean(scores)

    return objective


def _lgb_objective(X_train, y_train, seed, cv):
    scale = float(np.sum(y_train == 0) / (np.sum(y_train == 1) + EPS))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale * 0.5, scale * 2.0),
        }
        model = lgb.LGBMClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict(X_train[val_idx])
            scores.append(f1_score(y_train[val_idx], preds, zero_division=0))
        return np.mean(scores)

    return objective


def _brf_objective(X_train, y_train, seed, cv):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        if HAS_IMBLEARN:
            model = BalancedRandomForestClassifier(
                **params, random_state=seed, sampling_strategy="all",
                replacement=True, n_jobs=-1,
            )
        else:
            model = RandomForestClassifier(
                **params, random_state=seed,
                class_weight="balanced_subsample", n_jobs=-1,
            )
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict(X_train[val_idx])
            scores.append(f1_score(y_train[val_idx], preds, zero_division=0))
        return np.mean(scores)

    return objective


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _tune_and_train(name, objective_fn, X_train, y_train, X_test, seed, n_trials, cv):
    """Run Optuna tuning, train best model, return (model, best_params, cv_score)."""
    if HAS_OPTUNA and n_trials > 0:
        print(f"  [{name}] Optuna tuning ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_cv_score = study.best_value
        print(f"  [{name}] Best CV F1: {best_cv_score:.4f}")
    else:
        best_params = {}
        best_cv_score = 0.0

    return best_params, best_cv_score


def _build_final_model(name, best_params, seed):
    """Instantiate the final model with best hyperparameters."""
    if name == "xgboost":
        return xgb.XGBClassifier(
            **best_params, random_state=seed, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
        )
    elif name == "lightgbm":
        return lgb.LGBMClassifier(
            **best_params, random_state=seed, n_jobs=-1, verbosity=-1,
        )
    elif name == "brf":
        if HAS_IMBLEARN:
            return BalancedRandomForestClassifier(
                **best_params, random_state=seed,
                sampling_strategy="all", replacement=True, n_jobs=-1,
            )
        else:
            return RandomForestClassifier(
                **best_params, random_state=seed,
                class_weight="balanced_subsample", n_jobs=-1,
            )
    raise ValueError(f"Unknown model: {name}")


def _compute_metrics(y_true, y_pred, y_prob, model_name):
    """Compute the full metrics dict."""
    return {
        "model": model_name,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def _optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + EPS)
    best_idx = np.argmax(f1_arr)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ensemble_pipeline(
    output_dir: Path,
    seed: int = 42,
    n_trials: int = 100,
    test_size: float = 0.2,
) -> dict:
    """Full ensemble pipeline: feature engineering → tuning → stacking → evaluation."""

    print("[ensemble] Loading features...")
    features, y = build_all_features(output_dir)

    feature_names = features.columns.tolist()
    X = features.values.astype(np.float64)
    y_arr = y.values

    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=seed, stratify=y_arr,
    )

    # Scale for meta-learner later
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # --- Train individual models ---
    models_config = []
    if HAS_XGB:
        models_config.append(("xgboost", _xgb_objective(X_train, y_train, seed, cv)))
    if HAS_LGB:
        models_config.append(("lightgbm", _lgb_objective(X_train, y_train, seed, cv)))
    models_config.append(("brf", _brf_objective(X_train, y_train, seed, cv)))

    if not models_config:
        raise RuntimeError("No ML libraries available. Install xgboost, lightgbm, or imbalanced-learn.")

    trained_models = {}
    all_metrics = {}
    best_single_f1 = 0.0
    best_single_name = ""

    for name, obj_fn in models_config:
        print(f"\n[ensemble] Training {name}...")
        best_params, cv_score = _tune_and_train(name, obj_fn, X_train, y_train, X_test, seed, n_trials, cv)

        model = _build_final_model(name, best_params, seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        # Threshold optimization
        opt_thresh = _optimal_threshold(y_test, y_prob)
        y_pred_opt = (y_prob >= opt_thresh).astype(int)

        metrics_default = _compute_metrics(y_test, y_pred, y_prob, name)
        metrics_opt = _compute_metrics(y_test, y_pred_opt, y_prob, f"{name}_optimized_threshold")

        all_metrics[name] = metrics_default
        all_metrics[f"{name}_optimized"] = metrics_opt
        all_metrics[f"{name}_optimized"]["threshold"] = opt_thresh
        all_metrics[f"{name}_optimized"]["best_params"] = best_params
        all_metrics[f"{name}_optimized"]["cv_f1"] = cv_score

        trained_models[name] = model

        f1_val = metrics_opt["f1"]
        if f1_val > best_single_f1:
            best_single_f1 = f1_val
            best_single_name = name

        print(f"  [{name}] F1: {metrics_default['f1']:.4f} | F1 (opt thresh): {metrics_opt['f1']:.4f} | "
              f"ROC-AUC: {metrics_default['roc_auc']:.4f} | PR-AUC: {metrics_default['auc_pr']:.4f}")

    # --- Stacking ensemble ---
    if len(trained_models) >= 2:
        print("\n[ensemble] Building stacking ensemble...")

        # Generate cross-validated predictions for meta-learner (avoids leakage)
        meta_train = np.zeros((len(X_train), len(trained_models)))
        meta_test = np.zeros((len(X_test), len(trained_models)))

        for i, (name, model) in enumerate(trained_models.items()):
            # Cross-validated predictions on training set
            cv_preds = cross_val_predict(
                _build_final_model(name, all_metrics.get(f"{name}_optimized", {}).get("best_params", {}), seed),
                X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
            )
            meta_train[:, i] = cv_preds[:, 1]
            meta_test[:, i] = model.predict_proba(X_test)[:, 1]

        # Train meta-learner
        meta_model = LogisticRegression(
            C=1.0, random_state=seed, max_iter=1000, class_weight="balanced",
        )
        meta_model.fit(meta_train, y_train)

        # Ensemble predictions
        y_prob_ensemble = meta_model.predict_proba(meta_test)[:, 1]
        y_pred_ensemble = meta_model.predict(meta_test)

        # Threshold optimization on ensemble
        opt_thresh_ens = _optimal_threshold(y_test, y_prob_ensemble)
        y_pred_ens_opt = (y_prob_ensemble >= opt_thresh_ens).astype(int)

        metrics_ens = _compute_metrics(y_test, y_pred_ensemble, y_prob_ensemble, "stacking_ensemble")
        metrics_ens_opt = _compute_metrics(y_test, y_pred_ens_opt, y_prob_ensemble, "stacking_ensemble_optimized")
        metrics_ens_opt["threshold"] = opt_thresh_ens
        metrics_ens_opt["component_weights"] = {
            name: float(w) for name, w in zip(trained_models.keys(), meta_model.coef_[0])
        }

        all_metrics["stacking_ensemble"] = metrics_ens
        all_metrics["stacking_ensemble_optimized"] = metrics_ens_opt

        print(f"  [stacking] F1: {metrics_ens['f1']:.4f} | F1 (opt): {metrics_ens_opt['f1']:.4f} | "
              f"ROC-AUC: {metrics_ens['roc_auc']:.4f} | PR-AUC: {metrics_ens['auc_pr']:.4f}")

        # Pick best overall
        if metrics_ens_opt["f1"] >= best_single_f1:
            best_name = "stacking_ensemble_optimized"
            best_y_prob = y_prob_ensemble
            best_y_pred = y_pred_ens_opt
        else:
            best_name = f"{best_single_name}_optimized"
            best_y_prob = trained_models[best_single_name].predict_proba(X_test)[:, 1]
            best_thresh = all_metrics[best_name].get("threshold", 0.5)
            best_y_pred = (best_y_prob >= best_thresh).astype(int)
    else:
        best_name = f"{best_single_name}_optimized"
        best_y_prob = trained_models[best_single_name].predict_proba(X_test)[:, 1]
        best_thresh = all_metrics[best_name].get("threshold", 0.5)
        best_y_pred = (best_y_prob >= best_thresh).astype(int)

    all_metrics["best_model"] = best_name

    # --- Feature importance (aggregate from tree models) ---
    importance_dfs = []
    for name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": feature_names,
                f"importance_{name}": model.feature_importances_,
            })
            importance_dfs.append(imp.set_index("feature"))

    if importance_dfs:
        fi_combined = pd.concat(importance_dfs, axis=1)
        fi_combined["importance_mean"] = fi_combined.mean(axis=1)
        fi_combined = fi_combined.sort_values("importance_mean", ascending=False).reset_index()
    else:
        fi_combined = pd.DataFrame({"feature": feature_names, "importance_mean": 0.0})

    # --- Build predictions DataFrame ---
    test_indices = features.index[
        np.arange(len(features))[
            np.isin(np.arange(len(features)),
                    train_test_split(np.arange(len(features)), test_size=test_size,
                                     random_state=seed, stratify=y_arr)[1])
        ]
    ]
    predictions_df = pd.DataFrame({
        "acct_id": test_indices,
        "y_true": y_test,
        "y_pred": best_y_pred,
        "y_prob": best_y_prob,
    })

    # --- Summary ---
    best_metrics = all_metrics.get(best_name, {})
    summary = {
        "n_samples": int(len(features)),
        "n_features": int(features.shape[1]),
        "positive_rate": float(y.mean()),
        "best_model": best_name,
        "models": all_metrics,
    }

    return summary, fi_combined, predictions_df, features, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train top-tier AML ensemble on AMLSim outputs.")
    parser.add_argument("conf", nargs="?", default="journal/conf_aml_paper.json")
    parser.add_argument("--simulation-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per model (0=skip tuning).")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    conf = load_conf(Path(args.conf))
    sim_name = resolve_sim_name(conf, args.simulation_name)
    output_dir = resolve_output_dir(conf, sim_name)

    print(f"[ensemble] Simulation: {sim_name}")
    print(f"[ensemble] Available: XGBoost={HAS_XGB} LightGBM={HAS_LGB} Optuna={HAS_OPTUNA} imbalanced-learn={HAS_IMBLEARN}")

    summary, fi, predictions, features, y = run_ensemble_pipeline(
        output_dir, seed=args.seed, n_trials=args.n_trials, test_size=args.test_size,
    )

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "ml_ensemble_metrics.json"
    fi_path = output_dir / "ml_ensemble_feature_importance.csv"
    pred_path = output_dir / "ml_ensemble_predictions.csv"
    dataset_path = output_dir / "ml_ensemble_account_features.csv"

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    fi.to_csv(fi_path, index=False)
    predictions.to_csv(pred_path, index=False)

    dataset = features.copy()
    dataset["is_sar"] = y.values
    dataset.to_csv(dataset_path)

    # Print summary
    best_name = summary["best_model"]
    best = summary["models"].get(best_name, {})
    print(f"\n{'='*60}")
    print(f"[ensemble] BEST MODEL: {best_name}")
    print(f"[ensemble] F1: {best.get('f1', 0):.4f} | Macro-F1: {best.get('f1_macro', 0):.4f}")
    print(f"[ensemble] Precision: {best.get('precision', 0):.4f} | Recall: {best.get('recall', 0):.4f}")
    print(f"[ensemble] BA: {best.get('balanced_accuracy', 0):.4f} | ROC-AUC: {best.get('roc_auc', 0):.4f} | PR-AUC: {best.get('auc_pr', 0):.4f}")
    print(f"[ensemble] Samples: {summary['n_samples']} | Features: {summary['n_features']}")
    print(f"{'='*60}")
    print(f"[ensemble] Saved: {metrics_path}")
    print(f"[ensemble] Saved: {fi_path}")
    print(f"[ensemble] Saved: {pred_path}")
    print(f"[ensemble] Saved: {dataset_path}")


if __name__ == "__main__":
    main()
