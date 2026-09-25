from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from loan_prediction.enhanced_config import (  # noqa: E402
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    ENHANCED_DATA_PATH,
    IMPORTANCE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_FEATURES,
    TEST_SIZE,
)
from loan_prediction.enhanced_features import clean_enhanced_dataset  # noqa: E402
from loan_prediction.enhanced_modeling import baseline_candidates, tuned_pipeline  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def metric_row(y_true, probability, threshold: float) -> dict[str, Any]:
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, prediction)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, prediction)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 6),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 6),
        "precision_approved": round(float(precision_score(y_true, prediction, pos_label=1, zero_division=0)), 6),
        "recall_approved": round(float(recall_score(y_true, prediction, pos_label=1, zero_division=0)), 6),
        "recall_rejected": round(float(recall_score(y_true, prediction, pos_label=0, zero_division=0)), 6),
        "f1_approved": round(float(f1_score(y_true, prediction, pos_label=1, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def load_data():
    if not ENHANCED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ENHANCED_DATA_PATH}. Run `python backend/scripts/fetch_enhanced_data.py` first."
        )
    raw = pd.read_csv(ENHANCED_DATA_PATH)
    df, data_report = clean_enhanced_dataset(raw)
    required = set(RAW_FEATURES) | {"target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    X = df[RAW_FEATURES].copy()
    y = df["target"].astype(int)
    return df, X, y, data_report


def baseline_benchmark(X_train, y_train, cv, quality: str):
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
    }
    rows = []
    models = {}
    for spec in baseline_candidates(quality):
        result = cross_validate(spec.estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": spec.name}
        for key in scoring:
            values = result[f"test_{key}"]
            row[key] = round(float(np.mean(values)), 6)
            row[f"{key}_std"] = round(float(np.std(values)), 6)
        rows.append(row)
        models[spec.name] = spec.estimator
    rows.sort(key=lambda r: (r["accuracy"], r["roc_auc"]), reverse=True)
    return rows, models


def tune_model(name: str, X, y, cv, trials: int, quality: str):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if quality == "fast":
        tree_low, tree_high, tree_step = 120, 320, 100
        boost_low, boost_high, boost_step = 100, 300, 100
    elif quality == "balanced":
        tree_low, tree_high, tree_step = 300, 900, 150
        boost_low, boost_high, boost_step = 250, 700, 100
    else:
        tree_low, tree_high, tree_step = 500, 1400, 150
        boost_low, boost_high, boost_step = 300, 1000, 100

    def objective(trial):
        if name == "Extra Trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", tree_low, tree_high, step=tree_step),
                "max_depth": trial.suggest_categorical("max_depth", [None, 6, 8, 10, 14, 18]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.7, 0.9, 1.0]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            }
        elif name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", tree_low, tree_high, step=tree_step),
                "max_depth": trial.suggest_categorical("max_depth", [None, 5, 7, 9, 12, 16]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.6, 0.8, 1.0]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            }
        elif name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", boost_low, boost_high, step=boost_step),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 8),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 2),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 8.0, log=True),
            }
        elif name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", boost_low, boost_high, step=boost_step),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 48),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 8.0, log=True),
            }
        else:
            return None
        model = tuned_pipeline(name, params)
        result = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "roc_auc"], n_jobs=1)
        accuracy = float(np.mean(result["test_accuracy"]))
        auc = float(np.mean(result["test_roc_auc"]))
        trial.set_user_attr("roc_auc", auc)
        return accuracy

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params, float(study.best_value), float(study.best_trial.user_attrs.get("roc_auc", 0.0))


def optimize_blend(oof_predictions: dict[str, np.ndarray], y) -> tuple[list[str], list[float], float, dict]:
    # Accuracy is the explicit optimization goal. Rank candidates using only OOF
    # development predictions and keep four diverse finalists for blending.
    single_scores = {}
    for name, probability in oof_predictions.items():
        single_scores[name] = metric_row(y, probability, 0.5)
    ranked = sorted(
        oof_predictions,
        key=lambda name: (
            single_scores[name]["accuracy"],
            single_scores[name]["balanced_accuracy"],
            single_scores[name]["roc_auc"],
        ),
        reverse=True,
    )[:4]
    arrays = [np.asarray(oof_predictions[name], dtype=float) for name in ranked]

    # Enumerate all 0.1-spaced simplex weights. Zero weights are valid, so the
    # search can fall back to one strong model when an ensemble does not help.
    units = 10
    integer_weights: list[tuple[int, ...]] = []

    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            integer_weights.append(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            build(prefix + [value], remaining - value, slots - 1)

    build([], units, len(ranked))
    weight_sets = [tuple(value / units for value in values) for values in integer_weights]

    y_array = np.asarray(y, dtype=int)
    best = None
    for weights in weight_sets:
        probability = sum(w * arr for w, arr in zip(weights, arrays, strict=True))
        auc = float(roc_auc_score(y_array, probability))
        for threshold in np.arange(0.30, 0.701, 0.01):
            prediction = (probability >= threshold).astype(int)
            accuracy = float(np.mean(prediction == y_array))
            positive = y_array == 1
            negative = ~positive
            tpr = float(np.mean(prediction[positive] == 1)) if positive.any() else 0.0
            tnr = float(np.mean(prediction[negative] == 0)) if negative.any() else 0.0
            balanced = (tpr + tnr) / 2.0
            key = (accuracy, balanced, auc, -abs(float(threshold) - 0.5))
            if best is None or key > best[0]:
                best = (key, list(weights), float(threshold), probability)

    assert best is not None
    final_metrics = metric_row(y_array, best[3], best[2])
    return ranked, best[1], best[2], final_metrics


def feature_baselines(X: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for col in X.columns:
        if col in CATEGORICAL_FEATURES:
            mode = X[col].dropna().astype(str).mode()
            result[col] = mode.iloc[0] if not mode.empty else "Unknown"
        else:
            values = pd.to_numeric(X[col], errors="coerce")
            median = values.median()
            result[col] = 0.0 if pd.isna(median) else float(median)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=["fast", "balanced", "max"], default="balanced")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    trials = args.trials or {"fast": 3, "balanced": 12, "max": 50}[args.quality]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    source, X, y, data_report = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv_folds = 3 if args.quality == "fast" else 5
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    baseline_rows, baseline_models = baseline_benchmark(X_train, y_train, cv, args.quality)
    tune_names = [n for n in ["Extra Trees", "Random Forest", "XGBoost", "LightGBM"] if n in baseline_models]
    tuned = {}
    tuning_rows = []
    for name in tune_names:
        params, score, auc = tune_model(name, X_train, y_train, cv, trials, args.quality)
        tuned[name] = tuned_pipeline(name, params)
        tuning_rows.append({"model": name, "cv_accuracy": round(score, 6), "cv_roc_auc": round(auc, 6), "params": params})

    candidate_models = {**baseline_models, **tuned}
    # If a tuned variant exists, it intentionally replaces its baseline counterpart.
    oof = {}
    cv_rows = []
    for name, model in candidate_models.items():
        prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        oof[name] = prob
        row = metric_row(y_train, prob, 0.5)
        row["model"] = name
        cv_rows.append(row)
    cv_rows.sort(key=lambda r: (r["accuracy"], r["roc_auc"]), reverse=True)

    member_names, weights, threshold, blend_cv_metrics = optimize_blend(oof, y_train)
    members = []
    for name in member_names:
        fitted = clone(candidate_models[name])
        fitted.fit(X_train, y_train)
        members.append(fitted)

    test_member_probabilities = [m.predict_proba(X_test)[:, 1] for m in members]
    test_probability = sum(w * p for w, p in zip(weights, test_member_probabilities, strict=True))
    test_metrics = metric_row(y_test, test_probability, threshold)
    # Post-selection diagnostics only: these member results are reported for transparency
    # and are never used to change the selected ensemble or threshold.
    test_member_metrics = []
    for name, member_probability in zip(member_names, test_member_probabilities, strict=True):
        row = metric_row(y_test, member_probability, 0.5)
        row["model"] = name
        test_member_metrics.append(row)

    # Raw-feature permutation importance for the final ensemble. This is a
    # post-evaluation diagnostic only and never feeds back into model selection.
    # We compute this manually because the final predictor is a weighted ensemble,
    # not a single sklearn estimator.
    rng = np.random.default_rng(RANDOM_STATE)
    baseline_accuracy = accuracy_score(y_test, (test_probability >= threshold).astype(int))
    importance = []
    importance_repeats = {"fast": 3, "balanced": 8, "max": 12}[args.quality]
    for feature in X_test.columns:
        drops = []
        for _ in range(importance_repeats):
            permuted = X_test.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            p = sum(
                w * model.predict_proba(permuted)[:, 1]
                for w, model in zip(weights, members, strict=True)
            )
            acc = accuracy_score(y_test, (p >= threshold).astype(int))
            drops.append(float(baseline_accuracy - acc))
        importance.append({
            "feature": feature,
            "importance": round(float(np.mean(drops)), 6),
            "std": round(float(np.std(drops)), 6),
        })
    importance.sort(key=lambda row: row["importance"], reverse=True)

    now = datetime.now(timezone.utc)
    dataset_hash = hashlib.sha256(ENHANCED_DATA_PATH.read_bytes()).hexdigest()
    metadata = {
        "project": "Enhanced Loan Approval Model - High-Capacity Explainable Loan Approval Intelligence",
        "model_version": now.strftime("%Y.%m.%d") + ".3",
        "training_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "source_rows": int(len(source)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "dataset_sha256": dataset_hash,
        "ensemble_members": member_names,
        "ensemble_weights": [round(float(w), 4) for w in weights],
        "decision_threshold": round(float(threshold), 4),
        "selection_protocol": f"Model/tuning/ensemble selection used only 80% training data with {cv_folds}-fold stratified CV; final 20% test set was untouched until one final evaluation.",
        "target": "Approved=1, Rejected=0",
        "dataset_source": "Kaggle Loan Approval Prediction Dataset by Archit Sharma (public mirror used by fetch script)",
        "generated_at_utc": now.isoformat(),
        "data_quality_report": data_report,
    }
    metrics = {
        "primary_metric": "accuracy",
        "baseline_cross_validation": baseline_rows,
        "tuning": tuning_rows,
        "training_oof_candidates": cv_rows,
        "training_oof_ensemble": blend_cv_metrics,
        "untouched_test": test_metrics,
        "untouched_test_members_diagnostic_only": test_member_metrics,
        "class_balance": {
            "approved": int(y.sum()),
            "rejected": int((1-y).sum()),
            "approval_rate": round(float(y.mean()), 6),
        },
        "quality_mode": args.quality,
        "optuna_trials_per_model": trials,
        "cv_folds": cv_folds,
    }
    bundle = {
        "models": members,
        "model_names": member_names,
        "weights": weights,
        "threshold": threshold,
        "metadata": metadata,
        "feature_baselines": feature_baselines(X_train),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    write_json(METADATA_PATH, metadata)
    write_json(METRICS_PATH, metrics)
    write_json(IMPORTANCE_PATH, importance)
    print(json.dumps({"metadata": metadata, "untouched_test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
