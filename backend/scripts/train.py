from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from loan_prediction.config import (  # noqa: E402
    ARTIFACT_DIR,
    DATA_PATH,
    DECISION_THRESHOLD,
    FAIRNESS_PATH,
    IMPORTANCE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_FEATURES,
)
from loan_prediction.modeling import (  # noqa: E402
    aggregate_feature_importance,
    build_pipeline,
    candidate_models,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def round_metrics(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, float)):
            result[key] = round(float(value), 6)
        elif isinstance(value, (np.integer, int)):
            result[key] = int(value)
        else:
            result[key] = value
    return result


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    if df.shape[0] < 500:
        raise ValueError("Dataset unexpectedly small; refusing to train.")
    required = set(RAW_FEATURES) | {"Loan_Status", "Gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    X = df[RAW_FEATURES].copy()
    y = (df["Loan_Status"] == "Y").astype(int)
    return df, X, y


def benchmark(X: pd.DataFrame, y: pd.Series):
    repeats = int(os.getenv("LOAN_PROJECT_CV_REPEATS", "5"))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "f1_approved": "f1",
        "rejection_recall": make_scorer(recall_score, pos_label=0),
        "approval_recall": make_scorer(recall_score, pos_label=1),
    }

    rows = []
    pipelines = {}
    for spec in candidate_models():
        pipeline = build_pipeline(spec.estimator)
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        row = {"model": spec.name}
        for metric in scoring:
            values = result[f"test_{metric}"]
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(round_metrics(row))
        pipelines[spec.name] = pipeline

    rows.sort(key=lambda r: (r["roc_auc"], r["balanced_accuracy"]), reverse=True)
    return rows, pipelines


def fairness_audit(champion, source_df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(
        champion, X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    slices = {}
    audit_frame = source_df[["Gender", "Education", "Property_Area"]].copy()
    audit_frame["actual"] = y.to_numpy()
    audit_frame["predicted"] = predictions
    audit_frame["probability"] = probabilities

    for attribute in ["Gender", "Education", "Property_Area"]:
        group_rows = []
        for value, group in audit_frame.groupby(attribute, dropna=False):
            if len(group) < 10:
                continue
            actual = group["actual"].to_numpy()
            predicted = group["predicted"].to_numpy()
            row = {
                "group": "Missing" if pd.isna(value) else str(value),
                "n": int(len(group)),
                "actual_approval_rate": float(actual.mean()),
                "predicted_approval_rate": float(predicted.mean()),
                "mean_predicted_probability": float(group["probability"].mean()),
                "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                "approval_recall": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
                "rejection_recall": float(recall_score(actual, predicted, pos_label=0, zero_division=0)),
            }
            group_rows.append(round_metrics(row))
        slices[attribute] = group_rows

    gender_rows = [
        row for row in slices.get("Gender", []) if row["group"] in {"Female", "Male"}
    ]
    if len(gender_rows) == 2:
        parity = [row["predicted_approval_rate"] for row in gender_rows]
        opportunity = [row["approval_recall"] for row in gender_rows]
        rejection_recall = [row["rejection_recall"] for row in gender_rows]
        gender_summary = {
            "groups_compared": [row["group"] for row in gender_rows],
            "demographic_parity_difference": round(max(parity) - min(parity), 6),
            "equal_opportunity_difference": round(max(opportunity) - min(opportunity), 6),
            "rejection_recall_difference": round(max(rejection_recall) - min(rejection_recall), 6),
        }
    else:
        gender_summary = {}

    return {
        "method": "5-fold out-of-fold predictions using the champion model",
        "gender_is_prediction_feature": False,
        "summary": gender_summary,
        "slices": slices,
        "caveat": (
            "This small public practice dataset is not representative of any real lending population. "
            "Slice metrics are an educational audit, not evidence of regulatory fairness or absence of bias."
        ),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    source_df, X, y = load_dataset()
    benchmarks, pipelines = benchmark(X, y)

    tree_candidates = {"Random Forest", "Gradient Boosting", "XGBoost"}
    champion_name = next(
        row["model"] for row in benchmarks if row["model"] in tree_candidates
    )
    challenger_name = "Logistic Regression"

    champion = pipelines[champion_name]
    challenger = pipelines[challenger_name]
    champion.fit(X, y)
    challenger.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probability = cross_val_predict(
        champion, X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    oof_prediction = (oof_probability >= DECISION_THRESHOLD).astype(int)
    holdout_like = {
        "method": "5-fold out-of-fold predictions",
        "accuracy": accuracy_score(y, oof_prediction),
        "balanced_accuracy": balanced_accuracy_score(y, oof_prediction),
        "roc_auc": roc_auc_score(y, oof_probability),
        "precision_approved": precision_score(y, oof_prediction, pos_label=1, zero_division=0),
        "recall_approved": recall_score(y, oof_prediction, pos_label=1, zero_division=0),
        "recall_rejected": recall_score(y, oof_prediction, pos_label=0, zero_division=0),
        "f1_approved": f1_score(y, oof_prediction, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y, oof_prediction, labels=[0, 1]).tolist(),
    }
    holdout_like = round_metrics(holdout_like)

    dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    model_version = datetime.now(timezone.utc).strftime("%Y.%m.%d") + ".1"
    metadata = {
        "project": "Loan Approval Project - Explainable Loan Approval Intelligence",
        "model_version": model_version,
        "champion_model": champion_name,
        "challenger_model": challenger_name,
        "decision_threshold": DECISION_THRESHOLD,
        "training_rows": int(len(source_df)),
        "positive_class": "Loan_Status=Y (approval)",
        "dataset_sha256": dataset_hash,
        "random_state": RANDOM_STATE,
        "sensitive_feature_policy": (
            "Gender is excluded from prediction features and retained only for post-hoc fairness auditing."
        ),
        "engineered_features": [
            "TotalIncome",
            "LogTotalIncome",
            "LogLoanAmount",
            "LoanToAnnualIncome",
            "MonthlyPrincipalProxy",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    fairness = fairness_audit(champion, source_df, X, y)
    feature_importance = aggregate_feature_importance(champion)

    metrics = {
        "selection_metric": "mean ROC-AUC across repeated stratified cross-validation",
        "cross_validation": f"5 folds x {int(os.getenv('LOAN_PROJECT_CV_REPEATS', '5'))} repeats",
        "benchmark": benchmarks,
        "out_of_fold_champion": holdout_like,
        "class_balance": {
            "approved": int(y.sum()),
            "rejected": int((1 - y).sum()),
            "approval_rate": round(float(y.mean()), 6),
        },
    }

    bundle = {
        "champion": champion,
        "challenger": challenger,
        "metadata": metadata,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    write_json(METRICS_PATH, metrics)
    write_json(METADATA_PATH, metadata)
    write_json(FAIRNESS_PATH, fairness)
    write_json(IMPORTANCE_PATH, feature_importance)

    print(json.dumps({
        "champion": champion_name,
        "challenger": challenger_name,
        "model_version": model_version,
        "top_benchmark": benchmarks[0],
        "oof": holdout_like,
        "artifact": str(MODEL_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
