from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from .features import add_engineered_features


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(add_engineered_features, validate=False),
            ),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def candidate_models() -> list[ModelSpec]:
    models = [
        ModelSpec(
            "Logistic Regression",
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "RBF SVM",
            SVC(
                probability=True,
                class_weight="balanced",
                C=1.5,
                gamma="scale",
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    # XGBoost is a training-time enhancement. Runtime inference does not depend on it.
    try:
        from xgboost import XGBClassifier

        models.append(
            ModelSpec(
                "XGBoost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=4,
                ),
            )
        )
    except Exception:
        pass
    return models


def _base_feature_name(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__")
    raw = transformed_name.removeprefix("cat__")
    for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if raw == feature or raw.startswith(feature + "_"):
            return feature
    return raw


def aggregate_feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importances = np.asarray(estimator.feature_importances_, dtype=float)
    grouped: dict[str, float] = {}
    for name, value in zip(names, importances, strict=True):
        base = _base_feature_name(name)
        grouped[base] = grouped.get(base, 0.0) + float(value)
    total = sum(grouped.values()) or 1.0
    result = [
        {"feature": key, "importance": value, "share": value / total}
        for key, value in grouped.items()
    ]
    return sorted(result, key=lambda row: row["importance"], reverse=True)


def logistic_local_explanation(
    pipeline: Pipeline, frame: pd.DataFrame, top_n: int = 6
) -> list[dict[str, float | str]]:
    engineered = pipeline.named_steps["feature_engineering"].transform(frame)
    preprocessor = pipeline.named_steps["preprocessor"]
    matrix = np.asarray(preprocessor.transform(engineered), dtype=float)
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    contributions = matrix[0] * np.asarray(model.coef_[0], dtype=float)

    grouped: dict[str, float] = {}
    for name, contribution in zip(names, contributions, strict=True):
        base = _base_feature_name(name)
        grouped[base] = grouped.get(base, 0.0) + float(contribution)

    human_names = {
        "ApplicantIncome": "Applicant income",
        "CoapplicantIncome": "Co-applicant income",
        "LoanAmount": "Loan amount",
        "Loan_Amount_Term": "Loan term",
        "TotalIncome": "Total household income",
        "LogTotalIncome": "Income scale",
        "LogLoanAmount": "Loan size scale",
        "LoanToAnnualIncome": "Loan-to-income ratio",
        "MonthlyPrincipalProxy": "Monthly principal proxy",
        "Married": "Marital status",
        "Dependents": "Dependents",
        "Education": "Education",
        "Self_Employed": "Employment type",
        "Credit_History": "Credit history",
        "Property_Area": "Property area",
    }
    rows = [
        {
            "feature": feature,
            "label": human_names.get(feature, feature),
            "contribution": value,
            "direction": "supports approval" if value >= 0 else "reduces approval",
        }
        for feature, value in grouped.items()
    ]
    rows.sort(key=lambda row: abs(float(row["contribution"])), reverse=True)
    return rows[:top_n]
