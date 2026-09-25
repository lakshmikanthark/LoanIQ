from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .enhanced_config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from .enhanced_features import add_enhanced_features


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator: Any, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("feature_engineering", FunctionTransformer(add_enhanced_features, validate=False)),
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def baseline_candidates(quality: str = "max") -> list[ModelSpec]:
    tree_estimators = {"fast": 120, "balanced": 350, "max": 700}.get(quality, 700)
    boost_estimators = {"fast": 120, "balanced": 300, "max": 500}.get(quality, 500)
    models: list[ModelSpec] = [
        ModelSpec(
            "Logistic Regression",
            build_pipeline(LogisticRegression(max_iter=4000, C=1.0, random_state=RANDOM_STATE), True),
        ),
        ModelSpec(
            "Random Forest",
            build_pipeline(
                RandomForestClassifier(
                    n_estimators=tree_estimators,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ),
        ModelSpec(
            "Extra Trees",
            build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=tree_estimators,
                    min_samples_leaf=1,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ),
    ]
    try:
        from xgboost import XGBClassifier
        models.append(
            ModelSpec(
                "XGBoost",
                build_pipeline(
                    XGBClassifier(
                        n_estimators=boost_estimators,
                        max_depth=3,
                        learning_rate=0.04,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        reg_alpha=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=4,
                    )
                ),
            )
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models.append(
            ModelSpec(
                "LightGBM",
                build_pipeline(
                    LGBMClassifier(
                        n_estimators=boost_estimators,
                        learning_rate=0.04,
                        num_leaves=15,
                        max_depth=5,
                        subsample=0.9,
                        subsample_freq=1,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        verbosity=-1,
                        random_state=RANDOM_STATE,
                        n_jobs=4,
                    )
                ),
            )
        )
    except Exception:
        pass
    return models


def tuned_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "Extra Trees":
        return build_pipeline(ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params))
    if model_name == "Random Forest":
        return build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params))
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        return build_pipeline(XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4, **params))
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        return build_pipeline(LGBMClassifier(verbosity=-1, random_state=RANDOM_STATE, n_jobs=4, subsample_freq=1, **params))
    if model_name == "Logistic Regression":
        return build_pipeline(LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, **params), True)
    raise ValueError(f"Unknown model {model_name}")


def aggregate_transformed_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        return []
    pairs = sorted(zip(names, values, strict=True), key=lambda p: p[1], reverse=True)
    total = float(values.sum()) or 1.0
    return [{"feature": str(n), "importance": float(v), "share": float(v/total)} for n, v in pairs]
