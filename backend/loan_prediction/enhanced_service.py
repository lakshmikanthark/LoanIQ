from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .enhanced_config import IMPORTANCE_PATH, METADATA_PATH, METRICS_PATH, MODEL_PATH, RAW_FEATURES
from .schemas import EnhancedApplicantInput

DISCLAIMER = (
    "Educational project model only. This public practice dataset is not validated for "
    "real lending, underwriting, pricing, or any consequential financial decision."
)


def enhanced_ready() -> bool:
    return MODEL_PATH.exists() and METADATA_PATH.exists() and METRICS_PATH.exists()


@lru_cache(maxsize=1)
def get_enhanced_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError("Enhanced model is not trained. Run `python backend/scripts/bootstrap_enhanced.py`.")
    return joblib.load(MODEL_PATH)


def _read(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def enhanced_model_info() -> dict[str, Any]:
    if not enhanced_ready():
        return {
            "ready": False,
            "setup_command": "python backend/scripts/bootstrap_enhanced.py --quality max",
            "dataset_rows": 4269,
            "disclaimer": DISCLAIMER,
        }
    return {
        "ready": True,
        "metadata": _read(METADATA_PATH),
        "metrics": _read(METRICS_PATH),
        "feature_importance": _read(IMPORTANCE_PATH) if IMPORTANCE_PATH.exists() else [],
        "disclaimer": DISCLAIMER,
    }


def applicant_to_frame(applicant: EnhancedApplicantInput) -> pd.DataFrame:
    row = {
        "no_of_dependents": applicant.no_of_dependents,
        "education": applicant.education,
        "self_employed": applicant.self_employed,
        "income_annum": applicant.income_annum,
        "loan_amount": applicant.loan_amount,
        "loan_term": applicant.loan_term,
        "cibil_score": applicant.cibil_score,
        "residential_assets_value": applicant.residential_assets_value,
        "commercial_assets_value": applicant.commercial_assets_value,
        "luxury_assets_value": applicant.luxury_assets_value,
        "bank_asset_value": applicant.bank_asset_value,
    }
    return pd.DataFrame([row], columns=RAW_FEATURES)


def _ensemble_probability(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    prob = 0.0
    for model, weight, name in zip(bundle["models"], bundle["weights"], bundle["model_names"], strict=True):
        p = float(model.predict_proba(frame)[0, 1])
        prob += float(weight) * p
        rows.append({"model": name, "weight": round(float(weight), 4), "approval_probability": round(p, 6)})
    return float(prob), rows


def _local_sensitivity(bundle: dict[str, Any], frame: pd.DataFrame, base_probability: float) -> list[dict[str, Any]]:
    baselines = bundle.get("feature_baselines", {})
    labels = {
        "cibil_score": "CIBIL score",
        "loan_amount": "Loan amount",
        "income_annum": "Annual income",
        "loan_term": "Loan term",
        "residential_assets_value": "Residential assets",
        "commercial_assets_value": "Commercial assets",
        "luxury_assets_value": "Luxury assets",
        "bank_asset_value": "Bank assets",
        "no_of_dependents": "Dependents",
        "education": "Education",
        "self_employed": "Self-employment",
    }
    rows = []
    for feature, baseline in baselines.items():
        if feature not in frame.columns:
            continue
        modified = frame.copy()
        modified.loc[0, feature] = baseline
        p, _ = _ensemble_probability(bundle, modified)
        delta = base_probability - p
        rows.append({
            "feature": feature,
            "label": labels.get(feature, feature),
            "contribution": round(float(delta), 6),
            "direction": "supports approval" if delta >= 0 else "reduces approval",
        })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows[:6]


def predict_enhanced(applicant: EnhancedApplicantInput) -> dict[str, Any]:
    bundle = get_enhanced_bundle()
    frame = applicant_to_frame(applicant)
    probability, members = _ensemble_probability(bundle, frame)
    threshold = float(bundle["threshold"])
    approved = probability >= threshold
    confidence = probability if approved else 1 - probability
    return {
        "likely_approved": bool(approved),
        "approval_probability": round(probability, 6),
        "rejection_probability": round(1 - probability, 6),
        "confidence": round(confidence, 6),
        "threshold": threshold,
        "champion_model": "Optimized weighted ensemble",
        "ensemble_members": members,
        "top_factors": _local_sensitivity(bundle, frame, probability),
        "model_version": bundle["metadata"]["model_version"],
        "training_rows": bundle["metadata"]["training_rows"],
        "disclaimer": DISCLAIMER,
    }
