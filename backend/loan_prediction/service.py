from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from .config import (
    DECISION_THRESHOLD,
    FAIRNESS_PATH,
    IMPORTANCE_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
)
from .modeling import logistic_local_explanation
from .schemas import ApplicantInput

DISCLAIMER = (
    "Educational project model only. It is not validated for real lending, "
    "credit underwriting, or any consequential financial decision."
)


def applicant_to_frame(applicant: ApplicantInput) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Married": applicant.married,
                "Dependents": applicant.dependents,
                "Education": applicant.education,
                "Self_Employed": applicant.self_employed,
                "ApplicantIncome": applicant.applicant_income,
                "CoapplicantIncome": applicant.coapplicant_income,
                "LoanAmount": applicant.loan_amount,
                "Loan_Amount_Term": applicant.loan_amount_term,
                "Credit_History": applicant.credit_history,
                "Property_Area": applicant.property_area,
            }
        ]
    )


@lru_cache(maxsize=1)
def get_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python backend/scripts/train.py`."
        )
    return joblib.load(MODEL_PATH)


def _read_json(path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_info() -> dict[str, Any]:
    return {
        "metadata": _read_json(METADATA_PATH),
        "metrics": _read_json(METRICS_PATH),
        "fairness": _read_json(FAIRNESS_PATH),
        "feature_importance": _read_json(IMPORTANCE_PATH),
        "disclaimer": DISCLAIMER,
    }


def predict(applicant: ApplicantInput) -> dict[str, Any]:
    bundle = get_bundle()
    champion = bundle["champion"]
    challenger = bundle["challenger"]
    metadata = bundle["metadata"]
    frame = applicant_to_frame(applicant)

    champion_probability = float(champion.predict_proba(frame)[0, 1])
    challenger_probability = float(challenger.predict_proba(frame)[0, 1])
    threshold = float(metadata.get("decision_threshold", DECISION_THRESHOLD))
    likely_approved = champion_probability >= threshold
    challenger_approved = challenger_probability >= threshold
    confidence = champion_probability if likely_approved else 1 - champion_probability

    return {
        "likely_approved": likely_approved,
        "approval_probability": round(champion_probability, 6),
        "rejection_probability": round(1 - champion_probability, 6),
        "confidence": round(confidence, 6),
        "threshold": threshold,
        "champion_model": metadata["champion_model"],
        "challenger": {
            "model": metadata["challenger_model"],
            "approval_probability": round(challenger_probability, 6),
            "likely_approved": challenger_approved,
        },
        "models_agree": likely_approved == challenger_approved,
        "top_factors": logistic_local_explanation(challenger, frame, top_n=6),
        "model_version": metadata["model_version"],
        "disclaimer": DISCLAIMER,
    }
