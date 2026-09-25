import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from loan_prediction.api import app
from loan_prediction.enhanced_features import add_enhanced_features, clean_enhanced_dataset
from loan_prediction.enhanced_service import enhanced_ready
from loan_prediction.schemas import EnhancedApplicantInput

client = TestClient(app)


def enhanced_payload():
    return {
        "no_of_dependents": 1,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 7_200_000,
        "loan_amount": 18_000_000,
        "loan_term": 12,
        "cibil_score": 760,
        "residential_assets_value": 9_000_000,
        "commercial_assets_value": 4_000_000,
        "luxury_assets_value": 7_000_000,
        "bank_asset_value": 3_000_000,
    }


def test_enhanced_schema_accepts_valid_payload():
    item = EnhancedApplicantInput(**enhanced_payload())
    assert item.cibil_score == 760


def test_enhanced_schema_rejects_invalid_cibil():
    payload = enhanced_payload()
    payload["cibil_score"] = 100
    with pytest.raises(ValidationError):
        EnhancedApplicantInput(**payload)


def test_enhanced_feature_engineering_is_finite():
    raw = pd.DataFrame([enhanced_payload()])
    engineered = add_enhanced_features(raw)
    assert engineered.loc[0, "total_assets"] == 23_000_000
    assert engineered.loc[0, "loan_to_income"] > 0
    assert engineered.loc[0, "cibil_band"] == "excellent"


def test_cleaner_removes_feature_duplicates_even_if_ids_differ():
    base = enhanced_payload() | {"loan_status": "Approved"}
    frame = pd.DataFrame([{"loan_id": 1, **base}, {"loan_id": 2, **base}])
    clean, report = clean_enhanced_dataset(frame)
    assert len(clean) == 1
    assert report["duplicate_feature_rows_removed"] == 1


def test_v2_status_contract():
    response = client.get("/api/v2/status")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["ready"], bool)
    assert "disclaimer" in body


def test_v2_prediction_or_clean_setup_failure():
    response = client.post("/api/v2/predict", json=enhanced_payload())
    if enhanced_ready():
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0 <= body["approval_probability"] <= 1
        assert body["champion_model"] == "Optimized weighted ensemble"
        assert len(body["ensemble_members"]) >= 1
    else:
        assert response.status_code == 503
        assert "not trained" in response.text.lower()
