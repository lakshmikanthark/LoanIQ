from fastapi.testclient import TestClient

from loan_prediction.api import app
from loan_prediction.business_analysis import optimize_policy, build_analysis_frame, run_business_analysis

client = TestClient(app)


def test_analysis_simulation_is_deterministic_and_complete():
    df = build_analysis_frame()
    assert len(df) == 614
    required = {
        "approval_probability",
        "historical_approved",
        "channel",
        "document_completeness",
        "processing_hours",
        "baseline_route",
        "priority_score",
    }
    assert required.issubset(df.columns)
    assert df["approval_probability"].between(0, 1).all()
    assert df["document_completeness"].between(0, 1).all()


def test_run_business_analysis_returns_business_metrics():
    result = run_business_analysis()
    assert result["simulation"] is True
    assert result["policy"]["applications"] == 614
    assert 0 <= result["policy"]["auto_decision_rate"] <= 1
    assert 0 <= result["policy"]["auto_historical_agreement"] <= 1
    assert result["policy"]["manual_review"] >= 0
    assert len(result["trends"]) > 20
    assert result["queue"]
    assert result["analysis_summary"]


def test_more_conservative_thresholds_increase_manual_reviews():
    standard = run_business_analysis(0.75, 0.25)["policy"]
    conservative = run_business_analysis(0.90, 0.10)["policy"]
    assert conservative["manual_review_rate"] >= standard["manual_review_rate"]
    assert conservative["auto_decision_rate"] <= standard["auto_decision_rate"]


def test_volume_stress_increases_capacity_pressure():
    base = run_business_analysis(volume_multiplier=1.0)["policy"]
    stress = run_business_analysis(volume_multiplier=2.0)["policy"]
    assert stress["capacity_utilization"] >= base["capacity_utilization"]
    assert stress["estimated_review_hours"] >= base["estimated_review_hours"]


def test_optimizer_respects_requested_guardrail_when_feasible():
    result = optimize_policy(build_analysis_frame(), minimum_auto_agreement=0.85)
    assert "policy" in result
    if result["feasible"]:
        assert result["policy"]["auto_historical_agreement"] >= 0.85


def test_analysis_api_contract():
    status = client.get("/api/analysis/status")
    assert status.status_code == 200
    assert status.json()["ready"] is True

    response = client.post(
        "/api/analysis/run_business_analysis",
        json={
            "approve_threshold": 0.75,
            "reject_threshold": 0.25,
            "analysts": 8,
            "productive_hours": 6,
            "volume_multiplier": 1,
            "segment_dimension": "channel",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_summary"]["applications"] == 614
    assert body["segments"][0]["applications"] > 0


def test_case_endpoint_and_missing_case():
    first_id = build_analysis_frame().iloc[0]["application_id"]
    response = client.get(f"/api/analysis/case/{first_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["application_id"] == first_id
    assert "audit" in body
    assert client.get("/api/analysis/case/does-not-exist").status_code == 404


def test_invalid_threshold_order_rejected():
    response = client.post(
        "/api/analysis/run_business_analysis",
        json={
            "approve_threshold": 0.70,
            "reject_threshold": 0.72,
            "analysts": 8,
            "productive_hours": 6,
            "volume_multiplier": 1,
            "segment_dimension": "channel",
        },
    )
    assert response.status_code == 422
