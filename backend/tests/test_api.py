from fastapi.testclient import TestClient

from loan_prediction.api import app

client = TestClient(app)


def payload():
    return {
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 6000,
        "coapplicant_income": 1500,
        "loan_amount": 140,
        "loan_amount_term": 360,
        "credit_history": 1,
        "property_area": "Semiurban",
    }


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_contract():
    response = client.post("/api/v1/predict", json=payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["approval_probability"] <= 1
    assert isinstance(body["likely_approved"], bool)
    assert len(body["top_factors"]) > 0
    assert "Educational" in body["disclaimer"]


def test_gender_is_not_accepted_as_prediction_input():
    p = payload()
    p["gender"] = "Male"
    response = client.post("/api/v1/predict", json=p)
    assert response.status_code == 422


def test_model_info_contract():
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["champion_model"]
    assert len(body["metrics"]["benchmark"]) >= 4
    assert body["fairness"]["gender_is_prediction_feature"] is False


def test_batch_prediction_contract():
    response = client.post("/api/v1/predict/batch", json={"applicants": [payload(), payload()]})
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 2


def test_dashboard_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Loan Approval Prediction & Business Analysis | Data Science Internship Project" in response.text


def test_static_path_traversal_does_not_expose_repository_files():
    response = client.get("/%2E%2E/README.md")
    assert "Loan Approval Prediction & Business Analysis | Data Science Internship Project" in response.text
    assert "# Loan Approval Project" not in response.text
