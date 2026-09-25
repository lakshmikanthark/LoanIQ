import pytest
from pydantic import ValidationError

from loan_prediction.schemas import ApplicantInput


def valid_payload():
    return {
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 5000,
        "coapplicant_income": 0,
        "loan_amount": 140,
        "loan_amount_term": 360,
        "credit_history": 1,
        "property_area": "Urban",
    }


def test_valid_applicant():
    applicant = ApplicantInput(**valid_payload())
    assert applicant.loan_amount == 140


def test_rejects_negative_income():
    payload = valid_payload()
    payload["applicant_income"] = -1
    with pytest.raises(ValidationError):
        ApplicantInput(**payload)


def test_rejects_unknown_fields():
    payload = valid_payload()
    payload["gender"] = "Male"
    with pytest.raises(ValidationError):
        ApplicantInput(**payload)
