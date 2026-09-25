from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApplicantInput(BaseModel):
    """Legacy 614-row internship model input."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    married: Literal["Yes", "No"] = "Yes"
    dependents: Literal["0", "1", "2", "3+"] = "0"
    education: Literal["Graduate", "Not Graduate"] = "Graduate"
    self_employed: Literal["Yes", "No"] = "No"
    applicant_income: float = Field(ge=0, le=1_000_000, examples=[5000])
    coapplicant_income: float = Field(default=0, ge=0, le=1_000_000)
    loan_amount: float = Field(gt=0, le=10_000, description="Loan amount in thousands")
    loan_amount_term: float = Field(default=360, gt=0, le=600)
    credit_history: Literal[0, 1] = 1
    property_area: Literal["Rural", "Semiurban", "Urban"] = "Urban"

    @field_validator("applicant_income", "coapplicant_income", "loan_amount", "loan_amount_term")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Value must be finite")
        return value


class EnhancedApplicantInput(BaseModel):
    """Enhanced Loan Approval Model input for the 4,269-row CIBIL/assets model."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    no_of_dependents: int = Field(default=0, ge=0, le=10)
    education: Literal["Graduate", "Not Graduate"] = "Graduate"
    self_employed: Literal["Yes", "No"] = "No"
    income_annum: float = Field(gt=0, le=100_000_000, description="Annual income")
    loan_amount: float = Field(gt=0, le=200_000_000)
    loan_term: float = Field(gt=0, le=40, description="Loan term in years")
    cibil_score: int = Field(ge=300, le=900)
    residential_assets_value: float = Field(default=0, ge=0, le=500_000_000)
    commercial_assets_value: float = Field(default=0, ge=0, le=500_000_000)
    luxury_assets_value: float = Field(default=0, ge=0, le=500_000_000)
    bank_asset_value: float = Field(default=0, ge=0, le=500_000_000)

    @field_validator(
        "income_annum", "loan_amount", "loan_term", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
    )
    @classmethod
    def finite_number(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Value must be finite")
        return value


class PredictionFactor(BaseModel):
    feature: str
    label: str
    contribution: float
    direction: str


class ChallengerResult(BaseModel):
    model: str
    approval_probability: float
    likely_approved: bool


class PredictionResponse(BaseModel):
    likely_approved: bool
    approval_probability: float
    rejection_probability: float
    confidence: float
    threshold: float
    champion_model: str
    challenger: ChallengerResult
    models_agree: bool
    top_factors: list[PredictionFactor]
    model_version: str
    disclaimer: str


class EnhancedMemberResult(BaseModel):
    model: str
    weight: float
    approval_probability: float


class EnhancedPredictionResponse(BaseModel):
    likely_approved: bool
    approval_probability: float
    rejection_probability: float
    confidence: float
    threshold: float
    champion_model: str
    ensemble_members: list[EnhancedMemberResult]
    top_factors: list[PredictionFactor]
    model_version: str
    training_rows: int
    disclaimer: str


class BatchPredictionRequest(BaseModel):
    applicants: list[ApplicantInput] = Field(min_length=1, max_length=100)


class EnhancedBatchPredictionRequest(BaseModel):
    applicants: list[EnhancedApplicantInput] = Field(min_length=1, max_length=250)


class BusinessAnalysisScenario(BaseModel):
    """Policy and staffing scenario for the Business Analysis simulator."""
    model_config = ConfigDict(extra="forbid")

    approve_threshold: float = Field(default=0.75, gt=0.5, lt=1.0)
    reject_threshold: float = Field(default=0.25, gt=0.0, lt=0.5)
    analysts: int = Field(default=1, ge=1, le=500)
    productive_hours: float = Field(default=6.0, gt=0.5, le=12.0)
    volume_multiplier: float = Field(default=1.0, gt=0.1, le=10.0)
    segment_dimension: Literal["channel", "property_area", "education", "self_employed", "credit_history"] = "channel"

    @field_validator("reject_threshold")
    @classmethod
    def valid_threshold_order(cls, value: float, info):
        approve = info.data.get("approve_threshold")
        if approve is not None and value >= approve:
            raise ValueError("reject_threshold must be below approve_threshold")
        return value


class PolicyOptimizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_auto_agreement: float = Field(default=0.90, ge=0.50, le=0.999)
    analysts: int = Field(default=1, ge=1, le=500)
    productive_hours: float = Field(default=6.0, gt=0.5, le=12.0)
    volume_multiplier: float = Field(default=1.0, gt=0.1, le=10.0)
