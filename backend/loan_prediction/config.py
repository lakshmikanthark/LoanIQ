from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_applications.csv"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
FAIRNESS_PATH = ARTIFACT_DIR / "fairness_audit.json"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.json"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50

RAW_FEATURES = [
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "TotalIncome",
    "LogTotalIncome",
    "LogLoanAmount",
    "LoanToAnnualIncome",
    "MonthlyPrincipalProxy",
]

CATEGORICAL_FEATURES = [
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]
