from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ENHANCED_DATA_PATH = DATA_DIR / "loan_approval_4269.csv"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "artifacts" / "enhanced"
MODEL_PATH = ARTIFACT_DIR / "model_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.json"

RANDOM_STATE = 2026
TEST_SIZE = 0.20

RAW_FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CATEGORICAL_FEATURES = ["education", "self_employed", "cibil_band"]
NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "loan_to_income",
    "asset_to_loan",
    "bank_assets_to_income",
    "loan_per_term_year",
    "net_worth_proxy",
    "cibil_x_asset_coverage",
]
