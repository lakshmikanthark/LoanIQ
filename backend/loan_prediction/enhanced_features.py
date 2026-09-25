from __future__ import annotations

import numpy as np
import pandas as pd

ASSET_COLUMNS = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def clean_enhanced_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize the public 4,269-row loan approval dataset.

    Negative asset values are treated as data-quality errors and clipped to zero.
    No target-derived fields are created. The ID column is never used for modeling.
    """
    df = frame.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ["education", "self_employed", "loan_status"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()

    numeric = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        *ASSET_COLUMNS,
    ]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    corrections = {}
    for col in ASSET_COLUMNS:
        if col in df:
            count = int((df[col] < 0).sum())
            corrections[col] = count
            df[col] = df[col].clip(lower=0)

    target_map = {"Approved": 1, "Rejected": 0}
    if "loan_status" in df:
        df["target"] = df["loan_status"].map(target_map)

    rows_before = int(len(df))
    duplicate_subset = [
        c for c in [
            "no_of_dependents", "education", "self_employed", "income_annum",
            "loan_amount", "loan_term", "cibil_score", *ASSET_COLUMNS, "loan_status"
        ] if c in df.columns
    ]
    duplicate_mask = (
        df.duplicated(subset=duplicate_subset, keep="first")
        if duplicate_subset else df.duplicated(keep="first")
    )
    duplicate_feature_rows = int(duplicate_mask.sum())
    if duplicate_feature_rows:
        df = df.loc[~duplicate_mask].reset_index(drop=True)

    report = {
        "rows_before_deduplication": rows_before,
        "rows": int(len(df)),
        "duplicate_feature_rows_removed": duplicate_feature_rows,
        "negative_asset_values_clipped": corrections,
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0},
    }
    return df, report


def add_enhanced_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    eps = 1.0
    income = pd.to_numeric(x["income_annum"], errors="coerce").clip(lower=0)
    loan = pd.to_numeric(x["loan_amount"], errors="coerce").clip(lower=0)
    term = pd.to_numeric(x["loan_term"], errors="coerce").clip(lower=1)
    cibil = pd.to_numeric(x["cibil_score"], errors="coerce")

    assets = []
    for col in ASSET_COLUMNS:
        series = pd.to_numeric(x[col], errors="coerce").clip(lower=0)
        x[col] = series
        assets.append(series.fillna(0))
    total_assets = sum(assets)

    x["total_assets"] = total_assets
    x["loan_to_income"] = loan / (income + eps)
    x["asset_to_loan"] = total_assets / (loan + eps)
    x["bank_assets_to_income"] = x["bank_asset_value"] / (income + eps)
    x["loan_per_term_year"] = loan / term
    x["net_worth_proxy"] = total_assets - loan
    x["cibil_x_asset_coverage"] = cibil * np.log1p(x["asset_to_loan"].clip(lower=0))

    x["cibil_band"] = pd.cut(
        cibil,
        bins=[-np.inf, 499, 649, 749, np.inf],
        labels=["weak", "fair", "good", "excellent"],
    ).astype("object")
    return x
