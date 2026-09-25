from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create domain-inspired, leakage-safe features.

    The original college notebook used only raw columns. This rebuilt pipeline adds
    ratios/log transforms that can be computed from information known at inference
    time. No target-derived feature is used.
    """
    x = frame.copy()

    applicant = pd.to_numeric(x["ApplicantIncome"], errors="coerce")
    coapplicant = pd.to_numeric(x["CoapplicantIncome"], errors="coerce")
    loan_amount = pd.to_numeric(x["LoanAmount"], errors="coerce")
    term = pd.to_numeric(x["Loan_Amount_Term"], errors="coerce")

    total_income = applicant.fillna(0) + coapplicant.fillna(0)
    x["TotalIncome"] = total_income
    x["LogTotalIncome"] = np.log1p(total_income.clip(lower=0))
    x["LogLoanAmount"] = np.log1p(loan_amount.clip(lower=0))

    annual_income = (total_income * 12).replace(0, np.nan)
    safe_term = term.replace(0, np.nan)
    x["LoanToAnnualIncome"] = (loan_amount * 1000 / annual_income).replace(
        [np.inf, -np.inf], np.nan
    )
    # This is a principal-per-month proxy, not a real EMI because interest rate is
    # absent from the public dataset.
    x["MonthlyPrincipalProxy"] = (loan_amount * 1000 / safe_term).replace(
        [np.inf, -np.inf], np.nan
    )
    return x
