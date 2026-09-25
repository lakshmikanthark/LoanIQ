import pandas as pd

from loan_prediction.features import add_engineered_features


def test_feature_engineering_is_finite_for_normal_input():
    frame = pd.DataFrame([
        {
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 1500,
            "LoanAmount": 140,
            "Loan_Amount_Term": 360,
        }
    ])
    result = add_engineered_features(frame)
    assert result.loc[0, "TotalIncome"] == 6500
    assert result.loc[0, "LoanToAnnualIncome"] > 0
    assert result.loc[0, "MonthlyPrincipalProxy"] > 0
