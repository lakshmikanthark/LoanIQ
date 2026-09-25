# Data provenance

## v1 — original internship data

`raw/loan_applications.csv` is the 614-row dataset found in the original Technofly Solutions college-internship project. Its schema matches the commonly used Analytics Vidhya Loan Prediction practice dataset.

Important interpretation limits:

- target: historical `Loan_Status` approval (`Y` / `N`), not repayment/default;
- `LoanAmount` is represented in thousands in the source;
- the population is small and is not a production lending sample;
- `Gender` is excluded from v1 prediction in the rebuilt pipeline.

## v2 — expanded accuracy dataset

Enhanced Loan Approval Model expects `raw/loan_approval_4269.csv`, a 4,269-row public loan-approval dataset commonly distributed through Kaggle as **Loan Approval Prediction Dataset** (Archit Sharma). The project downloader uses public raw mirrors and validates the downloaded schema before training.

Expected columns:

```text
loan_id
no_of_dependents
education
self_employed
income_annum
loan_amount
loan_term
cibil_score
residential_assets_value
commercial_assets_value
luxury_assets_value
bank_asset_value
loan_status
```

The downloader intentionally does not merge this table row-wise with the 614-row dataset because the schemas and underlying populations differ. v1 and v2 are separate model generations.

Fetch + validate:

```bash
python backend/scripts/fetch_enhanced_data.py
```

The v2 source CSV and generated enhanced artifacts are excluded from the distributable repository by default so the source can be fetched directly and its current terms reviewed. The training metadata records a SHA-256 checksum of the file actually used.

### Data-quality handling

- `loan_id` is never a prediction feature.
- Feature-identical duplicates are removed even if their IDs differ.
- Known negative asset entries are treated as data-quality errors, clipped to zero and counted in the training report.
- Unknown target labels stop training.
- Missing predictors are handled inside the model pipeline, not before the split.

This is educational public data and must not be represented as production bank underwriting data.
