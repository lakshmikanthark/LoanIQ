# Data provenance and model-generation boundary

Loan Approval Project uses two different public-data schemas and therefore maintains two separate models.

## Generation 1

The archived internship project contains the 614-row Analytics Vidhya-style Loan Prediction dataset. It includes demographic, household-income, loan, credit-history and property-area fields.

## Generation 2

The v2 training script expects the public 4,269-row Loan Approval Prediction dataset associated with Archit Sharma's Kaggle dataset. It adds signals unavailable in v1, especially CIBIL score and asset values.

The downloader currently tries these public mirrors:

1. `https://raw.githubusercontent.com/fahadsultan/csc272/refs/heads/main/data/loan_approval_dataset.csv`
2. `https://raw.githubusercontent.com/AbhishekBiswas-github/AI-Engineer-Projects/refs/heads/main/Machine-Learning/Loan-Approval-Prediction/loan_approval_dataset.csv`

Before the file is accepted, Loan Approval Project verifies:

- at least 4,000 rows;
- the complete expected predictor schema;
- only `Approved` / `Rejected` target values;
- complete target mapping.

Training then stores the exact dataset SHA-256 in model metadata.

## Why the datasets are not concatenated

A row-wise union would create a misleading table because important features do not exist in v1 and the populations/data-generation processes are not documented as equivalent. Instead, v1 is the historical baseline and v2 is the expanded model generation.

## License / redistribution

The repository does not assert a license for the dataset itself. Review the source dataset's current terms before redistributing its CSV. The code, schema validation and downloader are independent of that decision.
