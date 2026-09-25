# Model card — Loan Approval Project

## System overview

Loan Approval Project exposes two educational binary-classification model generations.

### v1 — audited internship baseline

- Source population: 614 public practice applications.
- Target: historical loan approval (`Y`/`N`).
- Purpose: preserve and improve the original Technofly Solutions internship project.
- Rebuilt champion: selected tree model with Logistic Regression challenger.
- Sensitive-feature policy: Gender is excluded from inference.

### v2 — high-capacity accuracy rebuild

- Source population: 4,269 public loan-approval applications after download/validation; exact retained row count is written into metadata after deduplication.
- Target: `Approved` / `Rejected` historical label.
- Predictors: dependents, education, self-employment, annual income, requested loan, term, CIBIL score and four asset-value groups.
- Candidate families: Logistic Regression, Random Forest, Extra Trees, XGBoost and LightGBM.
- Champion: optimized weighted probability ensemble selected from development-set out-of-fold predictions.
- Threshold: selected on development OOF predictions; exact value stored in the trained artifact.

## Intended use

- ML/data-science project demonstration;
- interview discussion of model selection and leakage prevention;
- reproducible local inference;
- teaching classification evaluation and model limitations.

## Prohibited / out-of-scope use

Do not use Loan Approval Project for actual:

- credit underwriting or loan eligibility;
- pricing or interest-rate decisions;
- credit limits;
- probability-of-default estimation;
- automated decisions affecting a person's access to financial services;
- fairness, legal or regulatory certification.

The datasets are public educational data and are not demonstrated to represent a current lending population.

## Evaluation protocol for v2

The source is split once into 80% development and 20% locked holdout with target stratification. Preprocessing, tuning, OOF prediction, blending and threshold selection occur only inside the development partition. The holdout is evaluated after the model is frozen.

Exact metrics belong in `backend/artifacts/enhanced/metrics.json` after training. The repository does not hard-code a third-party accuracy claim.

## Data quality

The v2 cleaner:

- ignores ID for modeling;
- detects feature-identical duplicates even when IDs differ;
- removes exact feature/target duplicates before splitting;
- clips negative asset values to zero and records the correction count;
- rejects unmappable target values;
- keeps missing-predictor imputation inside the model pipeline.

## Explainability

Global v2 importance is computed by permuting raw input columns after final evaluation. Local explanations replace one input at a time with a training-set baseline and report the resulting probability change.

These are **sensitivity diagnostics**, not causal explanations. A positive local contribution does not mean changing that real-world attribute would cause approval.

## Known limitations

- approval labels reflect whatever rule/process generated the public dataset, not repayment outcomes;
- no temporal or external-population validation is available;
- feature distributions may be synthetic or simplified relative to real lending systems;
- dataset-specific high accuracy can reflect a relatively easy classification rule and should not be generalized to banks or borrowers;
- public datasets can change at the source; the downloader therefore validates schema and training records the dataset checksum;
- the model is not calibrated or governed as a production financial model.
