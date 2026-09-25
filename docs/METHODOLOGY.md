# Methodology

## Problem framing

Both Loan Approval Project generations predict a historical **approval/rejection label**. They do not estimate probability of default, loss given default, affordability or repayment behavior.

## v1 methodology

The 614-row internship baseline was rebuilt with imputation/encoding inside scikit-learn pipelines, class-aware evaluation and Gender excluded from the prediction vector. Its purpose is provenance and a fair comparison against the original notebook implementation.

## v2 methodology

### Data boundary

The expanded 4,269-row schema is treated as a separate model generation. Its rows are never concatenated with the original 614-row table.

### Quality controls before splitting

- normalize field names and categorical whitespace;
- convert declared numeric fields explicitly;
- reject unknown target labels;
- clip negative asset values to zero while counting corrections;
- remove feature/target-identical duplicates while ignoring `loan_id`.

### Fixed holdout

A seeded, stratified 80/20 split is created before any model selection. The 20% holdout remains outside hyperparameter tuning, OOF prediction, blend selection and threshold selection.

### Preprocessing

All imputation and one-hot encoding is part of the fitted pipeline. Therefore validation folds cannot inform preprocessing statistics for their corresponding training fold.

### Feature engineering

Every v2 derived feature is computable from request-time fields:

- total assets;
- loan / annual income;
- assets / loan;
- bank assets / annual income;
- loan / term year;
- total assets − loan proxy;
- CIBIL × log asset coverage interaction;
- CIBIL band.

### Model capacity

The benchmark includes Logistic Regression, Random Forest, Extra Trees, XGBoost and LightGBM. Four nonlinear families receive Optuna tuning in max mode.

### Ensemble selection

Each candidate generates OOF development probabilities. Candidates are ranked by OOF accuracy with balanced accuracy and ROC-AUC as tie-breakers. The best four enter a 0.1-spaced simplex weight search, including zero weights. Decision thresholds from 0.30 to 0.70 are evaluated on development OOF predictions.

The selection key is:

1. accuracy;
2. balanced accuracy;
3. ROC-AUC;
4. closeness to a neutral 0.50 threshold.

Only after this is frozen are ensemble members fitted on all development rows and evaluated on the locked holdout.

### Reporting

The project records accuracy, balanced accuracy, ROC-AUC, Brier score, class recall, precision, F1 and the confusion matrix. Member holdout results are reported only as post-selection diagnostics and never used to change the champion.

## Explainability

Permutation importance on the holdout is a post-evaluation diagnostic. Per-request local factors are probability sensitivities created by replacing one field at a time with a training baseline. Neither is a causal explanation.
