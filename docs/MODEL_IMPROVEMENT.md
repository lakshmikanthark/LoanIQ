# Accuracy upgrade — methodology

## Goal

Improve the original internship model's predictive performance without creating a misleading score through leakage, test-set tuning, duplicated records or synthetic copies of the same small dataset.

## Why the 614-row ceiling matters

The original dataset has limited sample size and limited financial signal. Repeatedly tuning more complex algorithms on the same 614 records increases the risk of overfitting. Loan Approval Project therefore keeps that model as v1 and trains v2 on a larger 4,269-row loan-approval dataset with CIBIL and asset fields.

## Evaluation boundary

1. Validate schema and target.
2. Remove feature-identical duplicates while ignoring the ID column.
3. Stratify once into 80% development / 20% locked test.
4. Do all preprocessing, model selection, hyperparameter search, ensemble selection and threshold selection using development data only.
5. Evaluate the final frozen ensemble once on the test split.
6. Use the test set only for final reporting and post-evaluation diagnostics; never change the model after observing it.

This makes the test accuracy a genuine hold-out result for this dataset split.

## Candidate models

- Logistic Regression — linear reference model.
- Random Forest — bagged nonlinear trees.
- Extra Trees — high-variance randomized ensemble.
- XGBoost — regularized gradient boosting.
- LightGBM — histogram-based gradient boosting.

The production path intentionally avoids fragile model/library combinations that fail the project's scikit-learn compatibility tests.

## Hyperparameter search

Max mode runs 50 seeded Optuna trials for each tuned tree/boosting family. Search spaces include estimator count, depth/leaf complexity, learning rate, row/column subsampling and regularization as appropriate.

The objective prioritizes cross-validated accuracy because improving accuracy is the explicit project goal. ROC-AUC is retained as a secondary diagnostic.

## Ensemble

Every candidate produces out-of-fold development probabilities. The strongest probability models are blended, and both blend weights and the decision threshold are chosen only from those OOF predictions. A zero weight is allowed, so the search can effectively fall back to a single model when blending does not help.

## Metrics

Final reporting includes:

- accuracy;
- balanced accuracy;
- ROC-AUC;
- Brier score;
- precision / recall / F1;
- rejected-class recall;
- confusion matrix;
- raw-feature permutation importance.

A high accuracy on this dataset should be described as **dataset-specific held-out classification performance**, not real-world credit-risk performance.

## Performance claims

`backend/artifacts/enhanced/metrics.json` is the sole source of truth after a real v2 training run. Synthetic smoke-test metrics, third-party blog scores and repository claims are never used as resume evidence.
