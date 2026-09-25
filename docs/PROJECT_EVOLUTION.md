# Original internship work vs later rebuild

## What the archived Technofly Solutions internship project supports

- Built a binary loan-approval classifier using a 614-row public practice dataset.
- Performed preprocessing and categorical encoding in Jupyter.
- Compared Decision Tree, Gaussian Naive Bayes, polynomial SVM, Logistic Regression, Random Forest and XGBoost.
- Logistic Regression produced about 83.96% accuracy on the notebook's particular 80/20 split.
- Serialized models with pickle.
- Connected the Logistic Regression artifact to a Tkinter desktop form.

Those are the internship-faithful claims.

## First independent rebuild (v1)

Added later:

- leakage-safer pipelines;
- one-hot encoding and in-fold imputation;
- engineered financial ratios;
- repeated/OOF evaluation with class-aware metrics;
- Gender removal from inference and descriptive fairness checks;
- model card and reproducibility metadata;
- FastAPI, browser dashboard, tests, CI and Docker.

## High-capacity independent rebuild (v2)

Added later to address the small-data accuracy ceiling:

- separate 4,269-row loan-approval dataset with CIBIL and asset features;
- strict schema validation and feature-level deduplication;
- locked 20% final test set;
- Optuna search across Random Forest, Extra Trees, XGBoost and LightGBM;
- Logistic Regression reference model;
- out-of-fold ensemble and threshold optimization;
- raw-feature permutation importance and local sensitivity analysis;
- versioned `/api/v2` endpoints and enhanced dashboard mode.

## Resume integrity rule

If the project is listed under **Technofly Solutions — Data Science Intern**, phrase the v2 work as a later independent extension. Do not imply that FastAPI, Optuna, the 4,269-row dataset or the v2 ensemble were delivered during the original internship unless you independently have evidence that they were.

A defensible structure is:

**Loan Approval Prediction — Data Science Intern, Technofly Solutions**

- internship bullets for the original 614-row project;
- one optional bullet beginning **"Later independently rebuilt..."** for Loan Approval Prediction & Business Analysis.
