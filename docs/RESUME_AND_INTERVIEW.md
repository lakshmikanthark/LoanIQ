# Resume and interview material

## Internship-faithful resume entry

**Loan Approval Prediction | Data Science Intern — Technofly Solutions**

- Built an end-to-end Python classification workflow on a 614-application loan dataset, covering missing-value handling, categorical encoding, train/test validation and model serialization.
- Benchmarked 6 classifiers including Logistic Regression, SVM, Random Forest and XGBoost; Logistic Regression achieved ~84% accuracy on the project's original hold-out split.
- Integrated the selected model into a Tkinter interface for interactive applicant-level loan-approval predictions.
- Later audited the project independently and rebuilt its ML methodology, deployment and evaluation as Loan Approval Project.

## Portfolio extension entry

**Loan Approval Prediction & Business Analysis | Independent Business Analysis + ML Engineering Rebuild**

Use this version **after running the real v2 max-quality training** and replace bracketed values only with values from `backend/artifacts/enhanced/metrics.json`:

- Expanded the original 614-row internship classifier into a separate **4,269-record** CIBIL/assets modeling pipeline with schema validation, duplicate-leakage controls and inference-time financial feature engineering.
- Benchmarked and tuned Logistic Regression, Random Forest, Extra Trees, XGBoost and LightGBM using stratified CV + Optuna, then optimized an out-of-fold probability ensemble and decision threshold without touching the final test partition.
- Achieved **[FINAL_TEST_ACCURACY]% locked-test accuracy**, **[ROC_AUC] ROC-AUC** and **[BALANCED_ACCURACY]% balanced accuracy** on an untouched 20% holdout; stored reproducible data/model metadata including the training-file SHA-256.
- Productized the model through FastAPI v2 endpoints, batch inference, model-sensitivity explanations, a responsive dashboard, automated tests, CI and Docker/local launch workflows.

Do not replace the brackets with external repository/blog scores.

## 60-second interview explanation

"During my Data Science internship at Technofly Solutions, I built a loan-approval prediction project on a public dataset of 614 applications. I cleaned and encoded the data, compared six classification algorithms, selected Logistic Regression from the original hold-out evaluation and connected the saved model to a Tkinter UI.

Later, I audited that work and found two important limitations: the dataset was small and the original evaluation could be made more rigorous. I preserved it as a historical baseline, then built Enhanced Loan Approval Model on a separate 4,269-record approval dataset containing CIBIL and asset signals. I lock 20% of the data before any tuning, run all preprocessing inside pipelines, use Optuna to tune multiple tree and boosting families, generate out-of-fold probabilities, optimize a weighted ensemble and threshold on development data only, and evaluate the frozen model once on the holdout. I then exposed both generations through FastAPI and a dashboard. The main lesson was that a higher score only matters when the data and evaluation boundary are defensible."

## Questions you should be ready for

### Why didn't you just add synthetic rows to the 614 records?
Synthetic copies do not create new information and can inflate validation results if near-duplicates cross folds. I used a genuinely larger labeled dataset as a separate model generation instead.

### Why are v1 and v2 separate rather than concatenated?
Their schemas and data-generating populations differ. Concatenating them would leave large blocks of systematically missing features and imply comparability that isn't documented.

### What prevents test leakage?
The v2 test set is created before tuning. Preprocessing, hyperparameter search, OOF predictions, blend weights and threshold selection use only the 80% development partition. The 20% holdout is evaluated after selection is frozen.

### Why optimize the threshold?
A 0.5 cutoff is conventional, not automatically optimal for a fitted probability model. The threshold is selected from OOF development predictions, never from the holdout.

### Why use an ensemble?
Random Forest/Extra Trees and gradient-boosting models have different inductive biases. If their errors are not perfectly correlated, a weighted probability blend can outperform any one member. The blend search allows zero weights, so it can also fall back to a single model.

### Why report balanced accuracy if the goal is accuracy?
Headline accuracy is the explicit optimization goal, but balanced accuracy and class recall expose whether that gain comes from simply favoring the majority class.

### Is this a credit-risk/default model?
No. The label is approval/rejection, not subsequent default or repayment. I would not call it a probability-of-default model.

### Why can public-dataset accuracy be very high?
A public educational dataset can contain strong, simplified decision signals such as CIBIL and asset coverage. A high held-out score is evidence for that dataset split, not proof the model would generalize to a real bank's applicants.

---

# Business Analytics / Business Operations version

## Recommended project title

**Loan Approval Prediction & Business Analysis | Data Science Internship Project**

## Resume bullets — Business Analytics

- Reframed a loan-classification internship project into a **business-analysis project** that translates model probabilities into auto-decision, manual-review and exception-routing policies across a 614-application project simulation.
- Built a **what-if policy simulator and constrained optimizer** to maximize automated coverage subject to a historical-decision-agreement guardrail, quantifying the trade-off between automation, manual-review workload and consistency.
- Designed a KPI framework covering **automation rate, manual-review rate, processing time, SLA breaches, analyst-capacity utilization, backlog growth, documentation completeness and value routed for review**.
- Built segment/segment analysis and an executive opportunity score to identify which acquisition channels or applicant segments create disproportionate review effort and processing delay.
- Added reproducible data-quality checks and case-level decision audit trails that separate source attributes from simulated operational metadata and preserve model/version/threshold evidence.

## Resume bullets — Business Operations

- Built a lending **Business Analysis command center** that turns application scores into analyst queues, priority rules, SLA monitoring and staffing decisions instead of stopping at a model prediction.
- Developed a **capacity planner and demand stress test** that converts manual-review volume into analyst hours, utilization and projected backlog under 0.75×–2.0× demand scenarios.
- Prioritized human-review cases using an impact score combining model uncertainty, SLA pressure, document friction and loan value, enabling operators to focus effort on the highest-impact exceptions first.
- Created policy benchmarking across baseline, user-defined and optimized routing strategies to show how operating rules change workload, automated coverage and historical decision agreement.
- Productized the workflow through FastAPI endpoints, a responsive multi-page operations dashboard, automated tests, model governance documentation and reproducible local/Docker deployment.

## 30-second BA/BizOps interview pitch

"The original internship project predicted loan approval, but I realized a business team does not get value from a probability alone. I rebuilt it as Loan Approval Prediction & Business Analysis. The project uses the model score and answers operational questions: which applications can be automated, which should go to a human, which cases should be reviewed first, whether the analyst team has enough capacity, which segment is driving delays, and how policy changes affect the queue. I added what-if simulation, a threshold optimizer with a decision-consistency guardrail, capacity and backlog forecasting, segment scorecards, data-quality checks and case-level audit trails. The main project lesson was moving from predictive analytics to prescriptive operational decision-making."

## Why this project is relevant to a Business Analyst role

A Business Analyst is expected to frame problems, define KPIs, quantify trade-offs, identify drivers, communicate implications and help teams choose actions. Business Analysis demonstrates each of those skills in a concrete operating workflow.

## Why this project is relevant to a Business Operations role

Business Operations work frequently involves throughput, staffing, process design, exception management, service levels, prioritization and cross-functional decision support. Business Analysis demonstrates those capabilities directly rather than implying them from a machine-learning notebook.

## Important interview integrity boundary

The operational timeline, acquisition channel, analyst assignment, processing time, SLA and document-completeness fields are **deterministically simulated** because the public internship dataset does not contain real lender-operations telemetry. Say that directly. The simulation is there to demonstrate the analysis workflow and operating logic; it is not represented as real bank data.
