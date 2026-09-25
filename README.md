# Loan Approval Prediction & Business Analysis — Data Science Internship Project

> **College Data Science internship project that predicts loan approval and extends the model with business analysis for review workload, staffing, policy scenarios and process bottlenecks.**

This project started as a **Data Science internship project at Technofly Solutions**: enter applicant details and predict whether a loan application looks more like historically **approved** or **rejected** cases.

That solved the ML problem. It did **not** solve the business problem.

A real lending operations team still has to answer:

- Which applications can be handled automatically?
- Which ones require a human analyst?
- Which manual-review cases should be worked first?
- Do we have enough analysts for today's demand?
- When will backlog start growing?
- Which segment or process is causing delays?
- What happens if application volume increases by 50% or 100%?
- How do different approval/rejection thresholds affect workload and consistency?

**The project was later extended to answer those business questions as well.** The ML model is one part of the analysis, not the entire project.

---

## What is this project? — 30-second explanation

Imagine a lending team receives **1,000 loan applications today**.

Instead of asking employees to inspect all 1,000 equally, Loan Approval Project scores each application and applies an operating policy:

```text
1,000 applications
        │
        ▼
  ML approval score
        │
        ▼
 Decision policy
   ┌────┼───────────────┐
   ▼    ▼               ▼
Auto   Manual          Auto
approve review         reject
        │
        ▼
Priority queue
        │
        ▼
Analyst capacity + SLA + backlog
        │
        ▼
Segment analysis + management action
```

For example:

- **94% approval probability** → high-confidence case → candidate for **Auto Approve**
- **51% approval probability** → uncertain case → **Manual Review**
- **7% approval probability** → low-confidence case → candidate for **Auto Reject**

The exact thresholds are configurable. Loan Approval Project lets an operations manager test those thresholds before adopting them and see the impact on workload, automation and historical decision consistency.

> This is an educational college project, not a production underwriting system. Approval probability is **not** probability of default.

For the full plain-English walkthrough, read [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md).

---

## The business problem it solves

A prediction alone does not tell an operations team what to **do**.

Loan Approval Project converts prediction output into five management decisions:

### 1. Routing
Decide whether an application should be:

- auto-approved,
- sent to manual review, or
- auto-rejected.

### 2. Prioritization
When humans are required, rank the review queue using:

- model uncertainty,
- SLA pressure,
- documentation friction,
- loan value at stake.

### 3. Policy
Let managers test different approve/reject thresholds and quantify the trade-off between:

- automation,
- human workload,
- historical decision agreement.

### 4. Capacity
Translate manual-review volume into:

- analyst hours,
- utilization,
- required staffing,
- backlog growth,
- maximum sustainable demand.

### 5. Process improvement
Identify which segments create disproportionate:

- manual review,
- long turnaround time,
- SLA breaches,
- document-quality problems.

This moves the project from **predictive analytics** ("what is likely to happen?") toward **prescriptive analytics** ("what should the team do next?").

---

## What a manager can do inside Loan Approval Project

### Project Dashboard
See the analysis summary of the simulated lending dataset at a glance:

- application volume,
- automation rate,
- manual-review rate,
- analyst utilization,
- backlog growth,
- SLA breaches,
- average processing time,
- analyst hours saved.

### Policy Analysis
Change the auto-approve / auto-reject thresholds and immediately see how the decision changes affect workload and consistency.

Example:

```text
Policy A: conservative thresholds
→ more manual review
→ higher historical agreement
→ more analyst workload

Policy B: wider automation band
→ fewer manual reviews
→ more automation
→ potentially lower historical agreement
```

Loan Approval Project also includes a **constrained policy optimizer** that searches for the highest automation level while respecting a user-selected historical-agreement guardrail.

### Staffing & Demand Simulator
Change analyst headcount or expected demand and see:

- utilization,
- analyst hours required,
- backlog growth,
- maximum demand the team can sustain.

This answers questions such as:

> "Can the current team handle a 50% increase in applications?"

> "How many analysts are required before backlog stops growing?"

### Manual Review Prioritization
Manual-review applications are ranked so analysts work on the highest-impact exceptions first instead of processing a random list.

### Segment Analysis
Break operational performance down by segments such as property area, education, employment type, credit history and simulated acquisition channel to find where avoidable workload or delay is concentrated.

### Application Detail View
Open any case and inspect:

- applicant attributes,
- historical outcome,
- model probability,
- routing decision,
- priority rationale,
- model version,
- threshold evidence,
- which fields are real source data vs simulated operations data.

### Analysis Summary
Turn the current operating scenario into a concise management readout such as:

- capacity is tight,
- backlog is expected to grow,
- one segment is creating disproportionate manual-review effort,
- specific high-priority applications require attention.

---

## Why this is a Business Analytics / Business Operations project

Loan Approval Project is intentionally designed to demonstrate more than model training.

| Capability | Where it appears in the project |
|---|---|
| KPI design | automation, review rate, SLA, TAT, utilization, backlog |
| What-if analysis | Policy Analysis and demand/staffing simulator |
| Segment analysis | segment diagnostics and opportunity ranking |
| Operations planning | analyst capacity and sustainable-demand calculations |
| Prioritization | manual-review priority score and queue |
| Decision support | policy optimizer and analysis summary |
| Data governance | quality checks, provenance and audit trail |
| Data Science | v1/v2 modeling, evaluation and explainability |
| Applied problem solving | turning a model into an business analysis workflow |

The intended project story is:

> **"I did not stop at predicting an outcome. I asked how a business team would use that prediction every day, then built the analytics and analysis system around it."**

---

## Original internship work vs later rebuild

This boundary is deliberately explicit.

### During the Technofly Solutions internship

The original project used a **614-row public loan dataset**, compared classic classifiers and connected the selected Logistic Regression model to a Tkinter prediction UI.

### Later independent rebuild

Loan Approval Project adds:

- stronger leakage-safe ML methodology,
- an optional **4,269-record** enhanced model,
- FastAPI services,
- Business Analysis analytics,
- policy simulation and optimization,
- staffing/capacity planning,
- priority queues,
- root-cause diagnostics,
- model/data governance,
- automated tests, CI and deployment tooling.

The repository does **not** present the later rebuild as work completed during the internship.

See [`docs/PROJECT_EVOLUTION.md`](docs/PROJECT_EVOLUTION.md) for the full boundary.

---

## Data honesty / project boundary

The original public dataset contains applicant attributes and historical approval labels. It does **not** contain real lender operations telemetry such as timestamps, acquisition channels, analyst assignments, SLA records or document-quality events.

To demonstrate Business Analytics / Business Operations workflows, Loan Approval Project adds a **deterministic simulation layer** for those operational fields. The UI and documentation label them as simulated.

The project does **not** claim to:

- predict probability of default,
- estimate expected credit loss,
- determine affordability,
- represent simulated operations data as real lender telemetry,
- be validated for production lending decisions.

That limitation is part of the design, not hidden in fine print.

---

## Product architecture

```text
PUBLIC LOAN DATA + HISTORICAL APPROVAL LABELS
                 │
                 ├── leakage-safe ML pipelines
                 ├── v1 audited internship model
                 └── optional v2 high-capacity model
                 │
                 ▼
          DECISION PROBABILITY
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   Auto approve  Human    Auto reject
                 review
                   │
          priority + SLA queue
                   │
                   ▼
          OPERATIONS ANALYTICS
        ├── automation / workload
        ├── capacity / backlog
        ├── policy simulation
        ├── segment root causes
        ├── data quality
        └── decision audit
```

## Main project modules

- **Project Dashboard** — executive operating health
- **Policy Analysis** — threshold what-if analysis + optimizer
- **Review Queue** — prioritized manual-review workload
- **Segment Analysis** — segment and process diagnostics
- **Loan Prediction** — applicant-level model scoring
- **Model Performance** — evaluation and model metadata
- **Project Overview** — internship-to-Business Analysis evolution

## Business Analysis API

- `GET /api/analysis/status`
- `POST /api/analysis/run_business_analysis`
- `POST /api/analysis/optimize-policy`
- `GET /api/analysis/case/{application_id}`

---

## GitHub / recruiter presentation

The repository is intentionally written so a non-technical recruiter can understand the project before reading the ML implementation. For the exact GitHub **About** description, suggested topics and sharing copy, see [`docs/GITHUB_GUIDE.md`](docs/GITHUB_GUIDE.md).

---

## Why v2 exists

The 614-row internship dataset is too small and information-limited to justify chasing a much higher score by repeatedly tuning against the same records. Enhanced Loan Approval Model improves the problem honestly by adding a larger labeled dataset and stronger financial predictors rather than duplicating rows or reporting a cherry-picked split.

The v2 dataset has **4,269 applications** and includes:

`no_of_dependents`, `education`, `self_employed`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, residential/commercial/luxury asset values, bank asset value and `loan_status`.

The data file is fetched and validated by the project rather than silently mixed with the legacy schema.

## ML architecture

```text
PUBLIC v2 CSV (4,269 labeled applications)
        │
        ├─ schema / target validation
        ├─ feature-level duplicate removal (loan_id ignored)
        ├─ negative-asset quality correction + audit report
        │
        ▼
80% DEVELOPMENT SET ───────────────────────────────┐
        │                                           │
        ├─ feature engineering                      │
        ├─ imputation + one-hot encoding in Pipeline│
        ├─ stratified cross-validation              │
        ├─ Optuna hyperparameter search             │
        │   ├─ Extra Trees                          │
        │   ├─ Random Forest                        │
        │   ├─ XGBoost                              │
        │   └─ LightGBM                             │
        ├─ Logistic Regression baseline             │
        ├─ out-of-fold probabilities                │
        └─ ensemble weights + threshold selected    │
                                                    │
20% LOCKED TEST SET ◄───────────────────────────────┘
        │  never used for model / threshold selection
        ▼
ONE FINAL EVALUATION
        │
        ├─ accuracy + balanced accuracy
        ├─ ROC-AUC + Brier score
        ├─ class precision / recall / F1
        ├─ confusion matrix
        └─ post-evaluation permutation importance
        ▼
weighted champion ensemble → FastAPI → Loan Approval Project dashboard
```

## What the rebuild fixes

- No target leakage or preprocessing fitted on validation/test records.
- No `Loan_ID`/`loan_id` modeling feature.
- Feature-identical duplicate records are removed even when IDs differ.
- No synthetic oversampling is used to manufacture a higher headline accuracy.
- The final 20% test set is locked before tuning and evaluated once.
- Hyperparameters are searched only on the development split.
- Ensemble weights and the decision threshold are selected from out-of-fold training predictions.
- Accuracy is reported together with balanced accuracy, ROC-AUC and class recall.
- Model metadata records row counts, dataset SHA-256, random seed, members, weights and threshold.
- Explanations are described as **model sensitivity**, not causal reasons.

## v2 engineered features

In addition to the raw fields, v2 creates only inference-time-available features:

- total asset value
- loan-to-income ratio
- asset-to-loan coverage
- bank-assets-to-income ratio
- loan amount per term year
- net-worth proxy (`assets - requested loan`)
- CIBIL × asset-coverage interaction
- CIBIL band

No target-derived field is used.

## Train the strongest version

### Windows

Double-click:

```text
train_best_windows.bat
```

or run:

```powershell
python backend/scripts/bootstrap_enhanced.py --quality max
```

### macOS / Linux

```bash
./train_best_unix.sh
```

or:

```bash
python backend/scripts/bootstrap_enhanced.py --quality max
```

`bootstrap_enhanced.py` downloads the 4,269-row dataset, validates it, then performs the full max-quality search. The exact final metrics are written to:

```text
backend/artifacts/enhanced/metrics.json
backend/artifacts/enhanced/model_metadata.json
backend/artifacts/enhanced/feature_importance.json
```

Do **not** copy a performance number from a blog or another repository. Use only the score produced by your own locked-test run.

## Quality modes

| Mode | CV | Optuna trials/model | Purpose |
|---|---:|---:|---|
| `fast` | 3-fold | 3 | smoke tests / development |
| `balanced` | 5-fold | 12 | normal local training |
| `max` | 5-fold | 50 | strongest training run |

You can override the trial count directly with `train_enhanced.py --trials N`.

## Run the project

After dependencies are installed:

```bash
uvicorn loan_prediction.api:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open:

- Dashboard: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`

The UI automatically exposes v2 when its trained artifact exists and keeps v1 available as the historical baseline.

## API generations

### Business Analysis business analytics

- `GET /api/analysis/status`
- `POST /api/analysis/run_business_analysis`
- `POST /api/analysis/optimize-policy`
- `GET /api/analysis/case/{application_id}`

### v2 enhanced

- `GET /api/v2/status`
- `GET /api/v2/model`
- `POST /api/v2/predict`
- `POST /api/v2/predict/batch`

### v1 internship baseline

- `GET /api/v1/model`
- `POST /api/v1/predict`
- `POST /api/v1/predict/batch`

## Reproducibility and QA

```bash
pip install -r backend/requirements-dev.txt
PYTHONPATH=backend pytest -q backend/tests
node --check frontend/dist/app.js
```

The repository also includes GitHub Actions, Docker configuration, strict Pydantic validation, model/data metadata and documented provenance.

## Repository map

```text
backend/loan_prediction/       v1 + v2 reusable ML/API package
backend/scripts/train.py         audited 614-row baseline training
backend/scripts/fetch_enhanced_data.py
backend/scripts/train_enhanced.py
backend/scripts/bootstrap_enhanced.py
backend/tests/                   API / schema / leakage / feature tests
backend/artifacts/               versioned generated model artifacts
data/raw/loan_applications.csv   original internship dataset
data/README.md                   provenance and schema notes
frontend/dist/                   build-free Business Analysis dashboard
docs/BUSINESS_ANALYSIS_FRAMEWORK.md    Business Analytics / Operations project framework
docs/MODEL_IMPROVEMENT.md         v2 evaluation design
docs/DATA_PROVENANCE.md          data-source boundaries
docs/MODEL_CARD.md               intended use + limitations
docs/PROJECT_EVOLUTION.md      resume-integrity boundary
docs/RESUME_AND_INTERVIEW.md     defensible project narrative
```

## Internship integrity

The original internship artifact used the 614-row dataset, classic classifiers and a Tkinter application. The 4,269-row v2 model, Optuna search, ensemble, FastAPI app, modern dashboard, tests, CI and Docker are **later independent engineering improvements**. Keep that distinction explicit on a resume and in interviews.
