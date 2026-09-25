# Loan Approval Prediction & Business Analysis QA report

## Verified build scope

The final repository was tested as a lending-operations analytics project, not only as an ML API.

### Automated test suite

`PYTHONPATH=backend pytest -q backend/tests`

**26/26 tests passing** in the working build.

Coverage includes:

- legacy single prediction and batch prediction contracts
- gender rejection from the prediction payload
- model metadata / fairness contract
- dashboard serving and static path-traversal protection
- feature engineering and schema validation
- enhanced-dataset cleaning / duplicate-leakage controls
- Business Analysis deterministic project simulation
- Business Analysis policy metrics
- conservative-policy behavior
- demand-stress / capacity monotonicity
- constrained policy optimizer guardrail
- Business Analysis run_business_analysis API contract
- case-level audit endpoint and 404 behavior
- invalid policy-threshold validation

### Static / compile checks

- `python -m compileall -q backend/loan_prediction backend/scripts`
- `node --check frontend/dist/app.js`

Both pass.

### Live HTTP integration

A real Uvicorn process was launched and verified through HTTP:

- `GET /api/health` → 200, version `1.0.0`
- `/` → serves `Loan Approval Prediction & Business Analysis | Data Science Internship Project`
- `GET /api/analysis/status` → Business Analysis ready
- `POST /api/analysis/run_business_analysis` → 614-app dataset summary, funnel, trend, segments, queue, data quality and analysis summary returned

Baseline 75/25 policy in the deterministic simulation produced approximately:

- 66.3% automated decisions
- 99.3% agreement with historical decisions among automated cases
- 33.7% manual review

Those figures describe the project simulation and historical approval labels. They are **not** default-risk or real-bank production metrics.

## Product integrity checks

- Source applicant/approval fields remain distinct from simulated operations metadata.
- The UI labels the operations layer as a simulation.
- Approval probability is not described as probability of default.
- Model Performance is separated from Operations Analytics.
- The Project Overview explicitly separates internship work from later independent rebuild work.
- The high-capacity 4,269-row model is not claimed as trained unless its real artifact exists.

## Final packaging gate

Before delivery, the repository is cleaned of runtime caches and re-tested from a fresh extracted ZIP.
