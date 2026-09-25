#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements-dev.txt
python backend/scripts/bootstrap_enhanced.py --quality max
PYTHONPATH=backend pytest -q backend/tests
printf 'Loan Approval Project: http://127.0.0.1:8000\nAPI docs: http://127.0.0.1:8000/docs\n'
python -m uvicorn loan_prediction.api:app --app-dir backend --host 127.0.0.1 --port 8000
