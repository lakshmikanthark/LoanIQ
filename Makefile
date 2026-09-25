.PHONY: train-v1 fetch-v2 train-v2 train-v2-max bootstrap-v2 run test check

train-v1:
	PYTHONPATH=backend python backend/scripts/train.py

fetch-v2:
	python backend/scripts/fetch_enhanced_data.py

train-v2:
	python backend/scripts/train_enhanced.py --quality balanced

train-v2-max:
	python backend/scripts/train_enhanced.py --quality max

bootstrap-v2:
	python backend/scripts/bootstrap_enhanced.py --quality max

run:
	uvicorn loan_prediction.api:app --app-dir backend --reload --port 8000

test:
	PYTHONPATH=backend pytest -q backend/tests

check:
	PYTHONPATH=backend pytest -q backend/tests
	node --check frontend/dist/app.js
