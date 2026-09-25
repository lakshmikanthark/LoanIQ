@echo off
setlocal
cd /d "%~dp0"

echo =====================================================
echo   Loan Approval Project - download data + MAX quality training
echo =====================================================

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.12+ and try again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend\requirements-dev.txt

python backend\scripts\bootstrap_enhanced.py --quality max
if errorlevel 1 (
  echo.
  echo Training failed. Read the error above; no fake v2 artifact was created.
  pause
  exit /b 1
)

echo.
echo Running test suite...
set PYTHONPATH=backend
pytest -q backend\tests

echo.
echo Enhanced Loan Approval Model is trained. Starting the project at http://127.0.0.1:8000
start "" http://127.0.0.1:8000
python -m uvicorn loan_prediction.api:app --app-dir backend --host 127.0.0.1 --port 8000
