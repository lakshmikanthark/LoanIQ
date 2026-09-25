@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   Loan Approval Project - local setup and launch
echo ==========================================

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
pip install -r backend\requirements.txt

start "" http://127.0.0.1:8000
python -m uvicorn loan_prediction.api:app --app-dir backend --host 127.0.0.1 --port 8000
