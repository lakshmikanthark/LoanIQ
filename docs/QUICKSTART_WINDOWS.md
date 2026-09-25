# Windows quick start

## Option A — strongest enhanced model + project

1. Install **Python 3.12+** from python.org and enable **Add Python to PATH**.
2. Extract the repository ZIP.
3. Double-click `train_best_windows.bat`.

The script creates `.venv`, installs dependencies, fetches and validates the 4,269-row v2 dataset, performs max-quality training, runs the test suite and launches the app at `http://127.0.0.1:8000`.

The exact trained metrics are saved under `backend/artifacts/enhanced/`.

## Option B — open the project immediately with the verified v1 baseline

Double-click:

```text
run_windows.bat
```

Then open `http://127.0.0.1:8000` if the browser does not open automatically.

Loan Approval Project will show v1 until the real enhanced artifact exists. The Enhanced v2 tab tells you the one-line training command.

## Manual v2 command

From PowerShell in the extracted folder:

```powershell
.\.venv\Scripts\activate
python backend\scripts\bootstrap_enhanced.py --quality max
```

If the source download is blocked by your network, obtain `loan_approval_dataset.csv` from the documented source, save it as `data\raw\loan_approval_4269.csv`, then run:

```powershell
python backend\scripts\fetch_enhanced_data.py
python backend\scripts\train_enhanced.py --quality max
```

The first command validates an already-present file instead of downloading it again.
