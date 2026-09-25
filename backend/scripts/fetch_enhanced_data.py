from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from loan_prediction.enhanced_config import ENHANCED_DATA_PATH, RAW_FEATURES  # noqa: E402
from loan_prediction.enhanced_features import clean_enhanced_dataset  # noqa: E402

EXPECTED_RAW_ROWS = 4269

URLS = [
    "https://raw.githubusercontent.com/fahadsultan/csc272/refs/heads/main/data/loan_approval_dataset.csv",
    "https://raw.githubusercontent.com/AbhishekBiswas-github/AI-Engineer-Projects/refs/heads/main/Machine-Learning/Loan-Approval-Prediction/loan_approval_dataset.csv",
]


def validate(path: Path) -> dict:
    frame = pd.read_csv(path)
    clean, report = clean_enhanced_dataset(frame)
    required = set(RAW_FEATURES) | {"loan_id", "loan_status", "target"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Enhanced dataset is missing required columns: {sorted(missing)}")
    rows_before = int(report.get("rows_before_deduplication", len(frame)))
    if rows_before != EXPECTED_RAW_ROWS:
        raise ValueError(f"Dataset drift detected: expected {EXPECTED_RAW_ROWS} raw rows, found {rows_before}")
    if len(clean) < 4000:
        raise ValueError(f"Too few usable rows after validation/deduplication: {len(clean)}")
    valid_targets = set(clean["loan_status"].dropna().unique())
    if not valid_targets.issubset({"Approved", "Rejected"}):
        raise ValueError(f"Unexpected target values: {sorted(valid_targets)}")
    if clean["target"].isna().any():
        raise ValueError("Some target values could not be mapped")
    return report


def fetch(destination: Path = ENHANCED_DATA_PATH) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Loan-Approval-Internship-Project/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            if len(payload) < 300_000:
                raise ValueError(f"download unexpectedly small ({len(payload)} bytes)")
            # Parse once before touching the final path.
            pd.read_csv(io.BytesIO(payload))
            tmp = destination.with_suffix(".download.csv")
            tmp.write_bytes(payload)
            report = validate(tmp)
            tmp.replace(destination)
            return {"url": url, "path": str(destination), **report}
        except Exception as exc:  # pragma: no cover - network fallback
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the enhanced dataset.\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the public 4,269-row Enhanced Loan Approval Model dataset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if ENHANCED_DATA_PATH.exists() and not args.force:
        report = validate(ENHANCED_DATA_PATH)
        print(f"Dataset already present: {ENHANCED_DATA_PATH}")
        print(report)
        return
    report = fetch()
    print("Downloaded and validated enhanced dataset")
    print(report)


if __name__ == "__main__":
    main()
