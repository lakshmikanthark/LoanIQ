from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, validate and train Enhanced Loan Approval Model")
    parser.add_argument("--quality", choices=["fast", "balanced", "max"], default="balanced")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    fetch = [str(HERE / "fetch_enhanced_data.py")]
    if args.force_download:
        fetch.append("--force")
    run(*fetch)
    run(str(HERE / "train_enhanced.py"), "--quality", args.quality)
    print("\nEnhanced Loan Approval Model enhanced model is ready.")


if __name__ == "__main__":
    main()
