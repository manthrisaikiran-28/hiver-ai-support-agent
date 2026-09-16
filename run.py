"""
Root CLI entry point for Hiver AI Support Agent & Evaluation Platform.
Usage:
  python run.py prepare     # Run data preparation & profiling
  python run.py eval        # Run full golden set evaluation pipeline
  python run.py dashboard   # Launch Streamlit interactive dashboard
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import subprocess
from scripts.prepare_data import run_data_pipeline
from app.evaluation.evaluator import run_eval


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent & Evaluation Platform CLI")
    parser.add_argument("command", choices=["prepare", "eval", "dashboard"], help="Action command to execute")
    args = parser.parse_args()

    if args.command == "prepare":
        run_data_pipeline()
    elif args.command == "eval":
        run_eval()
    elif args.command == "dashboard":
        print("Launching Streamlit Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui/dashboard.py"])


if __name__ == "__main__":
    main()
