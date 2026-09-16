"""
Orchestrates full evaluation pipeline (Version A TF-IDF & Version B SentenceTransformers leave-one-out evaluation).
Usage: python run_eval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.evaluation.evaluator import run_eval

if __name__ == "__main__":
    run_eval()
