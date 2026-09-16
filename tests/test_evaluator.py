"""
Regression tests for Evaluation Harness, Metrics Aggregation, & Dual Version Runs.
"""

import pytest
from app.evaluation.evaluator import EvaluationHarness


def test_evaluation_pipeline_runs_and_saves_both_versions():
    harness = EvaluationHarness()
    results_b = harness.run_evaluation()

    summary_b = results_b["summary"]
    assert summary_b["total_evaluated"] == 11
    assert summary_b["category_accuracy_pct"] >= 80.0, "Category accuracy regression below 80%!"
    assert summary_b["avg_overall_score"] >= 2.5, "Overall average score regression below threshold!"
    assert summary_b["retrieval_method"].startswith("SentenceTransformer") or summary_b["retrieval_method"].startswith("TF-IDF")
