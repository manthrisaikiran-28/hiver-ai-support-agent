"""
Regression tests for Evaluation Harness and Quality Thresholds.
"""

import pytest
from app.evaluation.evaluator import EvaluationHarness


def test_evaluation_pipeline_runs_and_meets_threshold():
    harness = EvaluationHarness()
    results = harness.run_evaluation()

    summary = results["summary"]
    assert summary["total_evaluated"] == 11
    assert summary["category_accuracy_pct"] >= 80.0, "Category accuracy regression below 80% threshold!"
    assert summary["avg_overall_score"] >= 3.0, "Average overall score regression below 3.0 threshold!"
