"""
Metrics Calculation & Aggregation Module.
Computes overall evaluation metrics, per-category precision/recall, and score distributions.
"""

from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict


def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    pass_count = sum(1 for r in results if r.get("overall_pass", r.get("category_correct", False)))
    cat_correct_count = sum(1 for r in results if r.get("category_correct", False))

    avg_overall = sum(r.get("overall_score", r.get("overall", 0.0)) for r in results) / n
    avg_correctness = sum(r.get("correctness", 0.0) for r in results) / n
    avg_completeness = sum(r.get("completeness", 0.0) for r in results) / n
    avg_relevance = sum(r.get("relevance", 4.0) for r in results) / n
    avg_groundedness = sum(r.get("groundedness", 4.0) for r in results) / n
    avg_det_score = sum(r.get("deterministic_score", 1.0) for r in results) / n
    avg_semantic_sim = sum(r.get("semantic_similarity", 0.0) for r in results) / n
    avg_latency = sum(r.get("agent_latency_ms", 0.0) for r in results) / n

    # Per-category metrics calculation
    cat_stats = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
    for r in results:
        cat = r.get("ideal_category", r.get("true_category", "unknown"))
        cat_stats[cat]["total"] += 1
        if r.get("category_correct", False):
            cat_stats[cat]["correct"] += 1
        cat_stats[cat]["scores"].append(r.get("overall_score", r.get("overall", 0.0)))

    per_category = {}
    for cat, data in cat_stats.items():
        tot = data["total"]
        cor = data["correct"]
        acc = (cor / tot) if tot > 0 else 0.0
        avg_s = sum(data["scores"]) / tot if tot > 0 else 0.0
        per_category[cat] = {
            "total_cases": tot,
            "correct": cor,
            "accuracy": round(acc, 2),
            "avg_score": round(avg_s, 2),
        }

    # Per-difficulty breakdown
    diff_stats = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
    for r in results:
        diff = r.get("difficulty", "medium")
        diff_stats[diff]["total"] += 1
        if r.get("category_correct", False):
            diff_stats[diff]["correct"] += 1
        diff_stats[diff]["scores"].append(r.get("overall_score", r.get("overall", 0.0)))

    per_difficulty = {}
    for diff, data in diff_stats.items():
        tot = data["total"]
        cor = data["correct"]
        acc = (cor / tot) if tot > 0 else 0.0
        avg_s = sum(data["scores"]) / tot if tot > 0 else 0.0
        per_difficulty[diff] = {
            "total_cases": tot,
            "accuracy": round(acc, 2),
            "avg_score": round(avg_s, 2),
        }

    return {
        "total_evaluated": n,
        "pass_rate_pct": round((pass_count / n) * 100, 1),
        "category_accuracy_pct": round((cat_correct_count / n) * 100, 1),
        "avg_overall_score": round(avg_overall, 2),
        "avg_correctness": round(avg_correctness, 2),
        "avg_completeness": round(avg_completeness, 2),
        "avg_relevance": round(avg_relevance, 2),
        "avg_groundedness": round(avg_groundedness, 2),
        "avg_deterministic_score": round(avg_det_score, 2),
        "avg_semantic_similarity": round(avg_semantic_sim, 4),
        "avg_agent_latency_ms": round(avg_latency, 2),
        "per_category_breakdown": per_category,
        "per_difficulty_breakdown": per_difficulty,
    }
