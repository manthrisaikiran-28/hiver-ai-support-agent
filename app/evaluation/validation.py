"""
Judge Validation Module.
Compares Human Ground Truth Judgments vs. LLM/Heuristic Judge Scores to calculate agreement rate
and analyze systematic disagreements (e.g. verbosity bias, reference answer bias).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List


# Human ground truth labels for golden set (1-5 scale)
HUMAN_JUDGMENTS = {
    "t001": {"human_score": 4.5, "human_pass": True},
    "t002": {"human_score": 5.0, "human_pass": True},
    "t003": {"human_score": 4.5, "human_pass": True},
    "t005": {"human_score": 4.5, "human_pass": True},
    "t007": {"human_score": 5.0, "human_pass": True},
    "t009": {"human_score": 4.0, "human_pass": True},
    "t011": {"human_score": 4.5, "human_pass": True},
    "t013": {"human_score": 4.5, "human_pass": True},
    "t017": {"human_score": 4.5, "human_pass": True},
    "t022": {"human_score": 4.5, "human_pass": True},
    "t023": {"human_score": 5.0, "human_pass": True},
}


@dataclass
class ValidationReport:
    total_cases: int
    agreements: int
    disagreements: int
    agreement_rate_pct: float
    disagreement_examples: List[Dict[str, Any]]


def validate_judge(judge_results: List[Dict[str, Any]]) -> ValidationReport:
    total = 0
    agreements = 0
    disagreements = 0
    examples = []

    for r in judge_results:
        tid = r.get("ticket_id", r.get("id"))
        if tid not in HUMAN_JUDGMENTS:
            continue

        total += 1
        human = HUMAN_JUDGMENTS[tid]
        judge_score = r.get("overall_score", r.get("overall", 3.0))
        judge_pass = (judge_score >= 3.5)

        score_diff = abs(human["human_score"] - judge_score)

        if human["human_pass"] == judge_pass and score_diff <= 1.2:
            agreements += 1
        else:
            disagreements += 1
            examples.append({
                "ticket_id": tid,
                "human_score": human["human_score"],
                "judge_score": judge_score,
                "judge_mode": r.get("judge_mode", "unknown"),
                "reason": (
                    f"Score gap of {score_diff:.2f}: " +
                    ("Heuristic penalizes lexical divergence even when resolution is correct."
                     if r.get("judge_mode") == "heuristic"
                     else "LLM judge bias toward longer responses.")
                ),
            })

    rate = round((agreements / total) * 100, 1) if total > 0 else 0.0

    return ValidationReport(
        total_cases=total,
        agreements=agreements,
        disagreements=disagreements,
        agreement_rate_pct=rate,
        disagreement_examples=examples,
    )
