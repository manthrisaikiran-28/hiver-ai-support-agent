"""
Failure Analysis & Categorization Module.
Analyzes evaluation results to categorize root causes of agent and evaluation failures.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
from collections import Counter


@dataclass
class FailureRecord:
    case_id: str
    query: str
    expected_behavior: str
    actual_response: str
    retrieved_context: List[Dict[str, Any]]
    retrieval_similarity: float
    deterministic_score: float
    judge_score: float
    failure_type: str
    root_cause: str
    suggested_improvement: str


def analyze_failures(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures: List[FailureRecord] = []

    for r in eval_results:
        case_id = r.get("id", r.get("case_id", ""))
        query = r.get("query", r.get("subject", ""))
        actual_response = r.get("agent_response", r.get("response", ""))
        expected_behavior = r.get("expected_answer", r.get("ideal_response", ""))

        det_score = r.get("deterministic_score", 1.0)
        judge_score = r.get("overall_score", r.get("overall", 5.0))
        cat_correct = r.get("category_correct", True)
        needs_human = r.get("needs_human", False)
        expected_escalation = r.get("expected_escalation", False)
        missing_facts = r.get("deterministic_details", {}).get("missing_facts", [])

        # Identify failure conditions
        failure_type = "NONE"
        root_cause = "No failure observed; agent response met required criteria."
        improvement = "Maintain current pattern."

        if not cat_correct:
            failure_type = "WRONG_INTENT"
            root_cause = f"Intent misclassified as '{r.get('predicted_category')}' instead of expected '{r.get('ideal_category')}'."
            improvement = "Expand intent keyword pattern matching or add vector intent classification."
        elif needs_human != expected_escalation:
            if needs_human:
                failure_type = "UNNECESSARY_ESCALATION"
                root_cause = "Agent triggered false positive escalation on standard query."
                improvement = "Tune risk keyword sensitivity threshold."
            else:
                failure_type = "INCORRECT_ESCALATION"
                root_cause = "Agent failed to escalate high risk / data loss / churn ticket to human support."
                improvement = "Add strict policy rules for data loss, 2FA lockout, and churn risk."
        elif missing_facts:
            failure_type = "INCOMPLETE_ANSWER"
            root_cause = f"Response omitted required resolution facts: {missing_facts}."
            improvement = "Refine prompt/template to ensure multi-intent completeness."
        elif judge_score < 3.5 or det_score < 0.8:
            failure_type = "GENERATION_FAILURE"
            root_cause = "Response quality score below target threshold."
            improvement = "Enhance context grounding and answer structure."

        if failure_type != "NONE":
            failures.append(FailureRecord(
                case_id=case_id,
                query=query,
                expected_behavior=expected_behavior,
                actual_response=actual_response,
                retrieved_context=r.get("retrieved_context", []),
                retrieval_similarity=r.get("retrieval_similarity", 0.0),
                deterministic_score=det_score,
                judge_score=judge_score,
                failure_type=failure_type,
                root_cause=root_cause,
                suggested_improvement=improvement,
            ))

    counts = Counter(f.failure_type for f in failures)
    total_eval = len(eval_results)
    stats = []
    for ftype, cnt in counts.items():
        stats.append({
            "failure_type": ftype,
            "count": cnt,
            "percentage": round((cnt / total_eval) * 100, 1) if total_eval > 0 else 0.0,
        })

    return {
        "total_failures": len(failures),
        "failure_rate_pct": round((len(failures) / total_eval) * 100, 1) if total_eval > 0 else 0.0,
        "statistics": stats,
        "failure_records": [asdict(f) for f in failures],
    }
