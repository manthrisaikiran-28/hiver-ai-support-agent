"""
Deterministic Evaluation Engine.
Performs fast, reproducible rule-based checks without relying on LLMs.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class DeterministicCheckResult:
    required_facts_pass: bool
    forbidden_claims_pass: bool
    intent_matched: bool
    escalation_matched: bool
    response_not_empty: bool
    grounded_in_context: bool
    overall_pass: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_deterministic(agent_output: Dict[str, Any],
                           golden_record: Dict[str, Any]) -> DeterministicCheckResult:
    response = agent_output.get("response", "").lower()
    predicted_intent = agent_output.get("intent", "")
    needs_human = agent_output.get("needs_human", False)

    expected_intent = golden_record.get("intent", golden_record.get("ideal_category", ""))
    expected_escalation = golden_record.get("expected_escalation", False)
    required_facts = golden_record.get("required_facts", [])
    forbidden_claims = golden_record.get("forbidden_claims", [])

    # Check 1: Response not empty
    not_empty = len(response.strip()) >= 15

    # Check 2: Required facts present
    missing_facts = []
    for fact in required_facts:
        if fact.lower() not in response:
            missing_facts.append(fact)
    facts_pass = len(missing_facts) == 0

    # Check 3: Forbidden claims absent
    found_forbidden = []
    for claim in forbidden_claims:
        if claim.lower() in response:
            found_forbidden.append(claim)
    forbidden_pass = len(found_forbidden) == 0

    # Check 4: Intent matched
    intent_pass = (predicted_intent == expected_intent)

    # Check 5: Escalation matched
    escalation_pass = (needs_human == expected_escalation)

    # Check 6: Grounded in context / non-empty
    retrieved_context = agent_output.get("retrieved_context", [])
    grounded_pass = not_empty  # Baseline check

    # Calculate aggregate deterministic score
    checks = [not_empty, facts_pass, forbidden_pass, intent_pass, escalation_pass, grounded_pass]
    pass_count = sum(1 for c in checks if c)
    score = round(pass_count / len(checks), 2)
    overall_pass = (score >= 0.83)

    return DeterministicCheckResult(
        required_facts_pass=facts_pass,
        forbidden_claims_pass=forbidden_pass,
        intent_matched=intent_pass,
        escalation_matched=escalation_pass,
        response_not_empty=not_empty,
        grounded_in_context=grounded_pass,
        overall_pass=overall_pass,
        score=score,
        details={
            "missing_facts": missing_facts,
            "found_forbidden": found_forbidden,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "expected_escalation": expected_escalation,
            "agent_escalation": needs_human,
        },
    )
