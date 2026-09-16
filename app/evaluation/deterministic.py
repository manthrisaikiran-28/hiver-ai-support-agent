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
    case_id: str
    intent_pass: bool
    required_facts_pass: bool
    forbidden_claims_pass: bool
    escalation_pass: bool
    grounding_pass: bool
    deterministic_pass: bool
    deterministic_score: float  # 0.0 to 1.0
    details: Dict[str, Any]

    @property
    def score(self) -> float:
        return self.deterministic_score

    @property
    def overall_pass(self) -> bool:
        return self.deterministic_pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_deterministic(agent_output: Dict[str, Any],
                           golden_record: Dict[str, Any]) -> DeterministicCheckResult:
    case_id = golden_record.get("id", "")
    response = agent_output.get("response", "").lower()
    predicted_intent = agent_output.get("intent", "")
    needs_human = agent_output.get("needs_human", False)

    expected_intent = golden_record.get("intent", golden_record.get("ideal_category", ""))
    expected_escalation = golden_record.get("expected_escalation", False)
    required_facts = golden_record.get("required_facts", [])
    forbidden_claims = golden_record.get("forbidden_claims", [])

    # Check 1: Intent matched
    intent_pass = (predicted_intent == expected_intent)

    # Check 2: Required facts present
    missing_facts = []
    for fact in required_facts:
        if fact.lower() not in response:
            missing_facts.append(fact)
    required_facts_pass = (len(missing_facts) == 0)

    # Check 3: Forbidden claims absent
    found_forbidden = []
    for claim in forbidden_claims:
        if claim.lower() in response:
            found_forbidden.append(claim)
    forbidden_claims_pass = (len(found_forbidden) == 0)

    # Check 4: Escalation matched
    escalation_pass = (needs_human == expected_escalation)

    # Check 5: Grounding / non-empty response
    not_empty = len(response.strip()) >= 15
    grounding_pass = not_empty

    # Aggregate deterministic score
    checks = [intent_pass, required_facts_pass, forbidden_claims_pass, escalation_pass, grounding_pass]
    pass_count = sum(1 for c in checks if c)
    score = round(pass_count / len(checks), 2)
    deterministic_pass = (score >= 0.80 and forbidden_claims_pass)

    return DeterministicCheckResult(
        case_id=case_id,
        intent_pass=intent_pass,
        required_facts_pass=required_facts_pass,
        forbidden_claims_pass=forbidden_claims_pass,
        escalation_pass=escalation_pass,
        grounding_pass=grounding_pass,
        deterministic_pass=deterministic_pass,
        deterministic_score=score,
        details={
            "missing_facts": missing_facts,
            "found_forbidden": found_forbidden,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "expected_escalation": expected_escalation,
            "agent_escalation": needs_human,
        },
    )
