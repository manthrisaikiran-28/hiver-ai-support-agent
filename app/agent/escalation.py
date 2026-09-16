"""
Escalation Logic & Policy Evaluation Engine.
Determines whether a ticket requires human agent escalation based on risk, complexity, or evidence thresholds.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List


RETRIEVAL_THRESHOLD = 0.25

ESCALATION_KEYWORDS = {
    "data_loss": ["data missing", "history gone", "deleted", "compliance", "lost history"],
    "account_lockout": ["2fa", "authenticator", "locked out", "backup codes", "manual reset"],
    "churn_risk": ["considering switching", "not happy", "cancel", "worst support", "leaving"],
    "systemic_bug": ["duplicate tickets", "40 duplicate", "all rules broken", "outage"],
}


@dataclass
class EscalationDecision:
    needs_human: bool
    escalation_reason: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"


def evaluate_escalation(query: str,
                        detected_intents: List[str],
                        retrieval_top_similarity: float,
                        is_sufficient_context: bool,
                        threshold: float = RETRIEVAL_THRESHOLD) -> EscalationDecision:
    """
    Evaluate policy conditions to determine if a ticket should be escalated.
    """
    query_lower = query.lower()

    # Rule 1: Insufficient context or low similarity
    if not is_sufficient_context or retrieval_top_similarity < threshold:
        return EscalationDecision(
            needs_human=True,
            escalation_reason=f"Insufficient retrieved evidence context (top similarity {retrieval_top_similarity:.4f} below threshold {threshold}).",
            risk_level="MEDIUM",
        )

    # Rule 2: Compliance / Data Loss
    for kw in ESCALATION_KEYWORDS["data_loss"]:
        if kw in query_lower:
            return EscalationDecision(
                needs_human=True,
                escalation_reason="Data loss or compliance impact detected; requires immediate tier-2 human escalation.",
                risk_level="CRITICAL",
            )

    # Rule 3: 2FA / Account Lockout requiring manual identity verification
    for kw in ESCALATION_KEYWORDS["account_lockout"]:
        if kw in query_lower:
            return EscalationDecision(
                needs_human=True,
                escalation_reason="2FA / Account Lockout requires manual identity verification by human support.",
                risk_level="HIGH",
            )

    # Rule 4: Systemic bug / high volume duplicate tickets
    for kw in ESCALATION_KEYWORDS["systemic_bug"]:
        if kw in query_lower:
            return EscalationDecision(
                needs_human=True,
                escalation_reason="Systemic technical issue impacting multiple tickets or rules.",
                risk_level="HIGH",
            )

    # Rule 5: Churn Risk
    for kw in ESCALATION_KEYWORDS["churn_risk"]:
        if kw in query_lower:
            return EscalationDecision(
                needs_human=True,
                escalation_reason="High churn risk detected; requires human retention review.",
                risk_level="HIGH",
            )

    return EscalationDecision(
        needs_human=False,
        escalation_reason="Standard inquiry resolved via automated knowledge base guidance.",
        risk_level="LOW",
    )
