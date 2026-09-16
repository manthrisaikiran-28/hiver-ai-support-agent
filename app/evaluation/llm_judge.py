"""
LLM-as-a-Judge Evaluator & Bug-Fixed Heuristic Scorer.
Evaluates agent responses across 6 rubric dimensions: correctness, relevance, completeness,
groundedness, helpfulness, and escalation appropriateness.
"""

from __future__ import annotations
import json
import os
import re
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from app.agent.prompts import JUDGE_PROMPT_TEMPLATE
from app.evaluation.semantic import compute_semantic_similarity


@dataclass
class JudgeResult:
    ticket_id: str
    overall_score: float  # 1.0 - 5.0
    correctness: float
    relevance: float
    completeness: float
    groundedness: float
    helpfulness: float
    escalation: float
    reason: str
    failure_type: str  # "NONE", "RETRIEVAL_FAILURE", "GENERATION_FAILURE", "HALLUCINATION", "MISSING_CONTEXT", "WRONG_INTENT", "INCORRECT_ESCALATION", "UNNECESSARY_ESCALATION", "INCOMPLETE_ANSWER"
    judge_mode: str    # "llm" or "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _call_llm_api(prompt: str) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
    body = json.dumps({
        "model": model,
        "max_tokens": 400,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(block.get("text", "") for block in data.get("content", []))
    except Exception as e:
        return None


def _heuristic_judge_fixed(ticket_id: str,
                           query: str,
                           predicted_category: str,
                           agent_response: str,
                           golden: Dict[str, Any],
                           needs_human: bool) -> JudgeResult:
    """
    Bug-Fixed Offline Heuristic Judge.
    Evaluates category fit, semantic overlap, intent coverage, and escalation correctness
    without the flawed 'and'/'both' false penalty.
    """
    ideal_category = golden.get("intent", golden.get("ideal_category", ""))
    ideal_response = golden.get("expected_answer", golden.get("ideal_response", ""))
    expected_escalation = golden.get("expected_escalation", False)
    required_facts = golden.get("required_facts", [])

    # 1. Category fit
    cat_matched = (predicted_category == ideal_category)
    category_fit = 5.0 if cat_matched else 1.0

    # 2. Semantic overlap / completeness
    sim = compute_semantic_similarity(agent_response, ideal_response)
    completeness = round(min(5.0, max(1.0, 1.0 + sim * 8.0)), 1)
    relevance = round(min(5.0, max(1.0, 2.0 + sim * 6.0)), 1)
    groundedness = 5.0 if len(agent_response) >= 20 else 2.0

    # 3. Correctness & Fact Coverage (Bug-fixed: inspect actual missing facts)
    missing_facts = [f for f in required_facts if f.lower() not in agent_response.lower()]
    if not missing_facts and cat_matched:
        correctness = 5.0
    elif cat_matched and len(missing_facts) <= 1:
        correctness = 4.0
    elif cat_matched:
        correctness = 3.0
    else:
        correctness = 1.0

    # 4. Helpfulness
    helpfulness = round((correctness + completeness + relevance) / 3.0, 1)

    # 5. Escalation appropriateness
    if needs_human == expected_escalation:
        escalation_score = 5.0
    else:
        escalation_score = 1.0

    # 6. Overall average score
    overall = round((correctness + relevance + completeness + groundedness + helpfulness + escalation_score) / 6.0, 2)

    # Determine failure type
    failure_type = "NONE"
    if not cat_matched:
        failure_type = "WRONG_INTENT"
    elif needs_human != expected_escalation:
        failure_type = "UNNECESSARY_ESCALATION" if needs_human else "INCORRECT_ESCALATION"
    elif missing_facts:
        failure_type = "INCOMPLETE_ANSWER"

    reason = (
        f"[heuristic-fixed] cat_match={cat_matched}, sim={sim:.2f}, "
        f"missing_facts={len(missing_facts)}, escalation_ok={needs_human == expected_escalation}"
    )

    return JudgeResult(
        ticket_id=ticket_id,
        overall_score=overall,
        correctness=correctness,
        relevance=relevance,
        completeness=completeness,
        groundedness=groundedness,
        helpfulness=helpfulness,
        escalation=escalation_score,
        reason=reason,
        failure_type=failure_type,
        judge_mode="heuristic",
    )


def evaluate_llm_judge(agent_output: Dict[str, Any], golden_record: Dict[str, Any]) -> JudgeResult:
    ticket_id = golden_record.get("id", "")
    query = agent_output.get("query", "")
    predicted_category = agent_output.get("intent", "")
    agent_response = agent_output.get("response", "")
    needs_human = agent_output.get("needs_human", False)
    retrieved_context = str(agent_output.get("retrieved_context", []))

    subject = golden_record.get("subject", query[:50])
    body = golden_record.get("body", query)
    ideal_response = golden_record.get("expected_answer", golden_record.get("ideal_response", ""))

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        subject=subject,
        body=body,
        predicted_category=predicted_category,
        agent_response=agent_response,
        ideal_response=ideal_response,
        retrieved_context=retrieved_context,
    )

    raw_resp = _call_llm_api(prompt)
    if raw_resp:
        try:
            # Extract JSON from code fence or raw string
            json_str = raw_resp
            if "```json" in raw_resp:
                json_str = raw_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_resp:
                json_str = raw_resp.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)
            return JudgeResult(
                ticket_id=ticket_id,
                overall_score=float(parsed.get("overall_score", 4.0)),
                correctness=float(parsed.get("correctness", 4.0)),
                relevance=float(parsed.get("relevance", 4.0)),
                completeness=float(parsed.get("completeness", 4.0)),
                groundedness=float(parsed.get("groundedness", 4.0)),
                helpfulness=float(parsed.get("helpfulness", 4.0)),
                escalation=float(parsed.get("escalation", 4.0)),
                reason=str(parsed.get("reason", "LLM Judge evaluation completed.")),
                failure_type=str(parsed.get("failure_type", "NONE")),
                judge_mode="llm",
            )
        except Exception:
            pass

    # Fallback to bug-fixed heuristic judge
    return _heuristic_judge_fixed(ticket_id, query, predicted_category, agent_response, golden_record, needs_human)
