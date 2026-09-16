"""
LLM-as-judge evaluator, with a heuristic fallback.

Real usage: set ANTHROPIC_API_KEY and this will call Claude to score the
agent's response against the golden/ideal response on a 1-5 rubric.

Fallback (no API key / no network, e.g. this sandbox): a cheap heuristic
"judge" based on category match + lexical overlap. This is clearly labeled
as a WEAKER, biased stand-in -- see failure_analysis.md for why treating
its output as equivalent to a real LLM judge would be misleading.
"""

from __future__ import annotations
import json
import os
import re
import urllib.request
from dataclasses import dataclass

JUDGE_PROMPT_TEMPLATE = """You are grading a customer support agent's response.

Customer ticket:
Subject: {subject}
Body: {body}

The agent classified this ticket as: {predicted_category}
The agent's response was:
"{agent_response}"

Here is what an ideal response should cover (ground truth, written by a human reviewer):
"{ideal_response}"

Score the agent's response from 1-5 on:
- correctness: does it address the actual problem, including ALL distinct asks in the ticket (not just one)?
- completeness: does it match the substance of the ideal response, or is it a generic template that happens to sound plausible?
- category_fit: is "{predicted_category}" a reasonable classification given the ideal category is intended to be inferred from context (do not just check surface keyword overlap)?

Respond ONLY with JSON: {{"correctness": <1-5>, "completeness": <1-5>, "category_fit": <1-5>, "reasoning": "<1-2 sentences>"}}
"""


@dataclass
class JudgeResult:
    ticket_id: str
    correctness: float
    completeness: float
    category_fit: float
    reasoning: str
    judge_mode: str  # "llm" or "heuristic"

    @property
    def overall(self) -> float:
        return round((self.correctness + self.completeness + self.category_fit) / 3, 2)


def _call_anthropic(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("no ANTHROPIC_API_KEY set")
    body = json.dumps({
        "model": model,
        "max_tokens": 300,
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "".join(block.get("text", "") for block in data.get("content", []))


def _heuristic_score(ticket: dict, predicted_category: str, agent_response: str,
                      golden: dict) -> JudgeResult:
    """
    Cheap stand-in judge: NOT an LLM. Scores category match as binary, and
    completeness via crude word overlap between agent response and the
    ideal response. Deliberately simple so its blind spots are visible.
    """
    category_fit = 5.0 if predicted_category == golden["ideal_category"] else 1.0

    def words(s: str) -> set:
        return set(re.findall(r"[a-z']+", s.lower()))

    agent_w = words(agent_response)
    ideal_w = words(golden["ideal_response"])
    overlap = len(agent_w & ideal_w) / max(1, len(ideal_w))
    completeness = round(1 + overlap * 4, 2)  # map [0,1] -> [1,5]

    # correctness heuristic: penalize if the ideal response implies multiple
    # distinct asks (very rough proxy: ideal response contains "AND" or "both")
    multi_intent = " and " in golden["ideal_response"].lower() or "both" in golden["ideal_response"].lower()
    correctness = completeness
    if multi_intent:
        correctness = max(1.0, completeness - 2.0)  # template agent almost never covers both asks

    reasoning = (
        f"[heuristic] category_match={predicted_category == golden['ideal_category']}, "
        f"lexical_overlap={overlap:.2f}, multi_intent_ticket={multi_intent}"
    )
    return JudgeResult(
        ticket_id=ticket["id"],
        correctness=correctness,
        completeness=completeness,
        category_fit=category_fit,
        reasoning=reasoning,
        judge_mode="heuristic",
    )


def judge(ticket: dict, predicted_category: str, agent_response: str,
          golden: dict) -> JudgeResult:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        subject=ticket.get("subject", ""),
        body=ticket.get("body", ""),
        predicted_category=predicted_category,
        agent_response=agent_response,
        ideal_response=golden["ideal_response"],
    )
    try:
        raw = _call_anthropic(prompt)
        parsed = json.loads(raw)
        return JudgeResult(
            ticket_id=ticket["id"],
            correctness=float(parsed["correctness"]),
            completeness=float(parsed["completeness"]),
            category_fit=float(parsed["category_fit"]),
            reasoning=parsed.get("reasoning", ""),
            judge_mode="llm",
        )
    except Exception:
        return _heuristic_score(ticket, predicted_category, agent_response, golden)
