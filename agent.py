"""
A small, deliberately simple AI support agent.

Pipeline: classify(ticket) -> respond(ticket, category)

This is intentionally a lightweight keyword/rule-based baseline rather than
a full LLM-backed agent, so that:
  1) it runs anywhere with zero dependencies / API keys / network access
  2) its failure modes are easy to reason about for the failure-analysis
     part of the exercise (a black-box LLM agent would make it harder to
     tell "the agent is wrong" from "the judge is wrong")

Swap `classify`/`respond` for real LLM calls (see llm_judge.py for the
pattern) once you want a stronger baseline to evaluate.
"""

from __future__ import annotations
import re
from dataclasses import dataclass

CATEGORY_KEYWORDS = {
    "billing": [
        "charge", "charged", "invoice", "refund", "bill", "billing",
        "payment", "seats", "seat", "plan", "downgrade", "upgrade",
        "annual", "discount", "cycle", "subscription",
    ],
    "account_access": [
        "login", "log in", "password", "locked", "2fa", "authenticator",
        "reset link", "sign in", "access restored", "cant login", "can't login",
    ],
    "technical_bug": [
        "bug", "crash", "crashing", "broken", "not syncing", "isnt there",
        "isn't there", "missing", "duplicate", "export", "csv", "stopped",
        "not working", "doesnt fix", "doesn't fix",
    ],
    "how_to": [
        "how do i", "how does", "where do i", "does the platform",
        "does this exist", "which plan is it on", "support whatsapp",
    ],
    "feature_request": [
        "would be great", "feature idea", "roadmap", "not urgent just an idea",
        "our it team requires", "can we get sso",
    ],
    "refund_complaint": [
        "not happy", "considering switching", "cancel", "partial refund",
        "barely used", "isnt working for our team",
    ],
}

# NOTE (intentional weakness, kept for failure analysis):
# categories are checked in this fixed order and we return on the FIRST
# match. Real tickets often contain multiple intents (see t002, t023) --
# this agent will silently only address whichever category's keywords
# happen to appear, or get matched, first.
CATEGORY_ORDER = [
    "account_access", "billing", "technical_bug", "how_to",
    "feature_request", "refund_complaint",
]

RESPONSE_TEMPLATES = {
    "account_access": (
        "Sorry you're having trouble accessing your account. We've flagged this "
        "as a login issue -- please try resetting your password, and let us "
        "know if you're still locked out so we can manually restore access."
    ),
    "billing": (
        "Thanks for flagging this billing question. We'll review your account "
        "and invoice details and follow up with a correction or refund if one "
        "is owed."
    ),
    "technical_bug": (
        "Thanks for the report -- this looks like a bug on our end. We're "
        "logging it for engineering to investigate and will update you once "
        "we know more."
    ),
    "how_to": (
        "Great question -- here's how that works on our platform: please check "
        "Settings for the relevant option, and let us know if you can't find it "
        "and we'll walk you through it step by step."
    ),
    "feature_request": (
        "Thanks for the suggestion! We've logged this as a feature request for "
        "our product team to consider for the roadmap."
    ),
    "refund_complaint": (
        "We're sorry to hear this hasn't met expectations. We'd like to "
        "understand what didn't work and see if there's a way to fix it before "
        "you go, but we can also process a refund/cancellation if you'd prefer."
    ),
}


@dataclass
class AgentResult:
    ticket_id: str
    predicted_category: str
    response: str


def classify(text: str) -> str:
    text_l = text.lower()
    for category in CATEGORY_ORDER:
        for kw in CATEGORY_KEYWORDS[category]:
            if kw in text_l:
                return category
    return "how_to"  # default fallback bucket


def respond(category: str) -> str:
    return RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["how_to"])


def run_agent(ticket: dict) -> AgentResult:
    full_text = f"{ticket.get('subject', '')} {ticket.get('body', '')}"
    category = classify(full_text)
    response = respond(category)
    return AgentResult(ticket_id=ticket["id"], predicted_category=category, response=response)
