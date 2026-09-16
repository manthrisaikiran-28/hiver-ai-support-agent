"""
Unit tests for AI Support Agent, RAG Evidence Response, & Escalation Rules.
"""

import pytest
from app.data.loader import DataLoader
from app.agent.retrieval import build_knowledge_base_from_tickets
from app.agent.agent import SupportAgent


def test_no_ticket_id_hardcoding():
    """Verify agent response generation does NOT depend on golden ticket IDs."""
    tickets = DataLoader().load_jsonl("data/raw/tickets.jsonl")
    kb = build_knowledge_base_from_tickets(tickets)
    agent = SupportAgent(vector_store=kb)

    # Pass dummy ticket ID to ensure answer depends only on query text
    dummy_ticket = {"id": "dummy_999", "subject": "refund pls", "body": "we decided to cancel product not working"}
    out = agent.process(dummy_ticket)
    assert out.intent == "refund_complaint"
    assert out.needs_human is True
    assert "refund" in out.response.lower() or "cancellation" in out.response.lower()


def test_multi_intent_handling():
    """Verify multi-intent queries generate responses covering all detected intents."""
    tickets = DataLoader().load_jsonl("data/raw/tickets.jsonl")
    kb = build_knowledge_base_from_tickets(tickets)
    agent = SupportAgent(vector_store=kb)

    # Query with duplicate charge + invoice address update
    multi_ticket = {
        "id": "dummy_002",
        "subject": "billing question",
        "body": "hey noticed we got charged twice this month? also need invoice with company address on it"
    }
    out = agent.process(multi_ticket)
    assert "duplicate charge" in out.response.lower()
    assert "company address" in out.response.lower()


def test_low_confidence_retrieval_escalation():
    """Verify queries with low retrieval similarity trigger human escalation."""
    tickets = DataLoader().load_jsonl("data/raw/tickets.jsonl")
    kb = build_knowledge_base_from_tickets(tickets)
    agent = SupportAgent(vector_store=kb)

    # Low relevance query with no matching keywords in KB
    low_ticket = {
        "id": "dummy_low",
        "subject": "xyz random text 999",
        "body": "completely unrelated random string zzz"
    }
    out = agent.process(low_ticket)
    assert out.needs_human is True
    assert "insufficient" in out.escalation_reason.lower() or "below threshold" in out.escalation_reason.lower()
