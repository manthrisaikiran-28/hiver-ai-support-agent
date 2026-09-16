"""
Unit tests for Semantic Vector Retrieval.
"""

import pytest
from app.data.loader import DataLoader
from app.agent.retrieval import build_knowledge_base_from_tickets


def test_vector_search():
    tickets = DataLoader().load_jsonl("data/raw/tickets.jsonl")
    kb = build_knowledge_base_from_tickets(tickets)
    
    result = kb.search("cant login password reset", top_k=3)
    assert len(result.retrieved_items) > 0
    assert result.is_sufficient_context is True
    assert result.retrieved_items[0].category == "account_access"
