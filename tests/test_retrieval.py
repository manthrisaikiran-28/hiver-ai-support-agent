"""
Unit tests for Dual Vector Retrieval & Leave-One-Out Evaluation Mode.
"""

import pytest
from app.data.loader import DataLoader
from app.agent.retrieval import build_knowledge_base_from_tickets, TFIDFVectorStore, SentenceTransformerVectorStore


def test_tfidf_retrieval_and_leave_one_out():
    tickets = DataLoader().load_jsonl("data/raw/tickets.jsonl")
    kb = build_knowledge_base_from_tickets(tickets, use_sentence_transformers=False)

    # Search with leave-one-out source ticket exclusion
    res = kb.search("cant login password reset", top_k=3, exclude_ticket_id="t001")
    assert "t001" not in res.retrieved_ids, "Leave-one-out failure: source ticket t001 was retrieved!"
    assert res.excluded_source_id == "t001"
    assert res.retrieval_method == "TF-IDF Baseline"


def test_sentence_transformer_retrieval():
    tickets = DataLoader().load_jsonl("data/raw/tickets.jsonl")
    kb = build_knowledge_base_from_tickets(tickets, use_sentence_transformers=True)

    res = kb.search("cant login password reset", top_k=3, exclude_ticket_id="t001")
    assert "t001" not in res.retrieved_ids
    assert res.retrieval_top_similarity > 0.0
