"""
Unit tests for Data Ingestion and Preprocessing.
"""

import pytest
from pathlib import Path
from app.data.loader import DataLoader
from app.data.preprocessing import clean_text, mask_pii, preprocess_ticket


def test_loader_loads_raw_tickets():
    loader = DataLoader()
    raw_path = Path("data/raw/tickets.jsonl")
    assert raw_path.exists(), "raw tickets.jsonl missing"
    tickets = loader.load_jsonl(raw_path)
    assert len(tickets) == 24
    assert tickets[0]["id"] == "t001"
    assert "subject" in tickets[0]
    assert "body" in tickets[0]


def test_pii_masking():
    sample_text = "Contact me at user@example.com or call 555-123-4567"
    masked = mask_pii(sample_text)
    assert "[EMAIL_REDACTED]" in masked
    assert "[PHONE_REDACTED]" in masked
    assert "user@example.com" not in masked
