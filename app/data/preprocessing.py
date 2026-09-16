"""
Data Preprocessing & PII Masking utilities for Customer Support Datasets.
"""

from __future__ import annotations
import re
from typing import Dict, List, Any


# Regular expressions for PII detection/masking
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mask_pii(text: str) -> str:
    """Mask email addresses, phone numbers, and credit card patterns."""
    if not text:
        return ""
    text = EMAIL_REGEX.sub("[EMAIL_REDACTED]", text)
    text = PHONE_REGEX.sub("[PHONE_REDACTED]", text)
    text = CARD_REGEX.sub("[CARD_REDACTED]", text)
    return text


def preprocess_ticket(ticket: Dict[str, Any], mask_sensitive: bool = True) -> Dict[str, Any]:
    """Preprocess subject and body, applying text cleaning and PII masking."""
    cleaned_subject = clean_text(ticket.get("subject", ""))
    cleaned_body = clean_text(ticket.get("body", ""))

    if mask_sensitive:
        cleaned_subject = mask_pii(cleaned_subject)
        cleaned_body = mask_pii(cleaned_body)

    processed = dict(ticket)
    processed["subject"] = cleaned_subject
    processed["body"] = cleaned_body
    processed["full_text"] = f"{cleaned_subject} {cleaned_body}".strip()
    return processed


def preprocess_dataset(tickets: List[Dict[str, Any]], mask_sensitive: bool = True) -> List[Dict[str, Any]]:
    return [preprocess_ticket(t, mask_sensitive=mask_sensitive) for t in tickets]
