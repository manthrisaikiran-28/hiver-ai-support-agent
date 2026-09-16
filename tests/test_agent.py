"""
Unit tests for AI Support Agent & Policy Escalation.
"""

import pytest
from app.agent.agent import SupportAgent


def test_agent_intent_classification():
    agent = SupportAgent()
    
    # Test refund complaint classification (fixing baseline ordering bug)
    t005 = {"id": "t005", "subject": "refund pls", "body": "we decided to cancel, product not working..."}
    out_t005 = agent.process(t005)
    assert out_t005.intent == "refund_complaint"
    assert out_t005.needs_human is True


def test_agent_multi_intent_handling():
    agent = SupportAgent()
    
    # Test t002 multi-intent ticket (duplicate charge + invoice address)
    t002 = {"id": "t002", "subject": "billing question", "body": "charged twice this month... also invoice with company address"}
    out_t002 = agent.process(t002)
    assert "duplicate charge" in out_t002.response.lower()
    assert "company address" in out_t002.response.lower()


def test_agent_escalation_rules():
    agent = SupportAgent()
    
    # Test data loss ticket escalation
    t007 = {"id": "t007", "subject": "urgent - data missing", "body": "half our conversation history from last week is just gone. compliance reasons."}
    out_t007 = agent.process(t007)
    assert out_t007.needs_human is True
    assert "data loss" in out_t007.escalation_reason.lower() or "compliance" in out_t007.escalation_reason.lower()
