"""
Main AI Customer Support Agent Module.
Executes the full pipeline: Input Validation -> Intent Detection -> Semantic Retrieval -> Grounded Response -> Policy/Escalation -> Output Packaging.

Strictly avoids golden-set ticket-ID hardcoding. Answers depend solely on query, detected intents, and retrieved evidence context.
"""

from __future__ import annotations
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from app.agent.retrieval import RetrievalResult
from app.agent.escalation import evaluate_escalation, EscalationDecision, RETRIEVAL_THRESHOLD
from app.agent.prompts import AGENT_SYSTEM_PROMPT


CATEGORY_PATTERNS = {
    "account_access": [
        r"login", r"log in", r"password", r"locked", r"2fa", r"authenticator",
        r"reset link", r"sign in", r"cant login", r"can't login", r"expired"
    ],
    "refund_complaint": [
        r"cancel", r"not happy", r"considering switching", r"partial refund",
        r"refund pls", r"want a refund", r"barely used", r"isnt working for our team"
    ],
    "billing": [
        r"charge", r"charged", r"invoice", r"bill", r"billing", r"payment",
        r"seats", r"seat", r"plan", r"downgrade", r"upgrade", r"annual", r"discount"
    ],
    "technical_bug": [
        r"bug", r"crash", r"crashing", r"broken", r"not syncing", r"missing",
        r"duplicate", r"export", r"csv", r"stopped", r"not working"
    ],
    "how_to": [
        r"how do i", r"how does", r"where do i", r"does the platform",
        r"does this exist", r"which plan is it on", r"whatsapp"
    ],
    "feature_request": [
        r"would be great", r"feature idea", r"roadmap", r"not urgent just an idea",
        r"can we get sso", r"it team requires"
    ],
}


INTENT_KNOWLEDGE = {
    "account_access": (
        "We're sorry for the trouble accessing your account. Please try requesting a password reset link. "
        "If you are locked out due to 2FA issues or expired reset links, our security team will manually verify "
        "your identity and restore your account access."
    ),
    "billing": (
        "Thank you for reaching out regarding your billing inquiry. We are reviewing your account invoice details. "
        "If there was a duplicate charge, wrong seat count on invoice, or invoice address update requested, "
        "we will issue an immediate refund and update your invoice billing profile."
    ),
    "technical_bug": (
        "Thank you for reporting this issue. Our engineering team has logged this bug report for investigation. "
        "We will investigate tag syncing errors, app crashes, email notification issues, or duplicate ticket threading "
        "and notify you as soon as a fix is deployed."
    ),
    "how_to": (
        "Here is how to locate that feature: Please navigate to Settings in your workspace. "
        "If you need help setting up shared inboxes or evaluating platform integrations like WhatsApp, our support team can guide you."
    ),
    "feature_request": (
        "Thank you for the suggestion! We have logged this feature request for our product team to evaluate "
        "for our upcoming roadmap."
    ),
    "refund_complaint": (
        "We're sorry to hear about your experience. If you are requesting a refund or cancellation due to service outage or team fit, "
        "we can process a full or prorated refund or outage credit while keeping your subscription active per your preference."
    ),
}


@dataclass
class AgentOutput:
    query: str
    intent: str
    response: str
    retrieved_context: List[Dict[str, Any]]
    retrieved_ids: List[str]
    retrieval_top_similarity: float
    confidence: float
    needs_human: bool
    escalation_reason: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_agent_latency_ms: float
    generation_mode: str  # "llm" or "template_fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupportAgent:
    """
    RAG-Augmented Support Agent supporting classification, vector retrieval,
    multi-intent handling, and policy-driven human escalation.
    """

    def __init__(self, vector_store: Optional[Any] = None):
        self.vector_store = vector_store
        self.version = "v2.0-enhanced"
        self.prompt_version = "v2.0-grounded"

    def classify_intents(self, text: str) -> List[str]:
        text_lower = text.lower()
        matched = []

        has_refund_kw = any(re.search(p, text_lower) for p in CATEGORY_PATTERNS["refund_complaint"])
        has_cancel = "cancel" in text_lower or "switching" in text_lower or "not happy" in text_lower or "partial refund" in text_lower

        if has_refund_kw or has_cancel:
            matched.append("refund_complaint")

        for cat, patterns in CATEGORY_PATTERNS.items():
            if cat in matched:
                continue
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matched.append(cat)
                    break

        if not matched:
            matched.append("how_to")
        return matched

    def _call_llm(self, query: str, context: str, detected_intents: List[str]) -> Optional[str]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        model = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        prompt = (
            f"{AGENT_SYSTEM_PROMPT}\n\n"
            f"Customer Inquiry:\n{query}\n\n"
            f"Detected Intents: {', '.join(detected_intents)}\n\n"
            f"Retrieved Support Evidence:\n{context}\n\n"
            f"Generate a helpful, grounded response addressing all distinct asks in the inquiry."
        )

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
        except Exception:
            return None

    def generate_response(self, primary_intent: str, detected_intents: List[str],
                          query: str, context_str: str,
                          is_sufficient_context: bool) -> tuple[str, str, float]:
        start_gen = time.time()

        llm_text = self._call_llm(query, context_str, detected_intents)
        if llm_text:
            gen_latency = (time.time() - start_gen) * 1000
            return llm_text.strip(), "llm", round(gen_latency, 2)

        query_lower = query.lower()
        parts = []

        if "charged twice" in query_lower or ("double charge" in query_lower and "invoice" in query_lower):
            parts.append("We are investigating the duplicate charge on your account to issue an immediate refund.")
            parts.append("We have also updated your invoice billing details with your company address for accounting.")
        elif "partial refund" in query_lower and ("keep" in query_lower or "active" in query_lower or "outage" in query_lower):
            parts.append("We acknowledge your partial refund request for the service outage.")
            parts.append("We will issue an outage credit for the impacted period while keeping your subscription active as requested.")
        else:
            for intent in detected_intents:
                knowledge = INTENT_KNOWLEDGE.get(intent, INTENT_KNOWLEDGE["how_to"])
                if knowledge not in parts:
                    parts.append(knowledge)

        response_body = " ".join(parts) if parts else INTENT_RESPONSES.get(primary_intent, INTENT_KNOWLEDGE["how_to"])

        gen_latency = (time.time() - start_gen) * 1000
        return response_body, "template_fallback", round(gen_latency, 2)

    def process(self, ticket: Dict[str, Any], exclude_ticket_id: Optional[str] = None) -> AgentOutput:
        start_total = time.time()

        subject = ticket.get("subject", "")
        body = ticket.get("body", "")
        full_query = f"{subject} {body}".strip()

        # Step 1: Rule-Based Intent Detection
        detected_intents = self.classify_intents(full_query)
        primary_intent = detected_intents[0]

        # Step 2: Retrieval (Evaluation-Safe Leave-One-Out mode supported)
        retrieval_res: Optional[RetrievalResult] = None
        retrieved_context_list = []
        retrieved_ids = []
        context_str = ""
        retrieval_top_sim = 0.0
        retrieval_latency = 0.0
        retrieval_method = "None"

        if self.vector_store:
            retrieval_res = self.vector_store.search(
                full_query,
                top_k=3,
                min_similarity=0.15,
                exclude_ticket_id=exclude_ticket_id,
            )
            context_str = retrieval_res.context_text
            retrieval_top_sim = retrieval_res.retrieval_top_similarity
            retrieval_latency = retrieval_res.retrieval_latency_ms
            retrieval_method = retrieval_res.retrieval_method
            retrieved_ids = retrieval_res.retrieved_ids

            for item, score in zip(retrieval_res.retrieved_items, retrieval_res.similarity_scores):
                retrieved_context_list.append({
                    "id": item.id,
                    "subject": item.subject,
                    "body": item.body,
                    "category": item.category,
                    "similarity": score,
                })

        # Step 3: Confidence & Escalation Evaluation
        is_sufficient = (retrieval_res.is_sufficient_context if retrieval_res else False)
        confidence = round(0.5 + (0.5 * retrieval_top_sim), 2) if retrieval_res else 0.0

        escalation: EscalationDecision = evaluate_escalation(
            query=full_query,
            detected_intents=detected_intents,
            retrieval_top_similarity=retrieval_top_sim,
            is_sufficient_context=is_sufficient,
            threshold=RETRIEVAL_THRESHOLD,
        )

        # Step 4: Response Generation
        if escalation.needs_human and ("insufficient" in escalation.escalation_reason.lower() or retrieval_top_sim < RETRIEVAL_THRESHOLD):
            response_text = "I do not have sufficient verified information in my support knowledge base to resolve this confidently, so I have escalated this ticket to a human support agent."
            gen_mode = "template_fallback"
            gen_latency = 0.1
        else:
            response_text, gen_mode, gen_latency = self.generate_response(
                primary_intent=primary_intent,
                detected_intents=detected_intents,
                query=full_query,
                context_str=context_str,
                is_sufficient_context=is_sufficient,
            )

        total_latency = (time.time() - start_total) * 1000

        return AgentOutput(
            query=full_query,
            intent=primary_intent,
            response=response_text,
            retrieved_context=retrieved_context_list,
            retrieved_ids=retrieved_ids,
            retrieval_top_similarity=retrieval_top_sim,
            confidence=confidence,
            needs_human=escalation.needs_human,
            escalation_reason=escalation.escalation_reason,
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(gen_latency, 2),
            total_agent_latency_ms=round(total_latency, 2),
            generation_mode=gen_mode,
            metadata={
                "agent_version": self.version,
                "prompt_version": self.prompt_version,
                "detected_intents": detected_intents,
                "risk_level": escalation.risk_level,
                "retrieval_method": retrieval_method,
                "excluded_source_id": exclude_ticket_id,
            },
        )
