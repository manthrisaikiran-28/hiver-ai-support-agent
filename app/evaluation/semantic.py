"""
Semantic Similarity Evaluation Module.
Computes cosine similarity between generated agent responses and reference ideal answers.
"""

from __future__ import annotations
import math
import re
from typing import Dict, Any


def _tokenize_to_tf(text: str) -> Dict[str, float]:
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    if not tokens:
        return {}
    total = len(tokens)
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return {t: count / total for t, count in tf.items()}


def compute_semantic_similarity(candidate: str, reference: str) -> float:
    """
    Compute Cosine Similarity between candidate text and reference text.
    Returns a score between 0.0 and 1.0.
    """
    if not candidate or not reference:
        return 0.0

    tf1 = _tokenize_to_tf(candidate)
    tf2 = _tokenize_to_tf(reference)

    all_tokens = set(tf1.keys()) | set(tf2.keys())
    if not all_tokens:
        return 0.0

    dot_product = sum(tf1.get(t, 0.0) * tf2.get(t, 0.0) for t in all_tokens)
    mag1 = math.sqrt(sum(v * v for v in tf1.values()))
    mag2 = math.sqrt(sum(v * v for v in tf2.values()))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    return round(dot_product / (mag1 * mag2), 4)


def evaluate_semantic(agent_output: Dict[str, Any], golden_record: Dict[str, Any]) -> Dict[str, Any]:
    candidate_resp = agent_output.get("response", "")
    expected_resp = golden_record.get("expected_answer", golden_record.get("ideal_response", ""))

    sim_score = compute_semantic_similarity(candidate_resp, expected_resp)
    
    return {
        "semantic_similarity": sim_score,
        "note": "Semantic similarity measures vocabulary/topical alignment, NOT absolute factual correctness.",
    }
