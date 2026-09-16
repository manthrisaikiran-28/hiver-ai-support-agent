# Failure Analysis & Metric Integrity Critique

## 1. Headline Accuracy vs. Per-Category Recall

A flat headline accuracy number (e.g. 80% or 90%) creates a false sense of reliability.

In the baseline agent:
- Aggregate accuracy was **80%**.
- However, recall on `refund_complaint` (churn risk) was **0%** (0 out of 2 test cases correct).
- **Why?** Keywords like `"refund"` were listed under `billing`, and `billing` was evaluated before `refund_complaint` in the hardcoded search order (`account_access` -> `billing` -> `technical_bug` -> `how_to` -> `feature_request` -> `refund_complaint`).
- **Impact**: The single category most vital to business retention was completely broken, yet masked by the headline number.

Our enhanced agent in `app/agent/agent.py` fixes this priority ordering and regex matching, achieving **100% recall on refund/churn risk tickets** (`t005`, `t023`).

---

## 2. Measurement Artifacts in Naive Heuristics

The initial heuristic evaluator in `llm_judge.py` penalized ticket correctness by `-2.0` if the **ideal reference answer** contained the words `"and"` or `"both"`. 
- Because human reference answers are written in natural prose, 10 out of 10 golden set reference answers contained `"and"`.
- This resulted in a flat **1.0 / 5.0 correctness score** across all 10 tickets in baseline runs.
- **Takeaway**: Synthetic evaluation heuristics must be audited for measurement artifacts before trusting automated evaluation outputs.

Our bug-fixed heuristic judge in `app/evaluation/llm_judge.py` replaces this rule with explicit required-fact coverage verification.

---

## 3. Lexical Overlap vs. Multi-Intent Completeness

Semantic similarity metrics (cosine similarity of embeddings or TF-IDF) measure vocabulary overlap, NOT resolution completeness.

- In ticket `t002`, the customer asks two distinct questions:
  1. Investigate duplicate charge on the 3rd and 9th.
  2. Change invoice billing address to company address.
- A single-template agent response that only addresses the duplicate charge still achieves a high semantic similarity score (~0.75) because billing terms match.
- Our enhanced evaluation harness (`app/evaluation/deterministic.py`) enforces multi-intent required fact checks (`required_facts`), ensuring single-intent responses on multi-intent tickets trigger an `INCOMPLETE_ANSWER` failure flag.

---

## 4. Failure Mode Taxonomy

The evaluation platform categorizes all failures into 10 explicit failure types:

1. `RETRIEVAL_FAILURE`: Top retrieved document similarity below confidence threshold.
2. `GENERATION_FAILURE`: Model response fails quality rubric.
3. `HALLUCINATION`: Claims present in response not supported by retrieved context.
4. `MISSING_CONTEXT`: Knowledge base lacks necessary information to answer query.
5. `WRONG_INTENT`: Intent misclassified by agent.
6. `INCOMPLETE_ANSWER`: Response omits required resolution facts on multi-intent queries.
7. `INCORRECT_ESCALATION`: Agent fails to escalate high-risk / compliance / 2FA lockout tickets.
8. `UNNECESSARY_ESCALATION`: Agent escalates standard low-risk query unnecessarily.
9. `JUDGE_DISAGREEMENT`: LLM judge score deviates significantly from human ground truth.
10. `DATA_QUALITY_ISSUE`: Ambiguity or error in original ticket text.
