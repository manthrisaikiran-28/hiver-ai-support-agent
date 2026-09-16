# Hiver SDE Intern — Mini Project Final Technical Report

## 1. Executive Summary

This project delivers an end-to-end AI Customer Support Agent and Evaluation Platform designed for the Hiver SDE Intern technical assessment. The system includes data ingestion with PII redaction, multi-intent classification, vector-based semantic retrieval (RAG), policy-driven human escalation, a 6-dimension LLM-as-a-Judge, and a comprehensive failure analysis framework.

### Key Measured Performance Metrics:
- **Category Classification Accuracy**: **90.9%**
- **Average Overall Rubric Score**: **3.55 / 5.0**
- **Groundedness Score**: **5.0 / 5.0**
- **Average Agent Latency**: **0.16 ms**
- **Human vs Judge Agreement**: **100.0%**

---

## 2. Dataset Understanding & Cleaning

The dataset consists of **24 synthetic customer support tickets** (`data/raw/tickets.jsonl`) modeled after real-world messy SaaS support interactions (typos, run-on sentences, bundled requests, ambiguous subjects).

### Profiling Breakdown:
- **Total Records**: 24
- **Unique IDs**: 24 (0 duplicates)
- **Category Distribution**:
  - `technical_bug`: 7 (29.2%)
  - `billing`: 5 (20.8%)
  - `how_to`: 4 (16.7%)
  - `account_access`: 3 (12.5%)
  - `refund_complaint`: 3 (12.5%)
  - `feature_request`: 2 (8.3%)

### Cleaning & PII Masking:
The preprocessing pipeline (`app/data/preprocessing.py`) applies regex-based redaction for emails, phone numbers, and credit cards before indexing or passing queries to external models.

---

## 3. Agent & Retrieval Architecture

The support agent implements a 5-stage pipeline:

1. **Input Validation & PII Masking**: Sanitizes query text.
2. **Multi-Intent Pattern Engine**: Detects primary and secondary intents.
3. **TF-IDF / Vector Knowledge Retrieval**: Embeds tickets and retrieves top-k relevant knowledge base context.
4. **Grounded Generator**: Generates concise resolution text grounded strictly in retrieved evidence.
5. **Policy & Escalation Engine**: Evaluates risk conditions (data loss, 2FA lockout, compliance, high churn risk) and assigns human escalation flags.

---

## 4. Evaluation Methodology

Evaluation is conducted against a versioned golden ground-truth set (`data/golden/golden_v1.jsonl`, 11 representative cases across Normal, Hard, Edge, and Escalation categories).

### Evaluation Signals:
1. **Deterministic Checks**: Validates required resolution facts, absence of forbidden claims, intent match, and escalation match.
2. **Semantic Similarity**: Computes cosine similarity between agent output and golden reference answers.
3. **LLM-as-a-Judge (6 Rubric Dimensions)**: Evaluates Correctness, Relevance, Completeness, Groundedness, Helpfulness, and Escalation Appropriateness.

---

## 5. Failure Analysis & Misleading Headline Metrics

A core contribution of this project is demonstrating **why headline evaluation metrics can be misleading**:

1. **Category Recall Disparity**: The baseline agent scored 80% accuracy overall, but had **0% recall on `refund_complaint`** due to keyword precedence ordering (`billing` checked before `refund_complaint`). Our enhanced agent resolves this priority bug, raising accuracy to 90.9% while fixing churn-risk classification.
2. **Heuristic Judge Artifacts**: Flawed heuristic evaluation rules that penalize text based on surface words (such as searching for `"and"` in ideal answers) corrupt correctness scores to 1.0 across all cases. Our bug-fixed judge inspects actual required fact presence.
3. **Multi-Intent Masking**: High semantic similarity scores often conceal missing multi-intent resolutions (e.g. addressing a duplicate charge while omitting an invoice address change in `t002`).

---

## 6. Conclusion & Recommendations

The evaluation platform proves that high aggregate accuracy numbers must always be accompanied by per-category breakdowns, multi-intent verification, and qualitative failure classification to ensure reliable customer support automation.
