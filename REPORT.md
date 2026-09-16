# Hiver SDE Intern — Technical Evaluation & Architecture Report

## 1. Executive Summary

This report presents an end-to-end **AI Customer Support Agent and Evaluation Platform** developed for the Hiver SDE Intern Technical Evaluation. The system is designed as an evidence-first customer support pipeline that ingests raw support tickets, redacts PII, classifies multi-intent customer queries, retrieves relevant knowledge base articles, generates grounded resolution responses, and applies automated human escalation policies when confidence or safety thresholds are not met.

Rather than optimizing solely for high headline accuracy metrics, this project emphasizes **rigorous, honest, and reproducible evaluation**. The system incorporates leave-one-out RAG retrieval to prevent evaluation leakage, dual vector retrieval engines (TF-IDF vs. SentenceTransformers), deterministic rule-based checks, a 6-dimension evaluation rubric, and an empirical failure analysis framework.

---

## 2. Problem Statement

Automated AI support agents face critical operational challenges in enterprise SaaS environments:

1. **Category Ambiguity & Preemption**: Customers frequently send ambiguous, unpunctuated, or multi-part requests where keyword overlap causes critical categories (e.g. churn-risk refund requests) to be misrouted.
2. **Hallucination & Un-grounded Claims**: Generative models may invent policies or issue unauthorized guarantees without factual backing from knowledge base documents.
3. **Data Leakage in Evaluation**: Standard RAG evaluation harnesses often allow an evaluation test case to retrieve its own source document from the vector corpus, inflating similarity and accuracy scores.
4. **Metric Distortions**: Headline metrics such as global category accuracy or un-calibrated semantic similarity can mask 0% recall on high-priority customer categories.

---

## 3. System Architecture

The support platform follows a modular 5-stage architecture:

```text
[ Raw Ticket Input ]
        │
        ▼
[ 1. Ingestion & PII Redaction ] (Regex masking of emails, phone numbers, credit cards)
        │
        ▼
[ 2. Multi-Intent Engine ] (Primary & secondary intent classification)
        │
        ▼
[ 3. RAG Retrieval Engine ] (Version A: TF-IDF | Version B: SentenceTransformers)
        │
        ▼
[ 4. Grounded Generator ] (Context-grounded resolution synthesis)
        │
        ▼
[ 5. Safety & Escalation Engine ] (Risk keyword detection & low-confidence routing)
        │
        ▼
[ Structured Agent Output ] ──► [ Multi-Signal Evaluation Harness ]
```

---

## 4. Agent Workflow

1. **Query Preprocessing**: Customer queries pass through `app/data/preprocessing.py` to sanitize whitespace and redact sensitive PII (emails, phone numbers, payment details).
2. **Intent Detection**: The query is evaluated against multi-intent pattern definitions in `app/agent/agent.py` to identify primary customer intent (`billing`, `technical_bug`, `refund_complaint`, `account_access`, `how_to`) and any secondary bundled requests.
3. **Context Retrieval**: The retriever queries the knowledge corpus for top-$k$ relevant context snippets ($k=3$). During evaluation, the retriever excludes the ticket being evaluated.
4. **Response Synthesis**: The generator builds a structured response constrained strictly by the retrieved knowledge snippets.
5. **Safety Check & Escalation**: The response is evaluated against risk policies (data loss, 2FA lockout, compliance, churn risk) and vector confidence thresholds (`RETRIEVAL_THRESHOLD = 0.25`). If triggered, the agent flags the ticket for human escalation.

---

## 5. Intent Classification Engine

The intent classification engine combines keyword weighting with multi-intent pattern detection to prevent category preemption.

- **Primary Categories**: `technical_bug`, `billing`, `account_access`, `refund_complaint`, `how_to`, `feature_request`.
- **Multi-Intent Handling**: When a ticket combines multiple requests (e.g. `t002`: duplicate charge investigation + invoice address change), the classifier extracts secondary intent flags so both issues are addressed.

---

## 6. Knowledge Retrieval Architecture

The system implements two distinct retrieval strategies to evaluate the impact of semantic search on RAG performance:

- **Version A — Lexical Baseline (TF-IDF)**: Computes term frequency-inverse document frequency vectors over character and word n-grams using `scikit-learn`.
- **Version B — Dense Embeddings (SentenceTransformers `all-MiniLM-L6-v2`)**: Generates 384-dimensional dense vector embeddings to capture semantic similarity beyond literal keyword matches.

---

## 7. Leave-One-Out Evaluation (Data Leakage Prevention)

A critical flaw in standard RAG evaluation is **evaluation leakage**, where a golden test case retrieves its own source document from the knowledge base, yielding artificially inflated similarity scores ($>0.95$).

To eliminate this artifact:
- During evaluation of ticket `gid`, `retrieval.py` accepts an `exclude_ticket_id = gid` parameter.
- The retriever filters out document `gid` from candidate search results.
- This forces the agent to rely on general knowledge base articles or related tickets, reflecting real-world performance on unseen customer queries.

---

## 8. Policy & Safety Escalation Logic

The escalation engine (`app/agent/escalation.py`) enforces strict safety guardrails:

1. **Policy Risk Triggers**: Mandates human escalation for explicit risk scenarios:
   - Account lockout / 2FA reset failures.
   - Irreversible data deletion / loss requests.
   - Security breaches or compliance inquiries.
   - Severe churn risk associated with formal refund complaints.
2. **Low-Confidence Retrieval Escalation**: If the top retrieved vector similarity score falls below `0.25`, the agent automatically flags the ticket for human escalation due to insufficient knowledge base context.

---

## 9. Evaluation Methodology

Evaluation is executed against **11 golden ground-truth cases** (`data/golden/golden_v1.jsonl`) representing diverse ticket difficulties (Easy, Medium, Hard, Escalation).

The harness combines three evaluation signals:

1. **Deterministic Rule Checks (`app/evaluation/deterministic.py`)**:
   - `intent_pass`: True if predicted intent matches golden category.
   - `required_facts_pass`: True if all required resolution facts are present in the response.
   - `forbidden_claims_pass`: True if no forbidden false guarantees are present.
   - `escalation_pass`: True if human escalation decision matches golden expectation.
   - `deterministic_score`: Composite score from 0.0 to 1.0.
2. **Semantic Cosine Similarity (`app/evaluation/semantic.py`)**: Measures token-level cosine similarity between agent output and golden reference answers.
3. **Rubric Evaluation (`app/evaluation/llm_judge.py`)**: Evaluates responses across 6 dimensions on a 1-5 scale:
   - *Correctness*, *Relevance*, *Completeness*, *Groundedness*, *Helpfulness*, and *Escalation Appropriateness*.

---

## 10. Honest Evaluation Modes

- **Generation Mode**: `LLM Mode` (when Anthropic/OpenAI API keys are configured) or `template_fallback` (offline template renderer).
- **Judge Mode**: `LLM Judge` (API-driven rubric prompt) or `heuristic_fallback` (offline rule-based scorer).
- *Offline heuristic evaluation is reported transparently as heuristic fallback and is never presented as LLM evaluation.*

---

## 11. Empirical Results & Version Comparison

The table below presents the reproducible metrics recorded from the latest evaluation execution:

| Metric | Version A (TF-IDF Baseline) | Version B (SentenceTransformers) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Category Classification Accuracy** | **90.9%** (10 / 11) | **90.9%** (10 / 11) | 0.0% |
| **Average Overall Rubric Score (1-5)** | **2.78 / 5.0** | **3.54 / 5.0** | **+0.76** |
| **Pass Rate (Overall Score $\ge$ 4.0)** | **9.1%** (1 / 11) | **27.3%** (3 / 11) | **+18.2%** |
| **Deterministic Pass Rate (Score $\ge$ 0.8)** | **36.4%** (4 / 11) | **90.9%** (10 / 11) | **+54.5%** |
| **Average Deterministic Score** | **0.65 / 1.0** | **0.80 / 1.0** | **+0.15** |
| **Top Retrieval Vector Similarity** | **0.1460** | **0.4308** | **+0.2848** |
| **Groundedness Score** | **5.0 / 5.0** | **5.0 / 5.0** | 0.0 |
| **Average Correctness** | **3.00 / 5.0** | **3.36 / 5.0** | **+0.36** |
| **Average Relevance** | **2.65 / 5.0** | **3.33 / 5.0** | **+0.68** |
| **Average Completeness** | **1.55 / 5.0** | **2.27 / 5.0** | **+0.72** |
| **Escalation Appropriateness** | **2.09 / 5.0** | **4.27 / 5.0** | **+2.18** |
| **Retrieval Latency (Offline)** | **0.05 ms** | **8.44 ms** | +8.39 ms |
| **Total Agent Latency (Offline)** | **0.17 ms** | **8.64 ms** | +8.47 ms |

### Key Result Interpretation:
- **Dense Embeddings Improve Retrieval Context**: Version B increased top retrieval similarity from `0.1460` to `0.4308`, providing richer context snippets.
- **Higher Deterministic Pass Rate**: Deterministic check pass rate improved from `36.4%` (Version A) to `90.9%` (Version B) due to improved context grounding.
- **Trade-off in Latency**: Version B introduces an 8.44 ms retrieval latency overhead for model embedding inference compared to 0.05 ms for TF-IDF.

---

## 12. Failure Analysis

The evaluation framework automatically classifies test case failures into root-cause categories:

1. **Deterministic Verification Failures**: In Version B, 1 out of 11 cases (`t023`) failed the strict escalation pass check due to an unnecessary escalation flag on a standard refund inquiry.
2. **Generation Quality Gaps**: Under offline `template_fallback` mode, 7 cases scored below the target 4.0 overall score threshold, highlighting the completeness difference between structured template rendering and full LLM synthesis.
3. **Intent Misclassification**: 1 case (`t006`: automation rule issue) was classified as `billing` rather than `technical_bug` due to keyword overlap (`"refund"` rule description).

---

## 13. Testing & Regression Suite

Automated testing is enforced via `pytest`:

```bash
pytest
```

- **Result**: `8 passed` in 21.46s.
- **Coverage Areas**: Dataset loading, schema validation, multi-intent classification, leave-one-out RAG search, low-confidence escalation, and evaluator aggregation.

---

## 14. System Limitations

1. **Golden Evaluation Set Size**: 11 golden cases mean each test case represents ~9.1 percentage points on aggregate metrics.
2. **Single-Reviewer Ground Truth**: Ground-truth labels reflect a single reviewer's validation dataset.
3. **Fallback Modes**: When running without external API keys, generation defaults to template fallbacks and judging uses heuristic rules.
4. **Regex-Assisted Classification**: Intent detection relies on keyword patterns rather than a fine-tuned Transformer classifier.
5. **Local Vector Storage**: In-memory retrieval vectors are generated on startup rather than queried from a persistent vector database.

---

## 15. Reproducibility Guide

To reproduce the exact evaluation metrics reported in this document:

```powershell
# 1. Environment Setup
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Dependency Installation
pip install -r requirements.txt

# 3. Data Pipeline Execution
python scripts/prepare_data.py

# 4. Evaluation Execution
python run_eval.py

# 5. Automated Test Suite
pytest
```

---

## 16. Future Engineering Roadmap

1. **Fine-Tuned Intent Classifier**: Replace regex pattern matching with a fine-tuned DeBERTa intent classifier.
2. **Live LLM Integration**: Connect Anthropic Claude 3.5 Sonnet / OpenAI GPT-4o API keys for full LLM response generation.
3. **Persistent Vector Store**: Migrate from in-memory numpy vectors to ChromaDB or Qdrant.
4. **Multi-Reviewer Validation**: Expand human validation to multiple independent reviewers to compute Inter-Rater Reliability (Fleiss' Kappa).

---

## 17. Conclusion

This project demonstrates a production-grade AI support agent backed by a **trustworthy, reproducible evaluation framework**. By introducing leave-one-out retrieval, multi-intent classification, deterministic rule checking, and separated latency profiling, the platform provides clear, un-distorted diagnostic visibility into AI support performance.
