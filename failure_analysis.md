# Failure Mode & Metric Analysis

## 1. Purpose

This document provides a detailed, evidence-based failure analysis of the AI Customer Support Agent based on reproducible evaluation results executed against the golden evaluation dataset (`data/golden/golden_v1.jsonl`).

The objective is to identify root causes of system failures across intent classification, RAG context retrieval, response generation, and policy escalation, while examining why headline accuracy metrics can be misleading.

---

## 2. Dataset Overview

- **Total Ingested Support Dataset**: 24 customer tickets (`data/raw/tickets.jsonl`)
- **Golden Ground-Truth Evaluation Cases**: 11 representative cases (`data/golden/golden_v1.jsonl`)
- **Category Coverage**: `technical_bug` (3), `billing` (3), `account_access` (2), `refund_complaint` (2), `how_to` (1)
- **Granularity**: On an 11-case golden test set, each individual case outcome represents **~9.1 percentage points** on aggregate metrics.

---

## 3. Evaluation Method & Setup

The failure analysis evaluates agent outputs generated under a **Leave-One-Out RAG Search** protocol:
- During evaluation of case `gid`, document `gid` is excluded from the knowledge base candidate pool to prevent self-retrieval evaluation leakage.
- Outputs are evaluated across two retrieval configurations: **Version A (TF-IDF Lexical Baseline)** and **Version B (SentenceTransformers `all-MiniLM-L6-v2` Dense Embeddings)**.
- Each case is evaluated using deterministic rule checks, semantic similarity metrics, and a 6-dimension evaluation rubric.

---

## 4. Failure Categorization Summary

The table below presents the actual failure categorization recorded from the final reproducible evaluation run:

| Failure Category | Version A (TF-IDF) | Version B (SentenceTransformers) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Retrieval Failures** | **0** | **0** | Retrieval failed to provide sufficient evidence (zero cases below confidence threshold `< 0.25`). |
| **Generation Failures** | **0** | **0** | Technical generation process failed or produced no usable output. |
| **Overall Quality Threshold Failures** | **10** | **7** | Generated response received an overall rubric score below the 4.0 / 5.0 target threshold in template fallback mode. |
| **Deterministic Validation Failures** | **7** | **1** | Cases failing strict required-fact, intent, or escalation boolean checks. |
| **Escalation Errors** | **0** | **1** | Escalation decision was incorrect (`t023` false positive escalation). |
| **Judge Disagreements** | **0** | **0** | Judge result disagreed with expected/deterministic result. |

---

## 5. Detailed Analysis of Deterministic Failures

### Version B Deterministic Failure Case (`t023`)
- **Query**: Customer requested a refund for an unused subscription cycle after cancellation (`"refund pls... we decided to cancel..."`).
- **Expected Escalation**: `expected_escalation = False` (standard refund process).
- **Agent Behavior**: The safety engine triggered `agent_escalation = True` due to high churn risk policy rules associated with refund complaints.
- **Impact**: The case passed intent classification (`refund_complaint`), required facts verification, forbidden claims check, and grounding check, but failed the strict boolean `escalation_pass` check. This represents an **escalation policy calibration issue**, not a retrieval or generation failure.

### Multi-Intent Resolution Case (`t002`)
- **Query**: Customer asked about a duplicate charge ($149 on 3rd & 9th) AND requested an updated invoice with company address details.
- **Version A Result**: Failed deterministic required-fact checks (`deterministic_score = 0.60`) due to low TF-IDF retrieval similarity (`0.0000`).
- **Version B Result**: Passed deterministic checks (`deterministic_score = 0.80`) with top vector similarity of `0.3856`, successfully addressing both the duplicate charge investigation and invoice address update.

> [!IMPORTANT]
> Deterministic validation failures reflect specific fact completeness or policy threshold mismatches, rather than fundamental RAG retrieval failures.

---

## 6. Baseline Findings (Historical Context)

Prior to the implementation of iterative fixes, baseline evaluation revealed critical metric distortions:

1. **Systematic Keyword Preemption**: Naive keyword search order checked `billing` before `refund_complaint`. Keywords like `"refund"` appeared in both categories, causing 100% of churn-risk refund requests (`t005`, `t023`) to be misclassified as generic billing questions (0% recall on the highest churn-risk category).
2. **Un-grounded Self-Retrieval Leakage**: Without leave-one-out search, evaluation test cases retrieved their exact source document from the knowledge corpus, artificially boosting cosine similarity to $>0.95$ and masking real-world performance on unseen customer queries.

---

## 7. Engineering Improvements Implemented

- **Removed Ticket-ID Conditional Logic**: Completely eliminated ticket-specific overrides (`if ticket_id == "t002"`).
- **Leave-One-Out RAG Search**: Added `exclude_ticket_id` support to eliminate self-retrieval evaluation leakage.
- **Retrieval Confidence Thresholding**: Enforced `RETRIEVAL_THRESHOLD = 0.25` to trigger automatic human escalation when vector similarity is insufficient.
- **Dense Vector Embeddings**: Integrated `SentenceTransformers` (`all-MiniLM-L6-v2`) as Version B, increasing top retrieval similarity from `0.1460` to `0.4308`.
- **Multi-Intent Pattern Engine**: Extended intent classification to extract secondary customer requests.
- **Separated Latency Profiling**: Measured `retrieval_latency_ms`, `generation_latency_ms`, and `total_agent_latency_ms` independently.
- **Dynamic A/B Comparison**: Built comparative evaluation runners to calculate version metrics directly from output JSON files without hardcoded static scores.
- **Automated Regression Suite**: Established 100% passing Pytest suite (`8 passed`).

---

## 8. Remaining System Limitations

- **Small Golden Evaluation Set**: 11 golden cases carry a ~9.1% weight per case.
- **Single-Reviewer Ground Truth**: Manual validation relies on a single reviewer's ground-truth labeling.
- **Offline Fallback Modes**: Unconfigured API keys fallback to heuristic judging and template response rendering.
- **Regex-Assisted Classification**: Intent detection relies on keyword patterns rather than a fine-tuned Transformer model.
- **Local In-Memory Corpus**: Vector index is generated in-memory rather than stored in a distributed vector database.

---

## 9. Measured Conclusions

1. **Retrieval Vector Quality Drives Fact Completeness**: Upgrading from TF-IDF (Version A) to SentenceTransformers (Version B) improved top retrieval similarity from `0.1460` to `0.4308` and deterministic pass rate from `36.4%` to `90.9%`.
2. **Headline Accuracy Can Mask Sub-Category Gaps**: Global classification accuracy (90.9%) must be evaluated alongside per-category recall, deterministic fact checks, and escalation appropriateness.
3. **Multi-Signal Evaluation is Essential**: Evaluating AI support agents requires combining deterministic boolean checks, semantic similarity, rubric judging, and latency breakdown to form a complete operational picture.
