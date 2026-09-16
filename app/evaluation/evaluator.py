"""
Evaluation Harness Orchestrator.
Loads golden dataset -> runs agent in leave-one-out evaluation safe mode -> executes deterministic checks
-> computes semantic similarity -> invokes LLM-as-a-Judge -> analyzes failures -> generates versioned evaluation reports.

Runs evaluation for both:
  - Version A: TF-IDF Retrieval Baseline (Evaluation Safe Leave-One-Out)
  - Version B: SentenceTransformers Dense Embedding Retrieval (Evaluation Safe Leave-One-Out)
"""

from __future__ import annotations
import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List
from app.data.loader import DataLoader
from app.agent.retrieval import build_knowledge_base_from_tickets
from app.agent.agent import SupportAgent
from app.evaluation.deterministic import evaluate_deterministic
from app.evaluation.semantic import evaluate_semantic
from app.evaluation.llm_judge import evaluate_llm_judge
from app.evaluation.failure_analysis import analyze_failures
from app.evaluation.metrics import calculate_aggregate_metrics
from app.evaluation.validation import validate_judge


class EvaluationHarness:
    """
    Complete, reproducible Evaluation Harness.
    """

    def __init__(self, golden_path: str = "data/golden/golden_v1.jsonl",
                 raw_tickets_path: str = "data/raw/tickets.jsonl"):
        self.golden_path = Path(golden_path)
        self.raw_tickets_path = Path(raw_tickets_path)

    def evaluate_version(self, use_sentence_transformers: bool,
                         version_label: str) -> Dict[str, Any]:
        loader = DataLoader()
        golden_set = loader.load_auto(self.golden_path)
        raw_tickets = loader.load_auto(self.raw_tickets_path)

        kb = build_knowledge_base_from_tickets(
            raw_tickets,
            use_sentence_transformers=use_sentence_transformers
        )

        agent = SupportAgent(vector_store=kb)
        results = []

        print(f"Running {version_label} evaluation across {len(golden_set)} test cases (Leave-One-Out mode)...")

        for golden in golden_set:
            gid = golden.get("id", "")
            query = golden.get("query", "")
            expected_intent = golden.get("intent", golden.get("ideal_category", ""))
            expected_answer = golden.get("expected_answer", golden.get("ideal_response", ""))
            difficulty = golden.get("difficulty", "medium")

            # 1. Run Agent with Leave-One-Out source exclusion to prevent retrieval leakage!
            agent_input = {"id": gid, "subject": golden.get("subject", query[:40]), "body": query}
            agent_output = agent.process(agent_input, exclude_ticket_id=gid)

            # 2. Deterministic Checks
            det_res = evaluate_deterministic(agent_output.to_dict(), golden)

            # 3. Semantic Similarity
            sem_res = evaluate_semantic(agent_output.to_dict(), golden)

            # 4. LLM-as-a-Judge
            judge_res = evaluate_llm_judge(agent_output.to_dict(), golden)

            cat_correct = (agent_output.intent == expected_intent)

            record = {
                "id": gid,
                "query": query,
                "ideal_category": expected_intent,
                "predicted_category": agent_output.intent,
                "category_correct": cat_correct,
                "agent_response": agent_output.response,
                "expected_answer": expected_answer,
                "needs_human": agent_output.needs_human,
                "expected_escalation": golden.get("expected_escalation", False),
                "escalation_reason": agent_output.escalation_reason,
                "difficulty": difficulty,
                "confidence": agent_output.confidence,

                # Latencies & Modes
                "retrieval_latency_ms": agent_output.retrieval_latency_ms,
                "generation_latency_ms": agent_output.generation_latency_ms,
                "total_agent_latency_ms": agent_output.total_agent_latency_ms,
                "generation_mode": agent_output.generation_mode,
                "retrieval_method": agent_output.metadata.get("retrieval_method", "Unknown"),
                "excluded_source_id": agent_output.metadata.get("excluded_source_id", gid),

                # Canonical Retrieval Similarity & Context
                "retrieval_top_similarity": agent_output.retrieval_top_similarity,
                "retrieved_context": agent_output.retrieved_context,
                "retrieved_ids": agent_output.retrieved_ids,

                # Evaluation Scores
                "deterministic_score": det_res.deterministic_score,
                "deterministic_pass": det_res.deterministic_pass,
                "deterministic_details": det_res.to_dict(),
                "semantic_similarity": sem_res["semantic_similarity"],
                "overall_score": judge_res.overall_score,
                "correctness": judge_res.correctness,
                "relevance": judge_res.relevance,
                "completeness": judge_res.completeness,
                "groundedness": judge_res.groundedness,
                "helpfulness": judge_res.helpfulness,
                "escalation_appropriateness": judge_res.escalation_appropriateness,
                "overall_pass": (judge_res.passed and det_res.deterministic_pass),
                "judge_mode": judge_res.judge_mode,
                "judge_reasoning": judge_res.reason,
                "judge_confidence": judge_res.judge_confidence,
                "failure_type": judge_res.failure_type,
            }
            results.append(record)

        metrics = calculate_aggregate_metrics(results)
        failure_analysis = analyze_failures(results)
        validation_report = validate_judge(results)

        return {
            "version_label": version_label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": metrics,
            "failure_analysis": failure_analysis,
            "judge_validation": {
                "total_cases": validation_report.total_cases,
                "agreements": validation_report.agreements,
                "disagreements": validation_report.disagreements,
                "agreement_rate_pct": validation_report.agreement_rate_pct,
                "reviewer_mode": validation_report.reviewer_mode,
                "disagreement_examples": validation_report.disagreement_examples,
            },
            "results": results,
        }

    def run_evaluation(self, output_dir: str = "evaluation/results") -> Dict[str, Any]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Starting full evaluation pipeline for Version A (TF-IDF) & Version B (SentenceTransformers)...")

        # Evaluate Version A (TF-IDF Baseline)
        data_a = self.evaluate_version(use_sentence_transformers=False, version_label="Version A (TF-IDF Baseline)")
        path_a = out_dir / "version_a_results.json"
        with open(path_a, "w", encoding="utf-8") as f:
            json.dump(data_a, f, indent=2)

        # Evaluate Version B (SentenceTransformers Enhanced)
        data_b = self.evaluate_version(use_sentence_transformers=True, version_label="Version B (SentenceTransformers Enhanced)")
        path_b = out_dir / "version_b_results.json"
        with open(path_b, "w", encoding="utf-8") as f:
            json.dump(data_b, f, indent=2)

        # Write latest_results.json and summary.json
        latest_path = out_dir / "latest_results.json"
        summary_path = out_dir / "summary.json"
        csv_path = out_dir / "latest_results.csv"

        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(data_b, f, indent=2)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": data_b["summary"], "failure_analysis": data_b["failure_analysis"]}, f, indent=2)

        if data_b["results"]:
            fieldnames = ["id", "ideal_category", "predicted_category", "category_correct",
                          "deterministic_score", "retrieval_top_similarity", "semantic_similarity",
                          "overall_score", "overall_pass", "needs_human", "expected_escalation", "failure_type"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data_b["results"])

        metrics_b = data_b["summary"]
        print(f"\nCompleted evaluation!")
        print(f"Version A (TF-IDF): Accuracy={data_a['summary']['category_accuracy_pct']}% | Avg Score={data_a['summary']['avg_overall_score']}/5.0")
        print(f"Version B (Dense Embeddings): Accuracy={metrics_b['category_accuracy_pct']}% | Avg Score={metrics_b['avg_overall_score']}/5.0")
        print(f"Results saved to: {latest_path}")

        return data_b


def run_eval():
    harness = EvaluationHarness()
    return harness.run_evaluation()


if __name__ == "__main__":
    run_eval()
