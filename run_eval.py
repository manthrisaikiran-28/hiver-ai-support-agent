"""
Orchestrates: load golden set -> run agent -> run judge -> aggregate + print.

Usage: python3 run_eval.py
Writes results.json with per-ticket scores plus an aggregate summary.
"""
import json
from agent import run_agent
from llm_judge import judge


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    tickets = {t["id"]: t for t in load_jsonl("data/tickets.jsonl")}
    golden = load_jsonl("data/golden_set.jsonl")

    results = []
    for g in golden:
        ticket = tickets[g["id"]]
        agent_result = run_agent(ticket)
        judge_result = judge(ticket, agent_result.predicted_category, agent_result.response, g)
        results.append({
            "id": g["id"],
            "subject": ticket["subject"],
            "true_category": ticket["true_category"],
            "ideal_category": g["ideal_category"],
            "predicted_category": agent_result.predicted_category,
            "category_correct": agent_result.predicted_category == g["ideal_category"],
            "agent_response": agent_result.response,
            "correctness": judge_result.correctness,
            "completeness": judge_result.completeness,
            "category_fit": judge_result.category_fit,
            "overall": judge_result.overall,
            "judge_mode": judge_result.judge_mode,
            "judge_reasoning": judge_result.reasoning,
        })

    n = len(results)
    category_acc = sum(r["category_correct"] for r in results) / n
    avg_overall = sum(r["overall"] for r in results) / n
    avg_correctness = sum(r["correctness"] for r in results) / n

    print(f"Judge mode: {results[0]['judge_mode']} (all tickets)")
    print(f"Tickets evaluated: {n}")
    print(f"Category accuracy: {category_acc:.0%}")
    print(f"Avg correctness (1-5): {avg_correctness:.2f}")
    print(f"Avg overall score (1-5): {avg_overall:.2f}")
    print()
    print(f"{'id':6} {'cat_ok':7} {'overall':8} subject")
    for r in results:
        print(f"{r['id']:6} {str(r['category_correct']):7} {r['overall']:<8} {r['subject']}")

    with open("results.json", "w") as f:
        json.dump({
            "summary": {
                "n": n,
                "category_accuracy": category_acc,
                "avg_correctness": avg_correctness,
                "avg_overall": avg_overall,
                "judge_mode": results[0]["judge_mode"],
            },
            "results": results,
        }, f, indent=2)
    print("\nWrote results.json")


if __name__ == "__main__":
    main()
