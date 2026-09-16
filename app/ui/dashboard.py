"""
Streamlit Dashboard for AI Customer Support Agent & Evaluation Platform.
Features:
- Executive Overview Metrics & Latency Breakdown
- Single Case Debugger / Leave-One-Out Trace View
- "Why the Headline Metric Can Be Misleading" Failure Analysis
- Dynamic Agent Version A (TF-IDF) vs Version B (SentenceTransformers) Comparison
- Honest Single-Reviewer Judge Validation
"""

from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Hiver AI Agent Evaluation Platform",
    page_icon="🤖",
    layout="wide",
)

PATH_LATEST = Path("evaluation/results/latest_results.json")
PATH_VER_A = Path("evaluation/results/version_a_results.json")
PATH_VER_B = Path("evaluation/results/version_b_results.json")


@st.cache_data
def load_json(file_path: Path):
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    st.title("🤖 Hiver AI Support Agent & Evaluation Platform")
    st.markdown(
        "**End-to-End Customer Support Agent, Dual RAG Engines, Golden Evaluation Harness, and Failure Analysis**"
    )

    data_latest = load_json(PATH_LATEST)
    if not data_latest:
        st.warning("No evaluation results found. Please run `python -m app.evaluation.evaluator` first.")
        return

    summary = data_latest.get("summary", {})
    failures = data_latest.get("failure_analysis", {})
    judge_val = data_latest.get("judge_validation", {})
    results = data_latest.get("results", [])

    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Overview",
        "🔍 Single Case Debugger",
        "⚠️ Misleading Metrics & Failure Analysis",
        "⚔️ Version A vs B Comparison",
        "⚖️ Judge Validation",
        "📈 Dataset & Limitations",
    ])

    # ------------------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # ------------------------------------------------------------------------
    with tab1:
        st.header("Executive Performance Summary")

        # Configured Modes Status Badges
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.info(f"**Retrieval Engine**: {summary.get('retrieval_method', 'TF-IDF Baseline')}")
        m_col2.info(f"**Generation Mode**: {summary.get('generation_mode', 'template_fallback').upper()}")
        m_col3.info(f"**Judge Evaluator**: {summary.get('judge_mode', 'heuristic_fallback').upper()}")

        st.divider()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Category Accuracy", f"{summary.get('category_accuracy_pct', 0)}%")
        col2.metric("Overall Avg Score", f"{summary.get('avg_overall_score', 0)} / 5.0")
        col3.metric("Pass Rate", f"{summary.get('pass_rate_pct', 0)}%")
        col4.metric("Groundedness", f"{summary.get('avg_groundedness', 0)} / 5.0")
        col5.metric("Local Agent Latency", f"{summary.get('avg_total_agent_latency_ms', 0)} ms")

        st.divider()

        # Latency breakdown
        st.subheader("⏱️ Latency & Similarity Breakdown")
        l_col1, l_col2, l_col3, l_col4 = st.columns(4)
        l_col1.metric("Retrieval Latency", f"{summary.get('avg_retrieval_latency_ms', 0)} ms")
        l_col2.metric("Generation Latency", f"{summary.get('avg_generation_latency_ms', 0)} ms")
        l_col3.metric("Total Agent Latency", f"{summary.get('avg_total_agent_latency_ms', 0)} ms")
        l_col4.metric("Top Retrieval Similarity", f"{summary.get('avg_retrieval_top_similarity', 0):.4f}")

        st.divider()

        # Visualizations row
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.subheader("Performance by Category")
            cat_breakdown = summary.get("per_category_breakdown", {})
            if cat_breakdown:
                cat_df = pd.DataFrame([
                    {"Category": cat, "Accuracy (%)": d["accuracy"] * 100, "Avg Score (1-5)": d["avg_score"]}
                    for cat, d in cat_breakdown.items()
                ])
                fig_cat = px.bar(
                    cat_df, x="Category", y="Accuracy (%)",
                    color="Avg Score (1-5)",
                    color_continuous_scale="Viridis",
                    text_auto=".1f",
                    title="Category Accuracy & Score Breakdown",
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        with row1_col2:
            st.subheader("Performance by Difficulty")
            diff_breakdown = summary.get("per_difficulty_breakdown", {})
            if diff_breakdown:
                diff_df = pd.DataFrame([
                    {"Difficulty": diff.capitalize(), "Accuracy (%)": d["accuracy"] * 100, "Avg Score": d["avg_score"]}
                    for diff, d in diff_breakdown.items()
                ])
                fig_diff = px.bar(
                    diff_df, x="Difficulty", y="Accuracy (%)",
                    color="Avg Score",
                    color_continuous_scale="Magma",
                    text_auto=".1f",
                    title="Accuracy Across Difficulty Tiers",
                )
                st.plotly_chart(fig_diff, use_container_width=True)

        st.subheader("Rubric Dimension Scores (1 - 5 Scale)")
        rubric_data = {
            "Dimension": ["Correctness", "Relevance", "Completeness", "Groundedness", "Escalation Appropriateness", "Deterministic Checks"],
            "Score": [
                summary.get("avg_correctness", 0),
                summary.get("avg_relevance", 0),
                summary.get("avg_completeness", 0),
                summary.get("avg_groundedness", 0),
                summary.get("avg_escalation_appropriateness", 0),
                summary.get("avg_deterministic_score", 0) * 5.0,
            ]
        }
        rubric_df = pd.DataFrame(rubric_data)
        fig_rubric = px.line(rubric_df, x="Dimension", y="Score", markers=True, range_y=[0, 5.5],
                             title="Evaluation Rubric Profile")
        st.plotly_chart(fig_rubric, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 2: SINGLE CASE DEBUGGER
    # ------------------------------------------------------------------------
    with tab2:
        st.header("Single Case Trace & Debug View")
        st.markdown("Inspect end-to-end processing pipeline for any individual customer ticket.")

        ticket_ids = [r["id"] for r in results]
        selected_id = st.selectbox("Select Ticket ID to Inspect:", ticket_ids)

        selected_case = next((r for r in results if r["id"] == selected_id), None)
        if selected_case:
            st.subheader(f"Case Trace: `{selected_case['id']}`")

            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            d_col1.metric("Predicted Intent", selected_case["predicted_category"],
                          delta="Match" if selected_case["category_correct"] else "Misclassified",
                          delta_color="normal" if selected_case["category_correct"] else "inverse")
            d_col2.metric("Overall Score", f"{selected_case['overall_score']} / 5.0")
            d_col3.metric("Needs Human Escalation?", str(selected_case["needs_human"]))
            d_col4.metric("Failure Type", selected_case["failure_type"])

            st.caption(f"**Leave-One-Out Source Exclusion**: Source ticket `{selected_case.get('excluded_source_id', selected_id)}` excluded from search corpus to prevent leakage.")

            st.markdown("### Pipeline Execution Steps")

            st.write("**1. Customer Query:**")
            st.info(selected_case["query"])

            st.write(f"**2. Retrieved Evidence Context & Similarity Scores (Top Sim: {selected_case.get('retrieval_top_similarity', 0.0):.4f}):**")
            retrieved_ctx = selected_case.get("retrieved_context", [])
            if retrieved_ctx:
                for idx, ctx in enumerate(retrieved_ctx):
                    st.code(
                        f"Doc #{idx+1} [ID: {ctx['id']} | Category: {ctx['category']} | Sim: {ctx['similarity']:.4f}]\n"
                        f"Subject: {ctx['subject']}\nBody: {ctx['body']}",
                        language="markdown"
                    )
            else:
                st.warning("No context retrieved above threshold.")

            st.write(f"**3. Generated Agent Response (Mode: `{selected_case.get('generation_mode', 'template_fallback')}`):**")
            st.success(selected_case["agent_response"])

            st.write("**4. Ideal Ground Truth Reference Answer:**")
            st.caption(selected_case["expected_answer"])

            st.write("**5. Evaluation Breakdown & Reasoning:**")
            st.json({
                "deterministic_score": selected_case["deterministic_score"],
                "deterministic_pass": selected_case.get("deterministic_pass", False),
                "deterministic_details": selected_case["deterministic_details"],
                "semantic_similarity": selected_case["semantic_similarity"],
                "retrieval_top_similarity": selected_case.get("retrieval_top_similarity", 0.0),
                "judge_mode": selected_case["judge_mode"],
                "judge_reasoning": selected_case["judge_reasoning"],
                "escalation_reason": selected_case["escalation_reason"],
            })

    # ------------------------------------------------------------------------
    # TAB 3: MISLEADING METRICS & FAILURE ANALYSIS
    # ------------------------------------------------------------------------
    with tab3:
        st.header("⚠️ Why Headline Evaluation Numbers Can Be Misleading")
        st.error(
            "CRITICAL FINDING: An aggregate headline accuracy number (e.g. 90.9%) masks critical underlying "
            "failure modes that impact real business operations and customer retention."
        )

        st.markdown("""
        ### Real Concrete Examples of Metric Distortions from Execution:
        1. **High Category Accuracy vs 0% Recall on High-Value Categories**:
           - In naive rule baselines, keyword search ordering can cause `refund_complaint` (churn risk) tickets to be misclassified as generic `billing`. A headline score of 80%+ masks that **the most retention-critical category has 0% recall**.
        2. **Semantic Similarity vs Fact Resolution**:
           - `Semantic similarity` measures vocabulary overlap, NOT whether all distinct asks were resolved.
           - *Real Case Example (`t002`)*: Bundles a duplicate charge + invoice address change. High word overlap score occurs even if the invoice address change is omitted!
        3. **Easy Case Sample Dominance**:
           - Standard control questions (`how_to`) inflate overall scores, concealing poor performance on complex multi-part or retention-sensitive tickets.
        """)

        st.subheader("Observed Failure Mode Statistics")
        stats = failures.get("statistics", [])
        if stats:
            fail_df = pd.DataFrame(stats)
            fig_fail = px.pie(fail_df, names="failure_type", values="count",
                              title="Observed Failure Modes Distribution",
                              color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_fail, use_container_width=True)

        st.subheader("Detailed Failure Log")
        fail_records = failures.get("failure_records", [])
        if fail_records:
            df_fails = pd.DataFrame(fail_records)
            expected_cols = ["case_id", "failure_type", "judge_score", "retrieval_top_similarity", "root_cause", "suggested_improvement"]
            avail_cols = [c for c in expected_cols if c in df_fails.columns]
            if avail_cols:
                st.dataframe(df_fails[avail_cols])
            else:
                st.dataframe(df_fails)

    # ------------------------------------------------------------------------
    # TAB 4: VERSION A VS VERSION B COMPARISON
    # ------------------------------------------------------------------------
    with tab4:
        st.header("⚔️ Dynamic Agent Version A (TF-IDF) vs Version B (SentenceTransformers)")
        st.markdown("Metrics dynamically calculated from actual evaluation result files (`version_a_results.json` and `version_b_results.json`).")

        data_a = load_json(PATH_VER_A)
        data_b = load_json(PATH_VER_B)

        if data_a and data_b:
            sum_a = data_a.get("summary", {})
            sum_b = data_b.get("summary", {})

            # Overall Summary Comparison Table
            st.subheader("Overall Metric Comparison & Calculated Deltas")
            metrics_comp = [
                {
                    "Metric": "Category Accuracy (%)",
                    "Version A (TF-IDF)": sum_a.get("category_accuracy_pct", 0),
                    "Version B (SentenceTransformers)": sum_b.get("category_accuracy_pct", 0),
                    "Delta": round(sum_b.get("category_accuracy_pct", 0) - sum_a.get("category_accuracy_pct", 0), 2),
                },
                {
                    "Metric": "Pass Rate (%)",
                    "Version A (TF-IDF)": sum_a.get("pass_rate_pct", 0),
                    "Version B (SentenceTransformers)": sum_b.get("pass_rate_pct", 0),
                    "Delta": round(sum_b.get("pass_rate_pct", 0) - sum_a.get("pass_rate_pct", 0), 2),
                },
                {
                    "Metric": "Avg Overall Score (1-5)",
                    "Version A (TF-IDF)": sum_a.get("avg_overall_score", 0),
                    "Version B (SentenceTransformers)": sum_b.get("avg_overall_score", 0),
                    "Delta": round(sum_b.get("avg_overall_score", 0) - sum_a.get("avg_overall_score", 0), 2),
                },
                {
                    "Metric": "Avg Groundedness Score",
                    "Version A (TF-IDF)": sum_a.get("avg_groundedness", 0),
                    "Version B (SentenceTransformers)": sum_b.get("avg_groundedness", 0),
                    "Delta": round(sum_b.get("avg_groundedness", 0) - sum_a.get("avg_groundedness", 0), 2),
                },
                {
                    "Metric": "Top Retrieval Similarity",
                    "Version A (TF-IDF)": sum_a.get("avg_retrieval_top_similarity", 0),
                    "Version B (SentenceTransformers)": sum_b.get("avg_retrieval_top_similarity", 0),
                    "Delta": round(sum_b.get("avg_retrieval_top_similarity", 0) - sum_a.get("avg_retrieval_top_similarity", 0), 4),
                },
                {
                    "Metric": "Avg Agent Latency (ms)",
                    "Version A (TF-IDF)": sum_a.get("avg_total_agent_latency_ms", 0),
                    "Version B (SentenceTransformers)": sum_b.get("avg_total_agent_latency_ms", 0),
                    "Delta": round(sum_b.get("avg_total_agent_latency_ms", 0) - sum_a.get("avg_total_agent_latency_ms", 0), 2),
                },
            ]
            st.dataframe(pd.DataFrame(metrics_comp), use_container_width=True)

            # Per-case Comparison Table
            st.subheader("Per-Case Score Comparison & Deltas")
            res_a = {r["id"]: r for r in data_a.get("results", [])}
            res_b = {r["id"]: r for r in data_b.get("results", [])}

            case_comp = []
            for cid, r_b in res_b.items():
                r_a = res_a.get(cid, {})
                s_a = r_a.get("overall_score", 0.0)
                s_b = r_b.get("overall_score", 0.0)
                delta = round(s_b - s_a, 2)
                status = "Improved" if delta > 0 else ("Regression" if delta < 0 else "Unchanged")

                case_comp.append({
                    "Case ID": cid,
                    "Category": r_b.get("ideal_category", ""),
                    "Version A Score": s_a,
                    "Version B Score": s_b,
                    "Score Delta": f"{delta:+.2f}",
                    "Version A Top Sim": r_a.get("retrieval_top_similarity", 0.0),
                    "Version B Top Sim": r_b.get("retrieval_top_similarity", 0.0),
                    "Status": status,
                })
            st.dataframe(pd.DataFrame(case_comp), use_container_width=True)
        else:
            st.info("Run `python -m app.evaluation.evaluator` to generate both Version A and Version B evaluation files.")

    # ------------------------------------------------------------------------
    # TAB 5: JUDGE VALIDATION
    # ------------------------------------------------------------------------
    with tab5:
        st.header("⚖️ Judge Validation against Human Ground Truth")
        st.warning("📌 **Validation Disclaimer**: Single-reviewer manual validation on 11 golden set cases.")

        val_rate = judge_val.get("agreement_rate_pct", 0.0)
        tot_val = judge_val.get("total_cases", 0)
        agr = judge_val.get("agreements", 0)
        dis = judge_val.get("disagreements", 0)

        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        v_col1.metric("Human-Judge Agreement", f"{val_rate}%")
        v_col2.metric("Total Validated Cases", str(tot_val))
        v_col3.metric("Agreed Cases", str(agr))
        v_col4.metric("Disagreed Cases", str(dis))

        st.subheader("Disagreement Analysis Log")
        dis_examples = judge_val.get("disagreement_examples", [])
        if dis_examples:
            st.dataframe(pd.DataFrame(dis_examples))
        else:
            st.success("Directional agreement observed across validated human ground-truth cases.")

    # ------------------------------------------------------------------------
    # TAB 6: DATASET & STATISTICAL LIMITATIONS
    # ------------------------------------------------------------------------
    with tab6:
        st.header("📈 Dataset Size & Statistical Sensitivity Analysis")
        st.markdown("""
        ### Dataset Breakdown:
        - **Total Dataset Tickets**: 24 tickets (`data/raw/tickets.jsonl`)
        - **Golden Evaluation Cases**: 11 representative test cases (`data/golden/golden_v1.jsonl`)
        - **Categories Represented**: 5 (`account_access`, `billing`, `technical_bug`, `refund_complaint`, `how_to`)
        """)

        golden_count = len(results) if results else 11
        sensitivity = round(100.0 / golden_count, 1)

        st.warning(
            f"⚡ **Statistical Sensitivity**: With **{golden_count} golden cases**, a single case result change shifts overall category accuracy by **~{sensitivity} percentage points**. "
            "Small evaluation sets are highly sensitive to individual test case performance and should be interpreted as diagnostic signals rather than production-level statistical guarantees."
        )


if __name__ == "__main__":
    main()
