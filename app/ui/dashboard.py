"""
Hiver Support Agent — Evaluation & Analytics Dashboard
Clean enterprise UI for inspecting support agent performance,
RAG retrieval traces, metric distributions, and version comparisons.
"""

from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Hiver Support Agent Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PATH_LATEST = Path("evaluation/results/latest_results.json")
PATH_VER_A = Path("evaluation/results/version_a_results.json")
PATH_VER_B = Path("evaluation/results/version_b_results.json")

# Custom Light Theme CSS
CUSTOM_CSS = """
<style>
    /* Global styles */
    .main {
        background-color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header styling */
    .app-header {
        padding: 1.25rem 0 1rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.825rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Status badge box */
    .badge-card {
        background-color: #F1F5F9;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.875rem;
        color: #334155;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748B;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        font-weight: 600;
    }
</style>
"""


@st.cache_data
def load_json(file_path: Path):
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_chart_theme(fig):
    """Apply clean light theme formatting to Plotly charts."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="sans-serif", size=12, color="#334155"),
        title_font=dict(size=14, color="#0F172A", family="sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    )
    return fig


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Clean Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Hiver Support Agent Evaluation Dashboard</h1>
        <p class="app-subtitle">Performance Analytics, Grounding Diagnostics, and A/B Model Comparisons</p>
    </div>
    """, unsafe_allow_html=True)

    data_latest = load_json(PATH_LATEST)
    if not data_latest:
        st.warning("No evaluation results found. Please run `python run_eval.py` first.")
        return

    summary = data_latest.get("summary", {})
    failures = data_latest.get("failure_analysis", {})
    judge_val = data_latest.get("judge_validation", {})
    results = data_latest.get("results", [])

    # Navigation Tabs (Clean Professional Titles)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary",
        "Case Debugger",
        "Metric Analysis",
        "A/B Comparison",
        "Human Validation",
        "Dataset Profiling",
    ])

    # ------------------------------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY
    # ------------------------------------------------------------------------
    with tab1:
        st.subheader("System Configuration & Performance Metrics")

        # System Config Status Cards
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="badge-card"><b>Retrieval Method:</b> {summary.get("retrieval_method", "TF-IDF")}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="badge-card"><b>Generation Mode:</b> {summary.get("generation_mode", "template_fallback").upper()}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="badge-card"><b>Evaluator Mode:</b> {summary.get("judge_mode", "heuristic_fallback").upper()}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # High-level Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Category Accuracy", f"{summary.get('category_accuracy_pct', 0)}%")
        m2.metric("Average Score", f"{summary.get('avg_overall_score', 0)} / 5.0")
        m3.metric("Pass Rate", f"{summary.get('pass_rate_pct', 0)}%")
        m4.metric("Groundedness", f"{summary.get('avg_groundedness', 0)} / 5.0")
        m5.metric("Avg Latency", f"{summary.get('avg_total_agent_latency_ms', 0)} ms")

        st.divider()

        # Latency Breakdown
        st.subheader("Latency & Retrieval Similarity Breakdown")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Retrieval Latency", f"{summary.get('avg_retrieval_latency_ms', 0)} ms")
        l2.metric("Generation Latency", f"{summary.get('avg_generation_latency_ms', 0)} ms")
        l3.metric("Total Latency", f"{summary.get('avg_total_agent_latency_ms', 0)} ms")
        l4.metric("Top Vector Similarity", f"{summary.get('avg_retrieval_top_similarity', 0):.4f}")

        st.divider()

        # Visualizations
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            cat_breakdown = summary.get("per_category_breakdown", {})
            if cat_breakdown:
                cat_df = pd.DataFrame([
                    {"Category": cat.replace("_", " ").title(), "Accuracy (%)": d["accuracy"] * 100, "Avg Score": d["avg_score"]}
                    for cat, d in cat_breakdown.items()
                ])
                fig_cat = px.bar(
                    cat_df, x="Category", y="Accuracy (%)",
                    color_discrete_sequence=["#2563EB"],
                    text_auto=".0f",
                    title="Accuracy by Ticket Category",
                )
                apply_chart_theme(fig_cat)
                st.plotly_chart(fig_cat, use_container_width=True)

        with col_chart2:
            diff_breakdown = summary.get("per_difficulty_breakdown", {})
            if diff_breakdown:
                diff_df = pd.DataFrame([
                    {"Difficulty": diff.capitalize(), "Accuracy (%)": d["accuracy"] * 100, "Avg Score": d["avg_score"]}
                    for diff, d in diff_breakdown.items()
                ])
                fig_diff = px.bar(
                    diff_df, x="Difficulty", y="Accuracy (%)",
                    color_discrete_sequence=["#0D9488"],
                    text_auto=".0f",
                    title="Accuracy across Difficulty Tiers",
                )
                apply_chart_theme(fig_diff)
                st.plotly_chart(fig_diff, use_container_width=True)

        st.subheader("Evaluation Rubric Profile (1 - 5 Scale)")
        rubric_data = {
            "Dimension": ["Correctness", "Relevance", "Completeness", "Groundedness", "Escalation", "Deterministic"],
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
                             color_discrete_sequence=["#2563EB"], title="Performance Across Evaluation Dimensions")
        apply_chart_theme(fig_rubric)
        st.plotly_chart(fig_rubric, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 2: SINGLE CASE DEBUGGER
    # ------------------------------------------------------------------------
    with tab2:
        st.subheader("Case Execution & RAG Trace Debugger")
        st.caption("Inspect individual golden test cases, RAG context retrieval, and decision rationale.")

        if results:
            case_options = {f"{r['id']} — [{r.get('difficulty', '').upper()}] {r.get('user_query', '')[:45]}...": r for r in results}
            selected_key = st.selectbox("Select Test Case:", list(case_options.keys()))
            case = case_options[selected_key]

            st.divider()

            # Case Overview Cards
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Case ID", case.get("id"))
            d2.metric("Predicted Category", case.get("predicted_category"))
            d3.metric("Ideal Category", case.get("ideal_category"))
            d4.metric("Overall Score", f"{case.get('overall_score', 0)} / 5.0")

            st.markdown("#### User Query")
            st.info(case.get("user_query", ""))

            st.markdown("#### Generated Agent Response")
            st.success(case.get("agent_response", ""))

            st.markdown("#### RAG Retrieved Knowledge Base Context")
            retrieved_docs = case.get("retrieved_context", [])
            if retrieved_docs:
                for idx, doc in enumerate(retrieved_docs, 1):
                    with st.expander(f"Context Document #{idx} (Similarity: {case.get('retrieval_top_similarity', 0.0):.4f})"):
                        st.markdown(f"**Source Ticket**: `{doc.get('source_ticket_id', 'N/A')}`")
                        st.write(doc.get("text", ""))
            else:
                st.write("No external context retrieved.")

            st.markdown("#### Evaluation Rubric Breakdown")
            scores = case.get("rubric_scores", {})
            r_cols = st.columns(6)
            r_cols[0].metric("Correctness", f"{scores.get('correctness', 0)}/5")
            r_cols[1].metric("Relevance", f"{scores.get('relevance', 0)}/5")
            r_cols[2].metric("Completeness", f"{scores.get('completeness', 0)}/5")
            r_cols[3].metric("Groundedness", f"{scores.get('groundedness', 0)}/5")
            r_cols[4].metric("Escalation", f"{scores.get('escalation_appropriateness', 0)}/5")
            r_cols[5].metric("Deterministic", f"{case.get('deterministic_score', 0):.2f}")

    # ------------------------------------------------------------------------
    # TAB 3: METRIC ANALYSIS & FAILURE MODES
    # ------------------------------------------------------------------------
    with tab3:
        st.subheader("Failure Mode Analysis & Diagnostic Insights")
        st.markdown("""
        ### Empirical Analysis of Metric Limitations:
        1. **Keyword Preemption vs High-Value Category Recall**:
           - Simple keyword search order can misclassify high-churn `refund_complaint` tickets as generic `billing`. A high headline accuracy score can mask 0% recall on critical categories.
        2. **Semantic Similarity vs Fact Completeness**:
           - Cosine similarity measures word overlap, not complete task resolution. Bundled queries (e.g. `t002`) can yield high similarity even if secondary requests are omitted.
        3. **Sample Bias in Test Distributions**:
           - Over-representing simple `how_to` tickets inflates global averages while obscuring performance gaps on edge cases.
        """)

        st.divider()
        st.subheader("Observed Failure Distribution")
        stats = failures.get("statistics", [])
        if stats:
            fail_df = pd.DataFrame(stats)
            fig_fail = px.pie(fail_df, names="failure_type", values="count",
                               title="Failure Modes Distribution",
                               color_discrete_sequence=["#2563EB", "#0D9488", "#6366F1", "#F59E0B", "#EF4444"])
            apply_chart_theme(fig_fail)
            st.plotly_chart(fig_fail, use_container_width=True)

        st.subheader("Detailed Failure Log")
        fail_records = failures.get("failure_records", [])
        if fail_records:
            df_fails = pd.DataFrame(fail_records)
            expected_cols = ["case_id", "failure_type", "judge_score", "retrieval_top_similarity", "root_cause", "suggested_improvement"]
            avail_cols = [c for c in expected_cols if c in df_fails.columns]
            if avail_cols:
                st.dataframe(df_fails[avail_cols], use_container_width=True)
            else:
                st.dataframe(df_fails, use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 4: A/B VERSION COMPARISON
    # ------------------------------------------------------------------------
    with tab4:
        st.subheader("Agent A/B Testing: Version A (TF-IDF) vs Version B (SentenceTransformers)")
        st.caption("Metrics dynamically computed from evaluation output files (`version_a_results.json` vs `version_b_results.json`).")

        data_a = load_json(PATH_VER_A)
        data_b = load_json(PATH_VER_B)

        if data_a and data_b:
            sum_a = data_a.get("summary", {})
            sum_b = data_b.get("summary", {})

            st.markdown("#### Aggregate Metrics Comparison")
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
                    "Metric": "Top Retrieval Similarity",
                    "Version A (TF-IDF)": sum_a.get("avg_retrieval_top_similarity", 0),
                    "Version B (SentenceTransformers)": sum_b.get("avg_retrieval_top_similarity", 0),
                    "Delta": round(sum_b.get("avg_retrieval_top_similarity", 0) - sum_a.get("avg_retrieval_top_similarity", 0), 4),
                },
                {
                    "Metric": "Avg Latency (ms)",
                    "Version A (TF-IDF)": sum_a.get("avg_total_agent_latency_ms", 0),
                    "Version B (SentenceTransformers)": sum_b.get("avg_total_agent_latency_ms", 0),
                    "Delta": round(sum_b.get("avg_total_agent_latency_ms", 0) - sum_a.get("avg_total_agent_latency_ms", 0), 2),
                },
            ]
            st.dataframe(pd.DataFrame(metrics_comp), use_container_width=True)

            st.markdown("#### Per-Case Score Deltas")
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
                    "Delta": f"{delta:+.2f}",
                    "Status": status,
                })
            st.dataframe(pd.DataFrame(case_comp), use_container_width=True)
        else:
            st.info("Run `python run_eval.py` to generate Version A and Version B evaluation files.")

    # ------------------------------------------------------------------------
    # TAB 5: HUMAN VALIDATION
    # ------------------------------------------------------------------------
    with tab5:
        st.subheader("Evaluator Validation against Human Ground Truth")
        st.caption("Validation conducted on 11 golden set cases.")

        val_rate = judge_val.get("agreement_rate_pct", 0.0)
        tot_val = judge_val.get("total_cases", 0)
        agr = judge_val.get("agreements", 0)
        dis = judge_val.get("disagreements", 0)

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Human-Judge Agreement", f"{val_rate}%")
        v2.metric("Validated Cases", str(tot_val))
        v3.metric("Agreed Cases", str(agr))
        v4.metric("Disagreed Cases", str(dis))

        st.markdown("#### Disagreement Analysis")
        dis_examples = judge_val.get("disagreement_examples", [])
        if dis_examples:
            st.dataframe(pd.DataFrame(dis_examples), use_container_width=True)
        else:
            st.success("Directional agreement observed across all validated ground-truth cases.")

    # ------------------------------------------------------------------------
    # TAB 6: DATASET PROFILING
    # ------------------------------------------------------------------------
    with tab6:
        st.subheader("Dataset Profiling & Statistical Sensitivity")
        st.markdown("""
        - **Total Raw Dataset**: 24 customer tickets (`data/raw/tickets.jsonl`)
        - **Golden Evaluation Set**: 11 representative test cases (`data/golden/golden_v1.jsonl`)
        - **Supported Categories**: 5 (`account_access`, `billing`, `technical_bug`, `refund_complaint`, `how_to`)
        """)

        golden_count = len(results) if results else 11
        sensitivity = round(100.0 / golden_count, 1)

        st.info(
            f"**Statistical Sensitivity Note**: With **{golden_count} golden evaluation cases**, a single test case outcome shifts category accuracy by **~{sensitivity}%**. "
            "Evaluation metrics serve as high-signal diagnostic indicators for iterative improvement rather than large-scale statistical proofs."
        )


if __name__ == "__main__":
    main()
