"""
Hackerthorn — SLA & Cost-Leakage Control Room (Streamlit).
Run: streamlit run dashbord.py  (from project root)

Features:
  • Upload your own CSV **or** use built-in demo data.
  • Full inline ML scoring → automation → multi-agent GenAI pipeline.
  • Agent Execution Logs displayed after analysis.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.automation import apply_automation_to_dataframe
from utils.feature_engineering import engineer
from utils.genai import heuristic_sla_pack, multi_agent_sla_pack, pack_to_column_value

ROOT = Path(__file__).resolve().parent

# ───────────────────────────── Page config ──────────────────────────────

st.set_page_config(
    page_title="SLA & Cost-Leakage Control Room",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────── Custom CSS ───────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* KPI metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(30,30,60,0.9), rgba(20,20,50,0.95));
    border: 1px solid rgba(100,100,255,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(80,80,255,0.15), inset 0 1px 0 rgba(255,255,255,0.08);
}
[data-testid="stMetricLabel"] {
    color: #a0a0d0 !important;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #e0e0ff !important;
    font-weight: 800;
    font-size: 1.95rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12112b 0%, #1b1a3a 100%);
    border-right: 1px solid rgba(100,100,255,0.1);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c5c5ff;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(20,20,50,0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #9090c0;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.35);
}

/* Expanders (for agent logs) */
.streamlit-expanderHeader {
    background: rgba(25,25,55,0.8) !important;
    border: 1px solid rgba(100,100,255,0.12) !important;
    border-radius: 12px !important;
    color: #c5c5ff !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}
.streamlit-expanderHeader:hover {
    background: rgba(35,35,70,0.9) !important;
    border-color: rgba(130,130,255,0.2) !important;
}
.streamlit-expanderContent {
    background: rgba(15,15,40,0.7) !important;
    border: 1px solid rgba(100,100,255,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* Upload area */
[data-testid="stFileUploader"] {
    border-radius: 16px !important;
}
[data-testid="stFileUploader"] > div {
    border-radius: 16px !important;
    border: 2px dashed rgba(100,100,255,0.25) !important;
    background: rgba(20,20,50,0.5) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(130,130,255,0.4) !important;
    background: rgba(30,30,65,0.6) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(79,70,229,0.5) !important;
    transform: translateY(-1px);
}

/* Success / warning boxes */
.stAlert {
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
}

/* Agent log badge pill */
.agent-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-right: 6px;
}
.badge-reason  { background: #312e81; color: #a5b4fc; }
.badge-action  { background: #1e3a5f; color: #7dd3fc; }
.badge-escalation { background: #4a1942; color: #f0abfc; }
.badge-heuristic  { background: #3f3f46; color: #a1a1aa; }
.badge-llm        { background: #064e3b; color: #6ee7b7; }
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────── Helpers ──────────────────────────────────


def safe_encode(le: LabelEncoder, val) -> int:
    val = str(val)
    if val in le.classes_:
        return int(le.transform([val])[0])
    return 0


def format_insight_from_json(s: str) -> str:
    if not s:
        return ""
    try:
        d = json.loads(s)
        return (
            f"[Reason] {d.get('reason', '')}\n"
            f"[Action] {d.get('action', '')}\n"
            f"[Escalation] {d.get('escalation', '')}"
        )
    except Exception:
        return s


def parse_agent_cell(raw: str) -> dict | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def run_pipeline(df_test: pd.DataFrame, logs: list[str]) -> pd.DataFrame:
    """Run the ML inference → automation → GenAI pipeline, appending progress to *logs*."""

    ts = lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3]

    logs.append(f"[{ts()}] 🔄 Pipeline started")
    logs.append(f"[{ts()}]    Rows to score: {len(df_test):,}")

    logs.append(f"[{ts()}] ⚙️  Loading pre-trained model artifacts from models/ …")
    model_path = ROOT / "models" / "model.pkl"
    artifacts_path = ROOT / "models" / "artifacts.pkl"
    if not model_path.exists() or not artifacts_path.exists():
        logs.append(f"[{ts()}] ❌ Error: Missing pre-trained model or artifacts.")
        st.error("Missing pre-trained model or artifacts in models/. Run app.py first.")
        return df_test
    
    model = pickle.load(open(model_path, "rb"))
    artifacts = pickle.load(open(artifacts_path, "rb"))
    le, imp, scaler, features = artifacts["le"], artifacts["imp"], artifacts["scaler"], artifacts["features"]

    # Feature engineering
    logs.append(f"[{ts()}] ⚙️  Feature engineering …")
    df_test = engineer(df_test)
    logs.append(f"[{ts()}]    Generated: priority_score, time_ratio, urgency_score, risk_score, tight_deadline, very_tight")

    X_test = df_test[features].copy()
    X_test["assigned_to"] = X_test["assigned_to"].apply(lambda v: safe_encode(le, v))
    X_test_s = scaler.transform(imp.transform(X_test))

    # Inference
    logs.append(f"[{ts()}] 🌲 Running Inference with pre-trained RandomForest …")
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    df_test["predicted_breach"] = y_pred
    df_test["probability"] = y_prob
    df_test["SLA_Status"] = df_test["predicted_breach"].map({0: "SAFE", 1: "AT_RISK"})

    at_risk_n = int((y_pred == 1).sum())
    safe_n = int((y_pred == 0).sum())
    logs.append(f"[{ts()}] 📊 Prediction complete → SAFE: {safe_n:,}  |  AT_RISK: {at_risk_n:,}")

    # Automation layer
    logs.append(f"[{ts()}] 🤖 Automation layer: pool simulation, reassignment, escalation tiers …")
    df_test = apply_automation_to_dataframe(df_test)

    action_counts = df_test[df_test["predicted_breach"] == 1]["automation_action"].value_counts()
    for act, cnt in action_counts.items():
        logs.append(f"[{ts()}]    {act}: {cnt}")

    # Multi-agent GenAI
    max_llm = int(os.environ.get("MAX_GENAI_ROWS", "40"))
    risk_mask = df_test["predicted_breach"] == 1
    top_llm_idx = (
        df_test.loc[risk_mask]
        .sort_values("probability", ascending=False)
        .head(max_llm)
        .index.tolist()
    )
    llm_set = set(top_llm_idx)

    skip_genai = bool(os.environ.get("SKIP_GENAI"))
    logs.append(f"[{ts()}] 🧠 Multi-agent GenAI layer (mode={'HEURISTIC' if skip_genai else 'LLM'}, max_llm_rows={max_llm}) …")

    packs: list[str] = []
    agent_logs_detail: list[dict] = []
    for idx, row in df_test.iterrows():
        if int(row["predicted_breach"]) != 1:
            packs.append("")
            continue

        t0 = time.time()
        if idx in llm_set and not skip_genai:
            pack = multi_agent_sla_pack(row)
            mode = "llm"
        else:
            pack = heuristic_sla_pack(row)
            mode = "heuristic"
        elapsed = time.time() - t0
        packs.append(pack_to_column_value(pack))

        agent_logs_detail.append({
            "idx": idx,
            "mode": mode,
            "elapsed_ms": round(elapsed * 1000, 1),
            "priority": row.get("priority", "?"),
            "assigned_to": row.get("assigned_to", "?"),
            "probability": float(row.get("probability", 0)),
            "reason": pack.get("reason", ""),
            "action": pack.get("action", ""),
            "escalation": pack.get("escalation", ""),
        })

    df_test["AI_MultiAgent"] = packs
    df_test["AI_Insight"] = df_test["AI_MultiAgent"].map(format_insight_from_json)

    llm_count = sum(1 for d in agent_logs_detail if d["mode"] == "llm")
    heur_count = sum(1 for d in agent_logs_detail if d["mode"] == "heuristic")
    logs.append(f"[{ts()}]    Processed {len(agent_logs_detail)} at-risk rows (LLM: {llm_count}, Heuristic: {heur_count})")

    # Financial rollup
    exposure = float(pd.to_numeric(df_test.get("financial_exposure_ev", 0), errors="coerce").fillna(0).sum())
    prevention = float(pd.to_numeric(df_test.get("prevention_value_estimate", 0), errors="coerce").fillna(0).sum())
    logs.append(f"[{ts()}] 💰 Financial → Exposure EV: ${exposure:,.0f}  |  Prevention value: ${prevention:,.0f}")
    logs.append(f"[{ts()}] ✅ Pipeline complete")

    # Stash extra state
    st.session_state["agent_logs_detail"] = agent_logs_detail
    st.session_state["pipeline_logs"] = logs

    return df_test


# ────────────────────── Sidebar (branding only) ─────────────────────────

st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <span style="font-size:2.2rem;">🛡️</span>
        <h2 style="margin:4px 0 0 0; color:#c5c5ff; font-weight:800; letter-spacing:-0.02em;">Control Room</h2>
        <p style="color:#7070a0; font-size:0.82rem; margin:0;">SLA & Cost-Leakage Prevention</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Continuous monitoring:** schedule `python app.py` "
    "(cron, Airflow, Azure Logic Apps) so this dashboard reflects "
    "fresh operational scoring."
)

# ───────────────────── Title + Hero ─────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center; padding: 24px 0 8px 0;">
        <h1 style="
            margin:0;
            font-size:2.5rem;
            font-weight:800;
            letter-spacing:-0.03em;
            background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">SLA & Cost-Leakage Control Room</h1>
        <p style="color:#7070a0; font-size:1.02rem; margin:8px 0 0 0; max-width:640px; margin-left:auto; margin-right:auto;">
            Upload your operational data or use the demo dataset — the pipeline scores, automates,
            and explains SLA risks with multi-agent GenAI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# ───────────────── Data source (main page) ──────────────────────────────

ds_left, ds_right = st.columns([1, 1])

with ds_left:
    data_mode = st.radio(
        "📂  Choose data source",
        ["🎯 Use demo data", "📤 Upload dataset"],
        index=0,
        horizontal=True,
    )

uploaded_test = None

if data_mode == "📤 Upload dataset":
    st.markdown(
        "<p style='color:#9090c0; font-size:0.85rem; margin-bottom:4px;'>"
        "Upload your operational data (CSV) with columns: "
        "<code>priority, estimated_time, time_remaining, assigned_to, breach, penalty_cost</code>"
        "</p>",
        unsafe_allow_html=True,
    )
    uploaded_test = st.file_uploader("Operational Data (CSV)", type=["csv"], key="up_test")

with ds_right:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

st.markdown("---")

# ───────────────────── Run pipeline ─────────────────────────────────────

if run_btn:
    logs: list[str] = []

    # Resolve data
    if data_mode == "📤 Upload dataset":
        if uploaded_test is None:
            st.error("Please upload your operational data CSV above.")
            st.stop()
        df_test = pd.read_csv(uploaded_test)
        logs.append(f"[DATA] Loaded uploaded CSV (rows: {len(df_test):,})")
    else:
        test_path = ROOT / "data" / "test.csv"
        if not test_path.exists():
            st.error(f"Demo data not found at `{test_path}`. Place test.csv in data/.")
            st.stop()
        df_test = pd.read_csv(test_path)
        logs.append(f"[DATA] Loaded demo CSV from data/ (rows: {len(df_test):,})")

    # Ensure required columns
    required = {"priority", "estimated_time", "time_remaining", "assigned_to", "breach", "penalty_cost"}
    missing = required - set(df_test.columns)
    if missing:
        st.error(f"**Uploaded CSV** is missing columns: `{', '.join(sorted(missing))}`")
        st.stop()

    df_test["breach"] = df_test["breach"].astype(int)

    with st.spinner("Running ML scoring → Automation → GenAI agents …"):
        df_result = run_pipeline(df_test, logs)

    st.session_state["results_df"] = df_result
    st.session_state["run_ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ───────────────────── Render results ───────────────────────────────────

if "results_df" not in st.session_state:
    # Landing state
    st.markdown(
        """
        <div style="
            text-align:center; padding:60px 20px;
            background: rgba(20,20,50,0.4);
            border-radius: 20px;
            border: 1px dashed rgba(100,100,255,0.15);
            margin: 24px 0;
        ">
            <span style="font-size:3.5rem;">🚀</span>
            <h3 style="color:#a0a0d0; margin:12px 0 4px 0; font-weight:700;">Ready to Analyze</h3>
            <p style="color:#6060a0; max-width:420px; margin:auto; font-size:0.95rem;">
                Choose <b>demo data</b> or <b>upload your own</b> above,
                then click <b>▶ Run Analysis</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

df = st.session_state["results_df"]
run_ts = st.session_state.get("run_ts", "")
logs = st.session_state.get("pipeline_logs", [])
agent_detail = st.session_state.get("agent_logs_detail", [])

# ── KPI Row ──

at_risk = df[df["predicted_breach"] == 1].copy()
safe_n = int((df["predicted_breach"] == 0).sum())
risk_n = len(at_risk)
exposure = float(pd.to_numeric(df.get("financial_exposure_ev", 0), errors="coerce").fillna(0).sum())
prevention = float(pd.to_numeric(df.get("prevention_value_estimate", 0), errors="coerce").fillna(0).sum())
leak_n = int(df["cost_leakage_flag"].sum()) if "cost_leakage_flag" in df.columns else 0

st.markdown("")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tasks SAFE", f"{safe_n:,}")
c2.metric("Tasks AT RISK", f"{risk_n:,}")
c3.metric("Penalty Exposure (EV)", f"${exposure:,.0f}")
c4.metric("Prevention Value", f"${prevention:,.0f}")
c5.metric("Leakage Flags", f"{leak_n:,}")

if run_ts:
    st.success(f"✅  Analysis completed at **{run_ts}**  ·  Scored **{len(df):,}** rows")

# ── Tabs ──

tab_overview, tab_finance, tab_queue, tab_agents, tab_logs = st.tabs(
    ["📊 Overview", "💰 Financial & Leakage", "🤖 Automation Queue", "🧠 Agent Insights", "📋 Execution Logs"]
)

# ─────── Tab: Overview ─────────────────────────────────────────────────

with tab_overview:
    st.subheader("Risk Concentration")
    col_l, col_r = st.columns(2)
    with col_l:
        if len(at_risk) and "automation_action" in at_risk.columns:
            ap = at_risk["automation_action"].value_counts().reset_index()
            ap.columns = ["Action", "Count"]
            fig = px.bar(
                ap, x="Action", y="Count",
                title="Automation Action Mix (At-Risk)",
                color="Count",
                color_continuous_scale=["#4f46e5", "#7c3aed", "#c084fc"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,20,50,0.3)",
                font_color="#c5c5ff",
                title_font_size=15,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if "assigned_to" in at_risk.columns and len(at_risk):
            owners = at_risk.groupby("assigned_to").size().reset_index(name="at_risk")
            owners = owners.sort_values("at_risk", ascending=False).head(12)
            fig2 = px.bar(
                owners, x="assigned_to", y="at_risk",
                title="At-Risk Workload by Owner (Top 12)",
                color="at_risk",
                color_continuous_scale=["#0ea5e9", "#6366f1", "#ec4899"],
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,20,50,0.3)",
                font_color="#c5c5ff",
                title_font_size=15,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Probability vs Penalty (At-Risk)")
    if len(at_risk):
        scat = at_risk.copy()
        scat["penalty_cost"] = pd.to_numeric(scat["penalty_cost"], errors="coerce")
        fig3 = px.scatter(
            scat, x="probability", y="penalty_cost",
            color="automation_action",
            hover_data=["priority", "time_remaining", "assigned_to", "reassigned_to"],
            title="Model Confidence × Contractual Penalty",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,50,0.3)",
            font_color="#c5c5ff",
            title_font_size=15,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ─────── Tab: Financial & Leakage ─────────────────────────────────────

with tab_finance:
    st.markdown(
        "**Financial lens:** Each row carries `penalty_cost` from operations. "
        "We surface **expected exposure** as roughly *probability × penalty* and a "
        "**prevention heuristic** when automation fires."
    )

    fl, fr = st.columns(2)
    with fl:
        if "financial_exposure_ev" in df.columns:
            hist_df = df[df["predicted_breach"] == 1]
            if len(hist_df):
                fig4 = px.histogram(
                    hist_df, x="financial_exposure_ev", nbins=40,
                    title="Distribution of Expected Exposure (At-Risk)",
                    color_discrete_sequence=["#818cf8"],
                )
                fig4.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,50,0.3)",
                    font_color="#c5c5ff",
                    title_font_size=15,
                )
                st.plotly_chart(fig4, use_container_width=True)

    with fr:
        if "prevention_value_estimate" in df.columns:
            prev_df = df[df["prevention_value_estimate"] > 0]
            if len(prev_df):
                fig5 = px.histogram(
                    prev_df, x="prevention_value_estimate", nbins=40,
                    title="Prevention Value Distribution",
                    color_discrete_sequence=["#34d399"],
                )
                fig5.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,50,0.3)",
                    font_color="#c5c5ff",
                    title_font_size=15,
                )
                st.plotly_chart(fig5, use_container_width=True)

    st.subheader("At-Risk Financial Detail")
    show_cols = [
        c for c in [
            "priority", "assigned_to", "reassigned_to", "probability",
            "penalty_cost", "financial_exposure_ev",
            "prevention_value_estimate", "automation_action", "escalation_tier",
        ] if c in at_risk.columns
    ]
    if show_cols and len(at_risk):
        st.dataframe(
            at_risk[show_cols].sort_values("financial_exposure_ev", ascending=False),
            use_container_width=True,
            height=420,
        )

# ─────── Tab: Automation Queue ────────────────────────────────────────

with tab_queue:
    st.markdown(
        "Queued **REASSIGN**, **ESCALATE**, and **RESOURCE** pathways computed via "
        "load-aware pool simulation."
    )
    action_vals = sorted(df["automation_action"].dropna().unique().tolist())
    if len(at_risk) and action_vals:
        default_q = sorted(at_risk["automation_action"].dropna().unique().tolist())[:5]
    else:
        default_q = []

    q = st.multiselect(
        "Filter automation action",
        options=action_vals,
        default=[x for x in default_q if x in action_vals],
    )
    show = at_risk[at_risk["automation_action"].isin(q)] if q else at_risk
    st.dataframe(show, use_container_width=True, height=460)

    csv_data = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Automation Queue (CSV)",
        data=csv_data,
        file_name=f"hackerthorn_queue_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# ─────── Tab: Agent Insights ──────────────────────────────────────────

with tab_agents:
    st.markdown(
        "**Multi-agent GenAI:** Reason → Action → Escalation specialists. "
        "Top at-risk rows may use the API; others use fast heuristics."
    )

    if not agent_detail:
        st.info("No agent detail available. Run the pipeline first.")
    else:
        # Summary strip
        llm_cnt = sum(1 for d in agent_detail if d["mode"] == "llm")
        heur_cnt = sum(1 for d in agent_detail if d["mode"] == "heuristic")
        avg_ms = np.mean([d["elapsed_ms"] for d in agent_detail]) if agent_detail else 0

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("LLM Agent Calls", f"{llm_cnt:,}")
        sc2.metric("Heuristic Calls", f"{heur_cnt:,}")
        sc3.metric("Avg Latency", f"{avg_ms:.1f} ms")

        st.markdown("---")

        for entry in agent_detail[:30]:
            prob = entry["probability"]
            badge_mode = (
                '<span class="agent-badge badge-llm">LLM</span>'
                if entry["mode"] == "llm"
                else '<span class="agent-badge badge-heuristic">HEUR</span>'
            )
            title = (
                f"{entry['priority']} · {entry['assigned_to']} · "
                f"p={prob:.2f} · {entry['elapsed_ms']}ms"
            )
            with st.expander(title):
                st.markdown(badge_mode, unsafe_allow_html=True)
                st.markdown(f"<span class='agent-badge badge-reason'>Reason</span> {entry['reason']}", unsafe_allow_html=True)
                st.markdown(f"<span class='agent-badge badge-action'>Action</span> {entry['action']}", unsafe_allow_html=True)
                st.markdown(f"<span class='agent-badge badge-escalation'>Escalation</span> {entry['escalation']}", unsafe_allow_html=True)

# ─────── Tab: Execution Logs ──────────────────────────────────────────

with tab_logs:
    st.markdown(
        """
        <div style="margin-bottom:16px;">
            <h3 style="color:#c5c5ff; margin:0; font-weight:700;">
                📋 Agent Execution Logs
            </h3>
            <p style="color:#7070a0; font-size:0.88rem; margin:4px 0 0 0;">
                Step-by-step trace of the analysis pipeline — ML scoring, automation decisions, and GenAI agent calls.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not logs:
        st.info("No logs yet. Run the pipeline to see execution traces.")
    else:
        # Render logs with visual styling
        log_html_parts = []
        for line in logs:
            # Color-code by prefix
            if "✅" in line:
                color = "#34d399"
            elif "❌" in line or "ERROR" in line.upper():
                color = "#f87171"
            elif "⚙️" in line or "🔄" in line:
                color = "#fbbf24"
            elif "🌲" in line:
                color = "#4ade80"
            elif "📊" in line:
                color = "#818cf8"
            elif "🤖" in line:
                color = "#38bdf8"
            elif "🧠" in line:
                color = "#c084fc"
            elif "💰" in line:
                color = "#facc15"
            elif "[DATA]" in line:
                color = "#94a3b8"
            else:
                color = "#a0a0c0"

            log_html_parts.append(
                f'<div style="padding:4px 12px; font-family:\'JetBrains Mono\',monospace; '
                f'font-size:0.82rem; color:{color}; border-left: 3px solid {color}30; '
                f'margin:2px 0; background:rgba(20,20,50,0.4); border-radius:0 6px 6px 0;">'
                f'{line}</div>'
            )

        st.markdown(
            f'<div style="background:rgba(10,10,30,0.6); border-radius:12px; padding:16px; '
            f'border:1px solid rgba(100,100,255,0.1); max-height:600px; overflow-y:auto;">'
            f'{"".join(log_html_parts)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Also offer raw download
        raw_logs = "\n".join(logs)
        st.download_button(
            "📥 Download Logs (.txt)",
            data=raw_logs.encode("utf-8"),
            file_name=f"hackerthorn_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )

# ─────── Footer ───────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center; padding:32px 0 16px 0; border-top:1px solid rgba(100,100,255,0.08); margin-top:40px;">
        <p style="color:#5050a0; font-size:0.82rem; margin:0;">
            <b>Hackerthorn</b> — AI that initiates corrective action with quantified financial framing.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
