"""
Hackerthorn — real-time SLA & cost-leakage control room (Streamlit).
Run: streamlit run dashbord.py  (from project root)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "results.csv"
MANIFEST = ROOT / "outputs" / "run_manifest.json"


try:
    from streamlit_autorefresh import st_autorefresh

    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False


st.set_page_config(
    page_title="Hackerthorn — SLA Control Room",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_results() -> pd.DataFrame:
    if not RESULTS.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS)


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_agent_cell(raw: str) -> dict | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


# --- Sidebar ---
st.sidebar.title("Control room")
refresh_sec = st.sidebar.slider(
    "Auto-refresh interval (seconds)", min_value=0, max_value=120, value=20
)
if _HAS_AUTOREFRESH and refresh_sec > 0:
    st_autorefresh(interval=refresh_sec * 1000, key="control_room_refresh")
else:
    st.sidebar.caption(
        "Install `streamlit-autorefresh` for timed refresh, or use the button below."
    )

if st.sidebar.button("Refresh data now", type="primary"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Continuous monitoring:** schedule `python app.py` (cron, Airflow, "
    "Azure Logic Apps) so this dashboard reflects fresh operational scoring."
)

# --- Main ---
st.title("Hackerthorn SLA & cost-leakage control room")
st.caption(
    "Beyond static charts: automation actions, financial exposure, and multi-agent "
    "explanations tied to real operational signals."
)

df = load_results()
m = load_manifest()

if df.empty:
    st.warning(
        f"No results at `{RESULTS}`. Run `python app.py` from the project root first."
    )
    st.stop()

at_risk = df[df["predicted_breach"] == 1].copy()
safe_n = int((df["predicted_breach"] == 0).sum())
risk_n = len(at_risk)

exposure = float(pd.to_numeric(df.get("financial_exposure_ev", 0), errors="coerce").fillna(0).sum())
prevention = float(
    pd.to_numeric(df.get("prevention_value_estimate", 0), errors="coerce")
    .fillna(0)
    .sum()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tasks SAFE", f"{safe_n:,}")
c2.metric("Tasks AT RISK (predicted)", f"{risk_n:,}")
c3.metric("Expected penalty exposure (EV)", f"${exposure:,.0f}")
c4.metric("Est. prevention value", f"${prevention:,.0f}")

if m:
    st.success(
        f"Last pipeline run (UTC): **{m.get('run_at_utc', '?')}** · "
        f"LLM-deep rows: **{m.get('genai_llm_rows', 0)}**"
    )

tab_a, tab_b, tab_c, tab_d = st.tabs(
    ["Overview", "Financial & leakage", "Automation queue", "Multi-agent insights"]
)

with tab_a:
    st.subheader("Risk concentration")
    colx, coly = st.columns(2)
    with colx:
        ap = at_risk["automation_action"].value_counts().reset_index()
        ap.columns = ["action", "count"]
        fig = px.bar(ap, x="action", y="count", title="Automation mix (at-risk only)")
        st.plotly_chart(fig, use_container_width=True)
    with coly:
        if "assigned_to" in at_risk.columns:
            owners = at_risk.groupby("assigned_to").size().reset_index(name="at_risk")
            owners = owners.sort_values("at_risk", ascending=False).head(12)
            fig2 = px.bar(
                owners,
                x="assigned_to",
                y="at_risk",
                title="At-risk workload by current owner (top 12)",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Probability vs penalty (at-risk tasks)")
    if len(at_risk):
        scat = at_risk.copy()
        scat["penalty_cost"] = pd.to_numeric(scat["penalty_cost"], errors="coerce")
        fig3 = px.scatter(
            scat,
            x="probability",
            y="penalty_cost",
            color="automation_action",
            hover_data=["priority", "time_remaining", "assigned_to", "reassigned_to"],
            title="Model confidence × contractual penalty",
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab_b:
    st.markdown(
        "**Financial lens:** Each row carries `penalty_cost` from operations. "
        "We surface **expected exposure** as roughly *probability × penalty* and a "
        "**prevention heuristic** when automation fires (see README for definitions)."
    )
    if "financial_exposure_ev" in df.columns:
        hist = df[df["predicted_breach"] == 1]
        if len(hist):
            fig4 = px.histogram(
                hist,
                x="financial_exposure_ev",
                nbins=40,
                title="Distribution of expected exposure (at-risk rows)",
            )
            st.plotly_chart(fig4, use_container_width=True)

    leak = int(df["cost_leakage_flag"].sum()) if "cost_leakage_flag" in df.columns else None
    if leak is not None:
        st.metric("Rows flagged for cost / SLA leakage signal", f"{leak:,}")

    st.dataframe(
        at_risk[
            [
                c
                for c in [
                    "priority",
                    "assigned_to",
                    "reassigned_to",
                    "probability",
                    "penalty_cost",
                    "financial_exposure_ev",
                    "prevention_value_estimate",
                    "automation_action",
                    "escalation_tier",
                ]
                if c in at_risk.columns
            ]
        ].sort_values("financial_exposure_ev", ascending=False),
        use_container_width=True,
        height=420,
    )

with tab_c:
    st.markdown(
        "Queued **REASSIGN**, **ESCALATE**, and **RESOURCE** pathways computed in "
        "`utils/automation.py` using live load simulation over a skill/tempo pool."
    )
    action_vals = sorted(df["automation_action"].dropna().unique().tolist())
    if len(at_risk) and action_vals:
        default_q = sorted(at_risk["automation_action"].dropna().unique().tolist())[
            : min(5, len(action_vals))
        ]
    else:
        default_q = []
    q = st.multiselect(
        "Filter automation action",
        options=action_vals,
        default=[x for x in default_q if x in action_vals],
    )
    show = at_risk[at_risk["automation_action"].isin(q)] if q else at_risk
    st.dataframe(show, use_container_width=True, height=460)

    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered automation queue (CSV)",
        data=csv,
        file_name=f"hackerthorn_automation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

with tab_d:
    st.markdown(
        "**Multi-agent GenAI:** Reason → Action → Escalation specialists (`utils/genai.py`). "
        "Top at-risk rows may use the API; others use fast heuristics so the demo stays stable."
    )
    detail = at_risk[at_risk["AI_MultiAgent"].notna() & (at_risk["AI_MultiAgent"] != "")]
    for _, r in detail.head(25).iterrows():
        pack = parse_agent_cell(str(r.get("AI_MultiAgent", "")))
        title = f"{r.get('priority','?')} · {r.get('assigned_to','?')} → {r.get('reassigned_to','?')} · p={float(r.get('probability',0)):.2f}"
        with st.expander(title):
            if pack:
                st.markdown(f"**Reason:** {pack.get('reason','')}")
                st.markdown(f"**Action:** {pack.get('action','')}")
                st.markdown(f"**Escalation:** {pack.get('escalation','')}")
                st.caption(f"Mode: {pack.get('agent_mode') or pack.get('mode','')}")
            else:
                st.text(r.get("AI_Insight", ""))

st.markdown("---")
st.caption("Hackerthorn — AI that initiates corrective action with quantified financial framing.")
