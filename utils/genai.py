"""
Multi-agent GenAI layer: Reason → Action → Escalation.
Each step uses a specialist system prompt; outputs are merged for ops + audit.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL


def _client() -> OpenAI | None:
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def _chat(system: str, user: str, max_tokens: int = 220) -> str:
    cl = _client()
    if cl is None:
        return ""
    try:
        res = cl.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.35,
            max_tokens=max_tokens,
        )
        return (res.choices[0].message.content or "").strip()
    except Exception:
        return ""


REASON_SYSTEM = (
    "You are the Reason Agent for enterprise SLA intelligence. "
    "Given operational fields, explain why the task is or is not at risk. "
    "Be concise (2-4 sentences). No markdown headers."
)

ACTION_SYSTEM = (
    "You are the Action Agent. Propose concrete corrective steps: "
    "reroute work, compress scope, add capacity, or sequence dependencies. "
    "Number 1-3 steps, imperative voice, each under 18 words."
)

ESCALATION_SYSTEM = (
    "You are the Escalation Agent. State who to notify (role, not names), "
    "by when urgency applies, and whether exec escalation is warranted. "
    "Two short sentences maximum."
)


def heuristic_sla_pack(row) -> Dict[str, str]:
    p = float(row.get("probability", 0))
    pri = row.get("priority", "?")
    tr = row.get("time_remaining", "?")
    pc = row.get("penalty_cost", 0)
    aa = row.get("automation_action", "MONITOR")
    reason = (
        f"Breach risk near {p:.0%} with priority {pri} and {tr} units of time left; "
        f"penalty exposure reference ${pc}. Automation state: {aa}."
    )
    action = (
        "1) Reassign to lowest-load qualified owner. "
        "2) Trim non-critical scope to protect committed SLA. "
        "3) Daily checkpoint until buffer is restored."
    )
    esc = (
        "Notify operations lead within 2 hours if probability stays above 70%. "
        "Executive escalation only if capacity cannot be freed same business day."
    )
    if p >= 0.88:
        esc = (
            "Notify director-tier owner within 30 minutes. "
            "Exec escalation if financial exposure cannot be capped today."
        )
    return {"reason": reason, "action": action, "escalation": esc, "mode": "heuristic"}


def multi_agent_sla_pack(row) -> Dict[str, str]:
    """
    Run three specialist calls (or heuristic fallback) and return a dict
    suitable for JSON serialization to CSV.
    """
    if os.environ.get("SKIP_GENAI"):
        return heuristic_sla_pack(row)

    ctx = f"""Task snapshot:
- priority: {row.get("priority")}
- estimated_time: {row.get("estimated_time")}
- time_remaining: {row.get("time_remaining")}
- assigned_to: {row.get("assigned_to")}
- reassigned_to: {row.get("reassigned_to")}
- breach_probability: {float(row.get("probability", 0)):.3f}
- penalty_cost: {row.get("penalty_cost")}
- automation_action: {row.get("automation_action")}
- escalation_tier: {row.get("escalation_tier")}
"""
    reason = _chat(REASON_SYSTEM, ctx)
    action = _chat(ACTION_SYSTEM, f"Context:\n{ctx}\nReason agent said:\n{reason}")
    escalation = _chat(
        ESCALATION_SYSTEM,
        f"Context:\n{ctx}\nReason:\n{reason}\nPlanned actions:\n{action}",
        max_tokens=160,
    )

    if not reason and not action and not escalation:
        h = heuristic_sla_pack(row)
        h["mode"] = "heuristic_fallback"
        return h

    fb = heuristic_sla_pack(row)
    return {
        "reason": reason or fb["reason"],
        "action": action or fb["action"],
        "escalation": escalation or fb["escalation"],
        "agent_mode": "openai_multi_agent",
    }


def pack_to_column_value(pack: Dict[str, Any]) -> str:
    try:
        return json.dumps(pack, ensure_ascii=False)
    except Exception:
        return str(pack)


def generate_insight(row) -> str:
    """Backward-compatible single string for consumers expecting one blob."""
    p = multi_agent_sla_pack(row)
    return (
        f"[Reason] {p.get('reason','')}\n"
        f"[Action] {p.get('action','')}\n"
        f"[Escalation] {p.get('escalation','')}"
    )
