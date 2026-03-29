"""
SLA prevention automation: reassign, resource shifts, escalation tiers,
and quantified financial exposure for enterprise operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from config import (
    AUTOMATION_ESCALATE_L1_PROB,
    AUTOMATION_ESCALATE_L2_PROB,
    AUTOMATION_REASSIGN_PROB_MIN,
    EMPLOYEE_LOAD_CAP,
)


@dataclass
class EmployeePool:
    """Mutable pool for simulated load-aware reassignment."""

    cap: int = EMPLOYEE_LOAD_CAP

    def __post_init__(self) -> None:
        self.df = pd.DataFrame(
            {
                "employee": ["A", "B", "C", "D"],
                "skill": [5, 4, 3, 5],
                "load": [3, 6, 8, 2],
            }
        )

    def _available(self) -> pd.DataFrame:
        return self.df[self.df["load"] < self.cap]

    def pick_reassignee(
        self, current: str, min_skill: int = 3
    ) -> Tuple[Optional[str], str]:
        """
        Choose best alternate owner by skill desc, load asc.
        Returns (employee_code or None, detail_note).
        """
        pool = self._available()
        pool = pool[pool["skill"] >= min_skill]
        pool = pool[pool["employee"].astype(str) != str(current)]
        if pool.empty:
            return None, "no_capacity_in_pool"
        best = pool.sort_values(["skill", "load"], ascending=[False, True]).iloc[0]
        return str(best["employee"]), "reassigned_within_pool"

    def apply_shift(self, to_employee: str, units: int = 1) -> None:
        to_employee = str(to_employee)
        mask = self.df["employee"].astype(str) == to_employee
        if mask.any():
            idx = self.df.index[mask][0]
            self.df.at[idx, "load"] = min(
                self.cap, float(self.df.at[idx, "load"]) + units
            )


def expected_financial_exposure(probability: float, penalty_cost: float) -> float:
    """Expected penalty-weighted exposure for a predicted-at-risk task."""
    return float(max(probability, 0.0)) * float(max(penalty_cost, 0.0))


def prevention_value_if_action_succeeds(
    probability: float, penalty_cost: float, confidence_discount: float = 1.0
) -> float:
    """
    Upper-bound style 'prevented' $ if corrective action avoids the breach,
    scaled by model confidence.
    """
    return expected_financial_exposure(probability, penalty_cost) * float(
        confidence_discount
    )


def decide_automation_action(
    row: pd.Series,
    reassigned_to: str,
    *,
    had_pool_capacity: bool,
) -> Tuple[str, str]:
    """
    Map model output + reassignment outcome to an automation verb and tier.
    Returns (automation_action, escalation_tier).
    """
    if int(row.get("predicted_breach", 0)) != 1:
        return "MONITOR", "T0"

    prob = float(row.get("probability", 0))
    if prob >= AUTOMATION_ESCALATE_L2_PROB:
        tier = "T2"
    elif prob >= AUTOMATION_ESCALATE_L1_PROB:
        tier = "T1"
    else:
        tier = "T0"

    if prob < AUTOMATION_REASSIGN_PROB_MIN:
        return "MONITOR", "T0"

    if not had_pool_capacity or reassigned_to == row.get("assigned_to"):
        if tier == "T2":
            return "ESCALATE_EXEC", tier
        if tier == "T1":
            return "ESCALATE_OPS", tier
        return "RESOURCE_REQUEST", tier

    if tier == "T2":
        return "REASSIGN_AND_ESCALATE", tier
    if tier == "T1":
        return "REASSIGN_NOTIFY_LEAD", tier
    return "REASSIGN", tier


def apply_automation_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds reassigned_to, automation_action, escalation_tier, had_pool_capacity,
    financial_exposure_ev, prevention_value_estimate, cost_leakage_flag.
    """
    out = df.copy()
    pool = EmployeePool()
    reassigned = []
    actions = []
    tiers = []
    had_cap = []
    prev_assign = []

    for _, row in out.iterrows():
        cur = str(row.get("assigned_to", ""))
        if int(row.get("predicted_breach", 0)) != 1:
            reassigned.append(cur)
            actions.append("MONITOR")
            tiers.append("T0")
            had_cap.append(True)
            prev_assign.append("")
            continue

        candidate, _note = pool.pick_reassignee(cur)
        ok = candidate is not None
        new_assign = candidate if ok else cur
        if ok and candidate:
            pool.apply_shift(candidate, 1)

        action, tier = decide_automation_action(
            row,
            new_assign,
            had_pool_capacity=ok,
        )
        reassigned.append(new_assign)
        actions.append(action)
        tiers.append(tier)
        had_cap.append(ok)
        prev_assign.append(cur)

    out["reassigned_to"] = reassigned
    out["automation_action"] = actions
    out["escalation_tier"] = tiers
    out["had_pool_capacity"] = had_cap
    out["previous_assignee"] = prev_assign

    penalties = pd.to_numeric(out.get("penalty_cost", 0), errors="coerce").fillna(0)
    probs = pd.to_numeric(out.get("probability", 0), errors="coerce").fillna(0)
    out["financial_exposure_ev"] = probs * penalties
    out["prevention_value_estimate"] = out.apply(
        lambda r: prevention_value_if_action_succeeds(
            float(r["probability"]),
            float(r.get("penalty_cost", 0) or 0),
            0.85 if r["automation_action"] != "MONITOR" else 0.0,
        ),
        axis=1,
    )
    breach = pd.to_numeric(out.get("breach", 0), errors="coerce").fillna(0)
    pred = pd.to_numeric(out.get("predicted_breach", 0), errors="coerce").fillna(0)
    out["cost_leakage_flag"] = ((breach == 1) | (pred == 1)).astype(int)

    return out
