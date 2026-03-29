"""
Hackerthorn SLA + cost-leakage prevention pipeline:
ML scoring → automation (reassign / escalate) → multi-agent GenAI → Slack/email digest.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.alerts import send_automation_email_digest, send_slack_digest
from utils.automation import apply_automation_to_dataframe
from utils.feature_engineering import engineer
from utils.genai import heuristic_sla_pack, multi_agent_sla_pack, pack_to_column_value

ROOT = Path(__file__).resolve().parent


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


def main() -> None:
    df_train = pd.read_csv(ROOT / "data" / "train.csv")
    df_test = pd.read_csv(ROOT / "data" / "test.csv")

    df_train["breach"] = df_train["breach"].astype(int)
    df_test["breach"] = df_test["breach"].astype(int)

    df_train = engineer(df_train)
    df_test = engineer(df_test)

    features = [
        "priority_score",
        "estimated_time",
        "time_remaining",
        "completion_percentage",
        "time_ratio",
        "urgency_score",
        "risk_score",
        "tight_deadline",
        "very_tight",
        "assigned_to",
    ]

    X_train = df_train[features].copy()
    y_train = df_train["breach"]
    X_test = df_test[features].copy()

    le = LabelEncoder()
    X_train["assigned_to"] = le.fit_transform(X_train["assigned_to"].astype(str))
    X_test["assigned_to"] = X_test["assigned_to"].apply(lambda v: safe_encode(le, v))

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imp.fit_transform(X_train))
    X_test = scaler.transform(imp.transform(X_test))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    df_test["predicted_breach"] = y_pred
    df_test["probability"] = y_prob
    df_test["SLA_Status"] = df_test["predicted_breach"].map(
        {0: "SAFE", 1: "AT_RISK"}
    )

    df_test = apply_automation_to_dataframe(df_test)

    max_llm = int(os.environ.get("MAX_GENAI_ROWS", "40"))
    risk_mask = df_test["predicted_breach"] == 1
    top_llm_idx = (
        df_test.loc[risk_mask]
        .sort_values("probability", ascending=False)
        .head(max_llm)
        .index.tolist()
    )
    llm_set = set(top_llm_idx)

    packs: list[str] = []
    for idx, row in df_test.iterrows():
        if int(row["predicted_breach"]) != 1:
            packs.append("")
            continue
        if idx in llm_set and not os.environ.get("SKIP_GENAI"):
            packs.append(pack_to_column_value(multi_agent_sla_pack(row)))
        else:
            packs.append(pack_to_column_value(heuristic_sla_pack(row)))

    df_test["AI_MultiAgent"] = packs
    df_test["AI_Insight"] = df_test["AI_MultiAgent"].map(format_insight_from_json)

    at_risk = df_test[df_test["predicted_breach"] == 1]
    exposure = float(df_test["financial_exposure_ev"].sum())
    prevention = float(df_test["prevention_value_estimate"].sum())
    esc_count = int((at_risk["escalation_tier"].isin(["T1", "T2"])).sum())

    if not os.environ.get("SKIP_SLACK") and len(at_risk) > 0:
        top_row = at_risk.sort_values("probability", ascending=False).head(1)
        top_ops = str(top_row["assigned_to"].iloc[0]) if len(top_row) else None
        send_slack_digest(
            at_risk_count=int(len(at_risk)),
            total_exposure=exposure,
            total_prevention_estimate=prevention,
            escalate_count=esc_count,
            top_ops_id=top_ops,
        )

    if not os.environ.get("SKIP_EMAIL") and len(at_risk) > 0:
        br = at_risk["automation_action"].value_counts().to_string()
        send_automation_email_digest(
            at_risk_count=int(len(at_risk)),
            total_exposure=exposure,
            total_prevention_estimate=prevention,
            action_breakdown=br,
        )

    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    df_test.to_csv(ROOT / "outputs" / "results.csv", index=False)

    pickle.dump(
        model,
        open(ROOT / "models" / "model.pkl", "wb"),
    )
    pickle.dump(
        {"le": le, "imp": imp, "scaler": scaler, "features": features},
        open(ROOT / "models" / "artifacts.pkl", "wb"),
    )

    manifest = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_scored": int(len(df_test)),
        "predicted_at_risk": int((df_test["predicted_breach"] == 1).sum()),
        "total_financial_exposure_ev": exposure,
        "total_prevention_value_estimate": prevention,
        "escalation_rows_t1_t2": esc_count,
        "automation_mix": at_risk["automation_action"]
        .value_counts()
        .to_dict()
        if len(at_risk)
        else {},
        "genai_llm_rows": 0
        if os.environ.get("SKIP_GENAI")
        else int(len(llm_set)),
        "notes": "EV = probability * penalty_cost per row; prevention value is confidence-weighted heuristic.",
    }
    (ROOT / "outputs" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Pipeline complete.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
