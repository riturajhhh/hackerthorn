"""
Notifications: Slack (digest + detail) and optional SMTP email for SLA automation.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import requests

from config import (
    EMAIL_FROM,
    EMAIL_PASSWORD,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PORT,
    EMAIL_TO,
    EMAIL_USERNAME,
    SLACK_WEBHOOK,
)


def send_slack(text: str) -> None:
    if not SLACK_WEBHOOK:
        return
    try:
        requests.post(SLACK_WEBHOOK, json={"text": text}, timeout=15)
    except Exception:
        pass


def send_slack_digest(
    *,
    at_risk_count: int,
    total_exposure: float,
    total_prevention_estimate: float,
    escalate_count: int,
    top_ops_id: Optional[str] = None,
) -> None:
    if not SLACK_WEBHOOK or at_risk_count <= 0:
        return
    lines = [
        "*Hackerthorn SLA Control Room — automation digest*",
        f"• Tasks flagged for action: *{at_risk_count}*",
        f"• Expected penalty-weighted exposure: *${total_exposure:,.0f}*",
        f"• Estimated preventable value (if actions land): *${total_prevention_estimate:,.0f}*",
        f"• Escalation-tier rows: *{escalate_count}*",
    ]
    if top_ops_id:
        lines.append(f"• Highest-risk work queue: `{top_ops_id}`")
    send_slack("\n".join(lines))


def send_email(subject: str, body_text: str) -> None:
    if not (EMAIL_SMTP_HOST and EMAIL_FROM and EMAIL_TO and EMAIL_PASSWORD):
        return
    user = EMAIL_USERNAME or EMAIL_FROM
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    try:
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(user, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
    except Exception:
        pass


def send_automation_email_digest(
    *,
    at_risk_count: int,
    total_exposure: float,
    total_prevention_estimate: float,
    action_breakdown: str,
) -> None:
    if at_risk_count <= 0:
        return
    body = (
        "Hackerthorn SLA + Cost Leakage Digest\n"
        "=====================================\n\n"
        f"Tasks requiring automation attention: {at_risk_count}\n"
        f"Expected financial exposure (EV): ${total_exposure:,.2f}\n"
        f"Estimated prevention value (confidence-weighted): ${total_prevention_estimate:,.2f}\n\n"
        "Action mix:\n"
        f"{action_breakdown}\n\n"
        "This is an automated message from the Hackerthorn prevention pipeline.\n"
    )
    send_email("[Hackerthorn] SLA automation digest", body)
