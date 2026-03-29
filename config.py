"""
Hackerthorn — configuration from environment only.
Copy .env.example to .env and fill values locally (never commit .env).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default


# GenAI
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")

# Slack
SLACK_WEBHOOK = _env("SLACK_WEBHOOK")

# Email (SMTP; Gmail uses app password)
EMAIL_SMTP_HOST = _env("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(_env("EMAIL_SMTP_PORT", "587"))
EMAIL_FROM = _env("EMAIL_FROM")
EMAIL_TO = _env("EMAIL_TO") or EMAIL_FROM
EMAIL_USERNAME = _env("EMAIL_USERNAME") or EMAIL_FROM
EMAIL_PASSWORD = _env("EMAIL_PASSWORD")

# Automation thresholds (optional overrides)
AUTOMATION_REASSIGN_PROB_MIN = float(_env("AUTOMATION_REASSIGN_PROB_MIN", "0.55"))
AUTOMATION_ESCALATE_L1_PROB = float(_env("AUTOMATION_ESCALATE_L1_PROB", "0.72"))
AUTOMATION_ESCALATE_L2_PROB = float(_env("AUTOMATION_ESCALATE_L2_PROB", "0.88"))
EMPLOYEE_LOAD_CAP = int(_env("EMPLOYEE_LOAD_CAP", "10"))
