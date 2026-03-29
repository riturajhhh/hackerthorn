import requests
from config import SLACK_WEBHOOK

def send_slack(msg):
    try:
        requests.post(SLACK_WEBHOOK, json={"text": msg})
    except:
        pass