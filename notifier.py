# notifier.py
import os, json, urllib.request

def _post_webhook(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()

def notify_slack(text: str):
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        print("[notify] SLACK_WEBHOOK_URL not set; message below:\n", text)
        return
    payload = {"text": text}
    try:
        _post_webhook(url, payload)
    except Exception as e:
        print("[notify] Slack post failed:", e, "\nMessage was:\n", text)
