import base64

from atlas.ingest.gmail_json import body_text, to_row
from eval.private_eval import evidence_hit


def part(mime, text, **extra):
    return {"mime_type": mime, "body": {"content": text}, **extra}


def test_alternatives_and_attachments():
    payload = {"mime_type": "multipart/mixed", "parts": [
        {"mime_type": "multipart/alternative", "parts": [part("text/html", "<p>duplicate</p>"),
                                                               part("text/plain", "actual body")]},
        part("text/plain", "private attachment", filename="notes.txt"),
        part("text/plain", "attachment without filename", headers=[{"name": "Content-Disposition", "value": "attachment"}])
    ]}
    assert body_text(payload) == "actual body"


def test_html_fallback_and_api_base64():
    assert body_text({"mime_type": "multipart/alternative", "parts": [
        part("text/plain", ""), part("text/html", "<p>Room 205</p>")]}) .strip() == "Room 205"
    encoded = base64.urlsafe_b64encode("Café".encode()).decode().rstrip("=")
    message = {"id": "one", "threadId": "thread", "internalDate": "1000000", "labelIds": ["INBOX"],
               "payload": {"mimeType": "text/plain", "body": {"data": encoded},
                           "headers": [{"name": "Subject", "value": "Receipt"}]}}
    row = to_row(message)
    assert row["body"] == "Café" and row["date"] == 1000 and row["thread_id"] == "thread"


def test_evidence_requires_target_and_all_facts():
    item = {"targets": ["one"], "evidence": [r"Room 205", r"8am"]}
    assert evidence_hit("Room 205 at 8am", ["one"], item)
    assert not evidence_hit("Room 205", ["one"], item)
    assert not evidence_hit("Room 205 at 8am", ["other"], item)
