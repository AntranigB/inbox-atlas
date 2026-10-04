"""Import read-only Gmail connector/API JSON exports without fetching attachments or calling an LLM.

    uv run python -m atlas.ingest.gmail_json messages-0.json messages-1.json

Accepts Gmail API camelCase and connector snake_case MIME trees. Raw exports should be
kept in the gitignored data/ directory. The caller chooses the database with ATLAS_DATA.
"""

from __future__ import annotations

import argparse
import base64
import json
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from atlas import store
from atlas.ingest.clean import clean_body, decode_hdr, html_to_text


def _text(part):
    body = part.get("body") or {}
    if body.get("content") is not None:
        return body["content"]
    data = body.get("data") or body.get("base64_url_content")
    if not data:
        return ""
    headers = {h["name"].lower(): h["value"] for h in part.get("headers", [])}
    from email.message import Message
    msg = Message()
    msg["Content-Type"] = headers.get("content-type", "text/plain; charset=utf-8")
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    return raw.decode(msg.get_content_charset() or "utf-8", errors="replace")


def body_text(payload):
    """Choose one alternative, concatenate mixed body parts, never read attachments."""
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    if payload.get("filename") or headers.get("content-disposition", "").lower().startswith("attachment"):
        return ""
    mime = payload.get("mimeType", payload.get("mime_type", ""))
    parts = payload.get("parts") or []
    if mime == "multipart/alternative":
        for p in parts:
            if p.get("mimeType", p.get("mime_type")) == "text/plain":
                text = body_text(p)
                if text.strip():
                    return text
        return next((text for p in parts if (text := body_text(p)).strip()), "")
    if parts:
        return "\n\n".join(text for p in parts if (text := body_text(p)).strip())
    if mime == "text/plain":
        return _text(payload)
    if mime == "text/html":
        return html_to_text(_text(payload))
    return ""


def to_row(message):
    payload = message.get("payload") or {}
    headers = {h["name"].lower(): decode_hdr(h["value"]) for h in payload.get("headers", [])}
    sender = getaddresses([headers.get("from", "")]) or [("", "")]
    date = message.get("internalDate", message.get("internal_date"))
    if date:
        date = int(date) // 1000
    else:
        try:
            date = int(parsedate_to_datetime(headers.get("date", "")).timestamp())
        except (ValueError, TypeError, OverflowError):
            date = 0
    body = clean_body(body_text(payload))
    return {"id": message["id"], "thread_id": message.get("threadId", message.get("thread_id", message["id"])),
            "from_addr": sender[0][1], "from_name": sender[0][0], "to_addrs": headers.get("to", ""),
            "date": date, "subject": headers.get("subject", ""), "body": body, "snippet": body[:240],
            "labels": json.dumps(message.get("labelIds", message.get("label_ids", []))), "source": "gmail"}


def read_rows(paths):
    rows = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        messages = data if isinstance(data, list) else data.get("responses", data.get("messages", [data]))
        for message in messages:
            row = to_row(message)
            rows[row["id"]] = row
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    rows = read_rows(args.paths)
    conn = store.connect()
    store.upsert_emails(conn, rows)
    conn.close()
    print(f"Imported {len(rows)} messages locally; no Gmail writes or model API calls.")


if __name__ == "__main__":
    main()
