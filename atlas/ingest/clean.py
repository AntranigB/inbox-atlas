"""Turn raw RFC822 bytes into a clean `emails` row.

Body cleaning strips HTML, quoted replies, forwarded header blocks, signatures, tracking junk
and long URLs, so that replies in a thread do not all embed to the same point.
"""

import hashlib
import json
import re
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime

from bs4 import BeautifulSoup

# lines that start a quoted reply block; everything after is dropped
_CUT_PATTERNS = [
    re.compile(r"^\s*On\s.{0,300}?wrote:\s*$", re.I | re.S),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}", re.I),
    re.compile(r"^\s*-{2,}\s*Forwarded message\s*-{2,}", re.I),
    re.compile(r"^\s*Begin forwarded message:", re.I),
    re.compile(r"^\s*_{10,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.I),  # outlook style header block (only when followed by Sent:/To:)
    re.compile(r"^\s*-- \s*$|^\s*--\s*$"),  # signature delimiter
    re.compile(r"^\s*Sent from my (iPhone|iPad|Android|Galaxy|mobile device|BlackBerry)", re.I),
    re.compile(r"^\s*Get Outlook for (iOS|Android)", re.I),
]
_ON_WROTE_START = re.compile(r"^\s*On\s.*(\d{4}|[AP]M|at\s)", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+")
_JUNK_LINE = re.compile(
    r"(unsubscribe|view (this |it )?(email |message )?(in|on) (your|a) (web )?browser|view (it |this )?online|update (your )?preferences|"
    r"manage (your )?(email )?(subscriptions|preferences)|you are receiving this|"
    r"this email was sent to|privacy policy|all rights reserved|©|copyright \d{4})", re.I)
_ZERO_WIDTH = re.compile(r"[​-‏  ­͏﻿⁠-⁤]")


def decode_hdr(v):
    if v is None:
        return ""
    try:
        return str(make_header(decode_header(str(v)))).strip()
    except Exception:
        return str(v).strip()


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "head", "title", "meta", "noscript"]):
        t.decompose()
    for t in soup.find_all("blockquote"):
        t.decompose()
    for cls in ("gmail_quote", "gmail_signature", "yahoo_quoted", "moz-cite-prefix"):
        for t in soup.find_all(class_=cls):
            t.decompose()
    for br in soup.find_all(["br", "p", "div", "tr", "li", "h1", "h2", "h3"]):
        br.insert_after("\n")
    return soup.get_text()


def _shorten_url(m):
    u = m.group(0)
    if len(u) <= 40:
        return u
    dom = re.sub(r"^(https?://)?(www\.)?", "", u).split("/")[0]
    return f"[{dom}]"


def clean_body(text):
    if not text:
        return ""
    text = _ZERO_WIDTH.sub("", text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " "))
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(">"):
            continue
        # multi-line "On ... \n ... wrote:" (Gmail wraps long attributions)
        if _ON_WROTE_START.match(line):
            joined = " ".join(l.strip() for l in lines[i:i + 3])
            if re.search(r"wrote:", joined, re.I):
                break
        cut = False
        for j, p in enumerate(_CUT_PATTERNS):
            if p.search(line):
                if j == 5:  # From: only cuts when it is an outlook header block
                    nxt = " ".join(lines[i + 1:i + 4])
                    if not re.search(r"^\s*(Sent|Date|To|Subject):", nxt, re.I | re.M):
                        continue
                cut = True
                break
        if cut:
            break
        if _JUNK_LINE.search(s) and len(s) < 300:
            continue
        out.append(line)
    text = "\n".join(out)
    text = _URL.sub(_shorten_url, text)
    text = re.sub(r"\[(?:image|cid):[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _part_text(part):
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        return payload.decode(part.get_content_charset() or "utf-8", errors="replace")


def extract_text(msg):
    """Prefer text/plain, else HTML converted to text. Skips attachments."""
    plain, html = None, None
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        ct = part.get_content_type()
        if ct == "text/plain" and plain is None:
            plain = _part_text(part)
        elif ct == "text/html" and html is None:
            html = _part_text(part)
    if plain and plain.strip():
        # some senders put HTML in text/plain
        if re.search(r"<(html|body|div|table)\b", plain[:2000], re.I):
            return html_to_text(plain)
        return plain
    if html:
        return html_to_text(html)
    return ""


def norm_subject(s):
    return re.sub(r"^\s*((re|fwd?|aw|sv)\s*:\s*)+", "", s or "", flags=re.I).strip().lower()


def parse_message(raw, gm_msgid=None, gm_thrid=None, labels=None, source="gmail"):
    """Parse RFC822 bytes into an emails row dict (CONTRACT schema)."""
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    subject = decode_hdr(msg.get("Subject"))
    frm = getaddresses([decode_hdr(msg.get("From"))])
    from_name, from_addr = (frm[0] if frm else ("", ""))
    tos = getaddresses([decode_hdr(msg.get(h)) for h in ("To", "Cc") if msg.get(h)])
    try:
        date = int(parsedate_to_datetime(msg.get("Date")).timestamp())
    except Exception:
        date = 0
    mid = (msg.get("Message-ID") or "").strip()
    eid = str(gm_msgid) if gm_msgid else hashlib.sha1((mid or raw[:4096].hex()).encode()).hexdigest()[:16]
    tid = str(gm_thrid) if gm_thrid else hashlib.sha1(norm_subject(subject).encode()).hexdigest()[:16]
    body = clean_body(extract_text(msg))
    return {
        "id": eid, "thread_id": tid,
        "from_addr": (from_addr or "").lower(), "from_name": from_name or from_addr or "",
        "to_addrs": ", ".join(a for _, a in tos if a),
        "date": date, "subject": subject, "body": body,
        "snippet": re.sub(r"\s+", " ", body)[:200],
        "labels": json.dumps(list(labels or [])), "source": source,
    }
