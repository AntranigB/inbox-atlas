"""Shared helpers for the training pipeline: paths, jsonl io, text cleaning, doc rendering."""

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("ATLAS_DATA", ROOT / "data"))
DATASETS = DATA / "datasets"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def sha(s, n=16):
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:n]


_PREFIX = re.compile(r"^\s*((re|fw|fwd|aw)\s*(\[\d+\])?\s*:\s*)+", re.I)


def norm_subject(s):
    s = _PREFIX.sub("", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def is_reply(s):
    return bool(re.match(r"^\s*re\s*:", s or "", re.I))


# markers after which the rest of a body is quoted / forwarded material
_CUT = [
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.I | re.M),
    re.compile(r"^-{2,}\s*Forwarded by", re.I | re.M),
    re.compile(r"^\s*-{5,}\s*Forwarded by", re.I | re.M),
    re.compile(r"^From:\s.*\n(Sent|Date|To):\s", re.I | re.M),
    re.compile(r"^.{1,80}\son\s\d{1,2}/\d{1,2}/\d{2,4}\s\d{1,2}:\d{2}(:\d{2})?\s?[AP]M\s*$", re.M),
    re.compile(r"^\s*On .{5,120} wrote:\s*$", re.M),
    re.compile(r"^>", re.M),
]
_HDR = re.compile(r"^\s*(From|To|cc|bcc|Sent|Date|Subject)\s*:.*$", re.I | re.M)
_SIG = re.compile(r"\n--\s*\n")


def clean_body(text, max_chars=4000):
    """Strip quoted replies, forward headers and signatures. If the author wrote almost nothing
    above a forward, keep the forwarded content with its header lines removed."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cut, fwd = len(text), False
    for i, rx in enumerate(_CUT):
        m = rx.search(text)
        if m and m.start() < cut:
            cut, fwd = m.start(), i in (1, 2)
    head = text[:cut]
    m = _SIG.search(head)
    if m:
        head = head[: m.start()]
    if (fwd and len(head.strip()) < 40) or len(head.strip()) < 5:
        rest = text[cut:]
        rest = "\n".join(l for l in rest.split("\n") if not l.lstrip().startswith(">"))
        rest = _HDR.sub("", rest)
        rest = re.sub(r"^.*Forwarded by.*$|^.*Original Message.*$", "", rest, flags=re.M | re.I)
        head = head + "\n" + rest
    head = re.sub(r"[ \t]+", " ", head)
    head = re.sub(r"\n\s*\n+", "\n\n", head).strip()
    return head[:max_chars]


def doc_text(r, mode="full"):
    """Embedding text for an email, matching CONTRACT.md. mode='nosubj' drops the subject
    (used for subject -> body pairs so the task is not a string match)."""
    body = (r.get("body") or "")[:2000]
    if mode == "nosubj":
        return f"{r.get('from_name') or ''}\n{body}"
    return f"{r.get('subject') or ''}\n{r.get('from_name') or ''}\n{body}"
