"""Build data/datasets/personal/ from data/mail.sqlite (read only; raw data is never modified).

Same row format as the Enron dataset so every stage runs on it unchanged:
  topics  = user Gmail labels (system labels and categories dropped), humanized
  splits  = 15% of labels held out (test_topic), then by thread: 10% test_thread, 5% val, rest train

Doubles as the zero-shot transfer eval: models trained only on 2001 Enron mail, evaluated on a
2026 student's Gmail:  uv run python -m train.eval --ds personal --run eval-personal

usage: uv run python -m train.build_personal [--db data/mail.sqlite] [--min-label 5]
"""

import argparse
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict

from train.common import DATA, DATASETS, is_reply, norm_subject, sha, write_jsonl

SYSTEM = {"inbox", "sent", "important", "starred", "draft", "drafts", "spam", "trash", "unread", "chat",
          "all mail", "opened", "sent mail", "notes", "junk"}


def human(label):
    s = label.strip().strip("\\")
    if s.lower().startswith(("category_", "[gmail]", "[imap]")) or s.lower() in SYSTEM:
        return None
    s = re.sub(r"[_/\-.]+", " ", s).strip().lower()
    return s if len(s) >= 2 and s not in SYSTEM else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA / "mail.sqlite"))
    ap.add_argument("--min-label", type=int, default=5)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    a = ap.parse_args()

    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("select * from emails")]
    emails = []
    for r in rows:
        if len((r.get("body") or "").strip()) < 20 and not r.get("subject"):
            continue
        try:
            labs = json.loads(r.get("labels") or "[]")
        except Exception:
            labs = []
        emails.append({
            "id": r["id"], "thread_id": r.get("thread_id") or sha("t:" + norm_subject(r.get("subject") or "")),
            "subject": r.get("subject") or "", "from_name": r.get("from_name") or r.get("from_addr") or "",
            "body": (r.get("body") or "")[:4000], "date": r.get("date") or 0,
            "is_reply": is_reply(r.get("subject")), "_labs": [h for h in map(human, labs) if h],
        })
    cnt = Counter(l for e in emails for l in set(e["_labs"]))
    keep = sorted(l for l, c in cnt.items() if c >= a.min_label)
    ks = set(keep)
    for e in emails:
        e["topics"] = sorted(set(l for l in e.pop("_labs") if l in ks))
    by_thread = defaultdict(list)
    for e in emails:
        by_thread[e["thread_id"]].append(e)
    for th in by_thread.values():
        th.sort(key=lambda x: x["date"])
        for i, e in enumerate(th):
            e["parent_id"] = th[i - 1]["id"] if (e["is_reply"] and i > 0) else None
    labs = keep[:]
    random.Random(0).shuffle(labs)
    held = set(labs[: int(len(labs) * a.heldout_frac)])
    for tid, th in by_thread.items():
        if any(l in held for e in th for l in e["topics"]):
            sp = "test_topic"
        else:
            u = int(sha("split:" + tid), 16) % 1000 / 1000
            sp = "test_thread" if u < 0.10 else ("val" if u < 0.15 else "train")
        for e in th:
            e["split"] = sp
    out = DATASETS / "personal"
    write_jsonl(out / "emails.jsonl", emails)
    (out / "topics.json").write_text(json.dumps({l: {"n": cnt[l], "split": "heldout" if l in held else "train"} for l in keep}, indent=1))
    print(f"wrote {len(emails)} emails, {len(keep)} labels ({len(held)} held out), splits {dict(Counter(e['split'] for e in emails))}")


if __name__ == "__main__":
    main()
