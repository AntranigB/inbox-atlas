"""Download and parse the CMU Enron corpus into data/datasets/enron/.

Outputs:
  emails.jsonl  one row per deduped email: id, thread_id, user, topics (humanized folder labels),
                subject, from_name, body, date, is_reply, parent_id, split
  topics.json   {label: {"n": int, "split": "train"|"heldout"}}

Splits (never leak across):
  test_topic   every email carrying a held-out folder label (~15% of labels)
  test_thread  10% of the remaining threads
  val          5% of the remaining threads (model selection)
  train        the rest

usage: uv run python -m train.enron [--download] [--limit N]
"""

import argparse
import email
import email.utils
import json
import os
import random
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from multiprocessing import Pool

from train.common import DATA, DATASETS, clean_body, is_reply, norm_subject, sha, write_jsonl

URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
RAW = DATA / "raw" / "enron"
OUT = DATASETS / "enron"

GENERIC = {
    "inbox", "sent", "sent_items", "_sent_mail", "sent_mail", "deleted_items", "all_documents",
    "discussion_threads", "notes_inbox", "calendar", "contacts", "archiving", "straw", "to_do",
    "notes", "untitled", "misc", "miscellaneous", "old", "archive", "archives", "save", "saved",
    "general", "new", "junk", "junk_file", "junk_mail", "drafts", "outbox", "tasks", "journal",
    "e_mail_bin", "online_trading", "mail", "personal_folders", "folder", "other", "stuff",
    "temp", "important", "bin", "x", "a", "b", "c", "info", "items", "read", "keep", "done",
    "1", "2", "3", "4", "5",
}


BAD_TOKENS = {"sent", "saved", "save", "deleted", "old", "messages", "attachments", "misc", "inbox", "folder", "mail", "e", "mails"}


def good_label(h):
    toks = h.split()
    if len(h) < 3 or not toks:
        return False
    if sum(t.isdigit() for t in toks) * 2 >= len(toks):
        return False
    return not all(t in BAD_TOKENS or t.isdigit() for t in toks) and not (set(toks) & {"sent", "deleted", "saved"})


def humanize(folder):
    s = folder.lower().strip()
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / "enron.tar.gz"
    if not (RAW / "maildir").exists():
        if not tgz.exists():
            print("downloading", URL, flush=True)
            urllib.request.urlretrieve(URL, tgz)
        print("extracting", flush=True)
        with tarfile.open(tgz) as t:
            t.extractall(RAW)


def _parse(path):
    try:
        with open(path, "rb") as f:
            msg = email.message_from_bytes(f.read())
        rel = os.path.relpath(path, RAW / "maildir").split(os.sep)
        user, folders = rel[0], rel[1:-1]
        payload = msg.get_payload(decode=False)
        if isinstance(payload, list):
            payload = "\n".join(p.get_payload(decode=False) for p in payload if isinstance(p.get_payload(decode=False), str))
        body = clean_body(payload or "")
        try:
            date = int(email.utils.parsedate_to_datetime(msg.get("Date")).timestamp())
        except Exception:
            date = 0
        from_name = msg.get("X-From") or msg.get("From") or ""
        from_name = re.sub(r"<.*?>", "", from_name).strip().strip('"')[:80]
        return {
            "msgid": (msg.get("Message-ID") or path).strip(),
            "user": user,
            "folder": folders[-1] if folders else "",
            "top": folders[0] if folders else "",
            "subject": (msg.get("Subject") or "").strip()[:300],
            "from_addr": (msg.get("From") or "").strip()[:120],
            "from_name": from_name,
            "date": date,
            "body": body,
        }
    except Exception:
        return None


def _files(limit=None):
    out = []
    for dp, _, fns in os.walk(RAW / "maildir"):
        for fn in fns:
            out.append(os.path.join(dp, fn))
    out.sort()
    if limit:
        random.Random(0).shuffle(out)
        out = out[:limit]
    return out


def build(limit=None, min_folder=20, heldout_frac=0.15, test_thread_frac=0.10, val_frac=0.05, seed=0):
    files = _files(limit)
    print(f"parsing {len(files)} files", flush=True)
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        rows = [r for r in pool.imap(_parse, files, chunksize=500) if r]
    print(f"parsed {len(rows)}", flush=True)

    # dedupe by normalized body hash, union the folder labels
    by_hash = {}
    for r in rows:
        if len(r["body"]) < 30:
            continue
        key = sha(re.sub(r"[^a-z0-9]", "", r["body"].lower())[:2000])
        lab = None
        if r["folder"] and r["folder"].lower() not in GENERIC and r["top"].lower() not in {"sent", "sent_items", "_sent_mail", "deleted_items"}:
            h = humanize(r["folder"])
            if good_label(h):
                lab = h
        if key in by_hash:
            if lab:
                by_hash[key]["_labs"].add(lab)
            continue
        r["id"] = key
        r["_labs"] = {lab} if lab else set()
        by_hash[key] = r
    emails = list(by_hash.values())
    print(f"deduped {len(emails)}", flush=True)

    cnt = Counter(l for e in emails for l in e["_labs"])
    labels = sorted(l for l, c in cnt.items() if c >= min_folder)
    keep = set(labels)
    for e in emails:
        e["topics"] = sorted(l for l in e.pop("_labs") if l in keep)

    # threads by normalized subject; huge generic subjects become singletons
    for e in emails:
        ns = norm_subject(e["subject"])
        e["thread_id"] = sha("t:" + ns) if len(ns) >= 4 else sha("s:" + e["id"])
        e["is_reply"] = is_reply(e["subject"])
    tsize = Counter(e["thread_id"] for e in emails)
    for e in emails:
        if tsize[e["thread_id"]] > 300:
            e["thread_id"] = sha("s:" + e["id"])
    by_thread = defaultdict(list)
    for e in emails:
        by_thread[e["thread_id"]].append(e)
    for th in by_thread.values():
        th.sort(key=lambda x: x["date"])
        for i, e in enumerate(th):
            e["parent_id"] = th[i - 1]["id"] if (e["is_reply"] and i > 0) else None

    rng = random.Random(seed)
    labs = labels[:]
    rng.shuffle(labs)
    held = set(labs[: int(len(labs) * heldout_frac)])
    # a thread touching a held-out label goes entirely to test_topic
    thread_split = {}
    for tid, th in by_thread.items():
        if any(l in held for e in th for l in e["topics"]):
            thread_split[tid] = "test_topic"
        else:
            u = int(sha("split:" + tid), 16) % 1000 / 1000
            thread_split[tid] = "test_thread" if u < test_thread_frac else ("val" if u < test_thread_frac + val_frac else "train")
    for e in emails:
        e["split"] = thread_split[e["thread_id"]]
        e.pop("folder", None)
        e.pop("top", None)
    topics = {l: {"n": cnt[l], "split": "heldout" if l in held else "train"} for l in labels}
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "emails.jsonl", emails)
    (OUT / "topics.json").write_text(json.dumps(topics, indent=1))
    sc = Counter(e["split"] for e in emails)
    print(f"wrote {len(emails)} emails, {len(labels)} topics ({len(held)} held out), splits {dict(sc)}", flush=True)
    print(f"labeled emails: {sum(1 for e in emails if e['topics'])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-folder", type=int, default=20)
    a = ap.parse_args()
    if a.download:
        download()
    build(limit=a.limit, min_folder=a.min_folder)


if __name__ == "__main__":
    main()
