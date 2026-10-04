"""Export ALL Gmail to data/raw/gmail/<X-GM-MSGID>.eml (read-only, resumable), then parse into
data/mail.sqlite.

    uv run python -m atlas.ingest.gmail_export            # GMAIL_MAX_MESSAGES newest (0 = all)
    uv run python -m atlas.ingest.gmail_export --max 0    # everything
    uv run python -m atlas.ingest.gmail_export --parse-only

Nothing in the mailbox changes: readonly select, BODY.PEEK[], no STORE/COPY/MOVE/EXPUNGE.
Raw files are never deleted.
"""

import argparse
import json
import os
import sys
import time

from atlas import config, store
from atlas.ingest import imap
from atlas.ingest.clean import parse_message

RAW_DIR = config.DATA / "raw" / "gmail"
META_PATH = RAW_DIR / "_meta.jsonl"
STATE_PATH = RAW_DIR / "_state.json"
META_BATCH = 1000
BODY_BATCH = 200


def raw_path(msgid):
    return RAW_DIR / f"{msgid}.eml"


def load_meta():
    meta = {}
    if META_PATH.exists():
        for line in open(META_PATH):
            try:
                m = json.loads(line)
                meta[m["msgid"]] = m
            except Exception:
                pass
    return meta


def append_meta(rows):
    with open(META_PATH, "a") as f:
        for m in rows:
            f.write(json.dumps(m) + "\n")


def load_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def save_state(st):
    STATE_PATH.write_text(json.dumps(st))


def save_raw(msgid, body):
    p = raw_path(msgid)
    if p.exists():
        return False
    tmp = p.with_suffix(".eml.part")
    tmp.write_bytes(body)
    os.replace(tmp, p)
    return True


def _bar(done, total, t0, what):
    rate = done / max(time.time() - t0, 1e-6)
    eta = (total - done) / rate if rate else 0
    sys.stdout.write(f"\r  {what}: {done}/{total}  {rate:.0f}/s  eta {eta/60:.1f} min   ")
    sys.stdout.flush()


def download(c, uids, meta_by_uid, uidvalidity):
    """Fetch raw bodies for uids whose .eml is missing. Returns list of meta for saved/existing."""
    need = [u for u in uids if not (u in meta_by_uid and raw_path(meta_by_uid[u]["msgid"]).exists())]
    print(f"  {len(uids) - len(need)} already on disk, {len(need)} to download")
    t0, done = time.time(), 0
    for i in range(0, len(need), BODY_BATCH):
        chunk = need[i:i + BODY_BATCH]
        for attempt in range(3):
            try:
                got = imap.fetch_raw(c, chunk)
                break
            except imap.ReadOnlyViolation:
                raise
            except Exception as e:  # transient network errors: retry the batch
                if attempt == 2:
                    raise
                print(f"\n  batch failed ({e}), retrying")
                time.sleep(2)
        new_meta = []
        for m, body in got:
            save_raw(m["msgid"], body)
            m["uidvalidity"] = uidvalidity
            if m["uid"] not in meta_by_uid:
                new_meta.append(m)
            meta_by_uid[m["uid"]] = m
        append_meta(new_meta)
        done += len(chunk)
        _bar(done, len(need), t0, "download")
    if need:
        print()


def export(max_n=None):
    max_n = int(config.env("GMAIL_MAX_MESSAGES", "0") or 0) if max_n is None else max_n
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    c = imap.connect()
    try:
        box, uidvalidity = imap.open_all_mail(c)
        print(f"opened {box} read-only (uidvalidity {uidvalidity})")
        uids = imap.search_uids(c)
        print(f"{len(uids)} messages in mailbox")
        uids = sorted(uids, reverse=True)  # newest first
        if max_n:
            uids = uids[:max_n]
        meta = load_meta()
        meta_by_uid = {m["uid"]: m for m in meta.values() if m.get("uidvalidity") == uidvalidity}
        # metadata pass for uids we have never seen (cheap, tells us which msgids are on disk)
        unseen = [u for u in uids if u not in meta_by_uid]
        t0 = time.time()
        for i in range(0, len(unseen), META_BATCH):
            got = imap.fetch_meta(c, unseen[i:i + META_BATCH])
            for m in got:
                m["uidvalidity"] = uidvalidity
                meta_by_uid[m["uid"]] = m
            append_meta(got)
            _bar(min(i + META_BATCH, len(unseen)), len(unseen), t0, "metadata")
        if unseen:
            print()
        download(c, uids, meta_by_uid, uidvalidity)
        st = load_state()
        last = st.get("last_uid", 0) if st.get("uidvalidity") == uidvalidity else 0
        save_state({"uidvalidity": uidvalidity, "last_uid": max([last] + uids), "mailbox": box})
    finally:
        try:
            c.logout()
        except Exception:
            pass


def parse_all(conn=None, reparse=False):
    """Parse every raw .eml into the emails table. Skips ids already present unless reparse."""
    conn = conn or store.connect()
    meta = load_meta()
    have = set() if reparse else set(store.all_ids(conn))
    files = sorted(RAW_DIR.glob("*.eml"))
    todo = [p for p in files if p.stem not in have]
    print(f"parsing {len(todo)} of {len(files)} raw messages")
    rows, t0, n_err = [], time.time(), 0
    for i, p in enumerate(todo, 1):
        m = meta.get(p.stem, {})
        try:
            rows.append(parse_message(p.read_bytes(), p.stem, m.get("thrid"), m.get("labels"), "gmail"))
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"\n  parse error {p.name}: {e}")
        if len(rows) >= 500:
            store.upsert_emails(conn, rows)
            rows = []
        if i % 200 == 0 or i == len(todo):
            _bar(i, len(todo), t0, "parse")
    if rows:
        store.upsert_emails(conn, rows)
    if todo:
        print()
    print(f"done: {len(store.all_ids(conn))} emails in {config.DB_PATH} ({n_err} parse errors)")
    return conn


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only Gmail export into data/")
    ap.add_argument("--max", type=int, default=None, help="newest N messages (0 = all; default GMAIL_MAX_MESSAGES)")
    ap.add_argument("--parse-only", action="store_true", help="skip IMAP, just parse raw files on disk")
    ap.add_argument("--reparse", action="store_true", help="re-parse emails already in the db")
    a = ap.parse_args(argv)
    if not a.parse_only:
        user, pw = imap.credentials()
        if not user or not pw:
            print(imap.SETUP_MSG)
            return 1
        export(a.max)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    parse_all(reparse=a.reparse)
    print("next: uv run python -m atlas.index.build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
