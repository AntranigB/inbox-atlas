"""Incremental Gmail poll (read-only): new UIDs since the last run -> raw .eml -> clean -> upsert ->
embed -> append to the index.

    from atlas.ingest.sync import sync_new
    new_ids = sync_new()        # list[str], [] if no credentials or nothing new

    uv run python -m atlas.ingest.sync [--loop 120]

Thread safe (one sync at a time; concurrent callers get []).
"""

import argparse
import sys
import threading
import time

from atlas import store
from atlas.ingest import gmail_export as gx
from atlas.ingest import imap
from atlas.ingest.clean import parse_message

_lock = threading.Lock()
MAX_PER_SYNC = 500


def _embed_and_index(conn, ids, encoder=None):
    from atlas.index import load_index
    from atlas.model.encoder import load_encoder

    enc = encoder or load_encoder()
    try:
        idx = load_index(enc.name)
    except FileNotFoundError:
        return False
    texts = [store.embed_text(store.get_email(conn, i)) for i in ids]
    idx.add(ids, enc.encode_docs(texts))
    return True


def sync_new(conn=None, encoder=None, verbose=False):
    user, pw = imap.credentials()
    if not user or not pw:
        return []
    if not _lock.acquire(blocking=False):
        return []
    try:
        return _sync(conn, encoder, verbose)
    finally:
        _lock.release()


def _sync(conn, encoder, verbose):
    gx.RAW_DIR.mkdir(parents=True, exist_ok=True)
    own_conn = conn is None
    conn = conn or store.connect()
    c = imap.connect()
    try:
        box, uidvalidity = imap.open_all_mail(c)
        st = gx.load_state()
        if not st.get("last_uid") or st.get("uidvalidity") != uidvalidity:
            # never exported (or mailbox reset): start watching from now, do not backfill
            uids = imap.search_uids(c)
            gx.save_state({"uidvalidity": uidvalidity, "last_uid": max(uids or [0]), "mailbox": box})
            if verbose:
                print("no previous export state, watching for mail from now on")
            return []
        last = st["last_uid"]
        uids = imap.search_uids(c, since_uid=last)[:MAX_PER_SYNC]
        if not uids:
            return []
        have = set(store.all_ids(conn))
        rows, metas = [], []
        for i in range(0, len(uids), gx.BODY_BATCH):
            for m, body in imap.fetch_raw(c, uids[i:i + gx.BODY_BATCH]):
                gx.save_raw(m["msgid"], body)  # raw first, never deleted
                m["uidvalidity"] = uidvalidity
                metas.append(m)
                try:
                    rows.append(parse_message(body, m["msgid"], m["thrid"], m["labels"], "gmail"))
                except Exception as e:
                    if verbose:
                        print(f"parse error {m['msgid']}: {e}")
        gx.append_meta(metas)
        store.upsert_emails(conn, rows)
        new_ids = [r["id"] for r in rows if r["id"] not in have]
        gx.save_state({"uidvalidity": uidvalidity, "last_uid": max(uids), "mailbox": box})
    finally:
        try:
            c.logout()
        except Exception:
            pass
    if new_ids:
        try:
            _embed_and_index(conn, new_ids, encoder)
        except Exception as e:
            if verbose:
                print(f"index update failed: {e}")
    if verbose:
        print(f"{len(new_ids)} new emails")
    if own_conn:
        conn.close()
    return new_ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="poll every N seconds")
    a = ap.parse_args(argv)
    if not all(imap.credentials()):
        print(imap.SETUP_MSG)
        return 1
    while True:
        ids = sync_new(verbose=True)
        for i in ids:
            e = store.get_email(store.connect(), i)
            print(f"  + {e['from_name']}: {e['subject']}")
        if not a.loop:
            return 0
        time.sleep(a.loop)


if __name__ == "__main__":
    sys.exit(main())
