"""Copy data/mail.sqlite and data/index/<encoder>/emb.npy into Postgres (DATABASE_URL).

    uv run python -m atlas.db.migrate                 # every index under data/index
    uv run python -m atlas.db.migrate --encoder base  # just one
"""

import argparse
import json
import sys
import time

import numpy as np

from atlas import config, store


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", help="index dir name under data/index (repeatable)")
    ap.add_argument("--sqlite", default=None, help="path to mail.sqlite (default data/mail.sqlite)")
    ap.add_argument("--batch", type=int, default=1000)
    a = ap.parse_args(argv)

    from atlas.db.pg import PgStore, database_url

    if not database_url():
        print("DATABASE_URL is not set. Local: docker-compose up -d, then "
              "DATABASE_URL=postgresql://postgres:atlas@localhost:5433/atlas")
        return 1
    t0 = time.time()
    db = PgStore()
    conn = store.connect(a.sqlite)

    rows = [dict(r) for r in conn.execute("select * from emails order by date")]
    for i in range(0, len(rows), a.batch):
        db.upsert_emails(rows[i:i + a.batch])
    print(f"emails: {len(rows)}")

    ev = [dict(r) for r in conn.execute("select * from events")]
    if ev:
        db.upsert_events(ev)
    wa = [dict(r) for r in conn.execute("select * from watches")]
    if wa:
        db.upsert_watches(wa)
    print(f"events: {len(ev)}  watches: {len(wa)}")

    names = a.encoder or (sorted(p.name for p in config.INDEX_DIR.iterdir() if (p / "emb.npy").exists())
                          if config.INDEX_DIR.exists() else [])
    have = {r["id"] for r in rows}
    for name in names:
        d = config.INDEX_DIR / name
        E = np.load(d / "emb.npy").astype(np.float32)
        ids = json.loads((d / "ids.json").read_text())
        keep = [j for j, i in enumerate(ids) if i in have]
        for i in range(0, len(keep), a.batch):
            part = keep[i:i + a.batch]
            db.upsert_vectors(name, [ids[j] for j in part], E[part])
        skipped = len(ids) - len(keep)
        print(f"vectors[{name}]: {len(keep)} x {E.shape[1]}" + (f"  ({skipped} ids not in emails, skipped)" if skipped else ""))

    db.execute("analyze emails; analyze email_vectors")
    n = db.fetchone("select count(*) from email_vectors")[0]
    print(f"done in {time.time() - t0:.1f}s. email_vectors rows: {n}. timescaledb hypertable for query_log: {db.has_timescale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
