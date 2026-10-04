"""SQLite storage for emails, calendar events and watches. Schema lives in CONTRACT.md."""

import json
import sqlite3

from atlas import config

SCHEMA = """
create table if not exists emails(
  id text primary key, thread_id text, from_addr text, from_name text, to_addrs text,
  date integer, subject text, body text, snippet text, labels text, source text);
create virtual table if not exists emails_fts using fts5(
  subject, body, from_name, content='emails', content_rowid='rowid');
create trigger if not exists emails_ai after insert on emails begin
  insert into emails_fts(rowid, subject, body, from_name) values (new.rowid, new.subject, new.body, new.from_name);
end;
create trigger if not exists emails_ad after delete on emails begin
  insert into emails_fts(emails_fts, rowid, subject, body, from_name) values ('delete', old.rowid, old.subject, old.body, old.from_name);
end;
create trigger if not exists emails_au after update on emails begin
  insert into emails_fts(emails_fts, rowid, subject, body, from_name) values ('delete', old.rowid, old.subject, old.body, old.from_name);
  insert into emails_fts(rowid, subject, body, from_name) values (new.rowid, new.subject, new.body, new.from_name);
end;
create table if not exists events(id text primary key, title text, start integer, end integer, location text, source text);
create table if not exists watches(id text primary key, name text, positive text, negative text, created integer, last_checked integer);
"""

COLS = ["id", "thread_id", "from_addr", "from_name", "to_addrs", "date", "subject", "body", "snippet", "labels", "source"]


def connect(path=None):
    path = path or config.DB_PATH
    if str(path) != ":memory:":
        config.DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)


def upsert_emails(conn, rows):
    q = f"insert or replace into emails({','.join(COLS)}) values ({','.join('?' * len(COLS))})"
    conn.executemany(q, [[r.get(c) for c in COLS] for r in rows])
    conn.commit()


def get_email(conn, id):
    r = conn.execute("select * from emails where id=?", (id,)).fetchone()
    return dict(r) if r else None


def all_ids(conn):
    return [r[0] for r in conn.execute("select id from emails order by date")]


def embed_text(r):
    return f"{r['subject'] or ''}\n{r['from_name'] or ''}\n{(r['body'] or '')[:2000]}"


def texts_for_embedding(conn):
    return [(r["id"], embed_text(r)) for r in conn.execute("select * from emails order by date")]


def fts_search(conn, query, k=50):
    toks = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
    if not toks:
        return []
    match = " OR ".join(f'"{t}"' for t in toks)
    sql = ("select e.id, bm25(emails_fts) s from emails_fts join emails e on e.rowid=emails_fts.rowid "
           "where emails_fts match ? order by s limit ?")
    return [(r[0], -r[1]) for r in conn.execute(sql, (match, k))]


def upsert_events(conn, rows):
    conn.executemany("insert or replace into events values (?,?,?,?,?,?)",
                     [[r["id"], r["title"], r["start"], r["end"], r.get("location"), r.get("source")] for r in rows])
    conn.commit()


def events_between(conn, t0, t1):
    return [dict(r) for r in conn.execute("select * from events where start < ? and end > ? order by start", (t1, t0))]


def load_fixture(conn, path=None):
    path = path or (config.ROOT / "tests" / "fixtures" / "mailbox.jsonl")
    rows = [json.loads(l) for l in open(path)]
    upsert_emails(conn, rows)
    return rows
