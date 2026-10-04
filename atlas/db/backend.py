"""Storage backend switch: Postgres (Tiger Data) or the SQLite store + numpy index.

    from atlas.db.backend import get_backend
    db = get_backend()      # ATLAS_DB=pg|sqlite, default pg when DATABASE_URL is set
    db.kind                 # "pg" | "sqlite"

Both expose: upsert_emails, get_email, all_ids, email_meta, fts, upsert_vectors, knn,
vectors_matrix, new_session, append_message, history, list_sessions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid

import numpy as np

from atlas import config, store

log = logging.getLogger("atlas.db")

CHAT_SCHEMA = """
create table if not exists chat_sessions(id text primary key, channel text, user_handle text, title text,
  created real, updated real);
create table if not exists chat_messages(id integer primary key autoincrement, session_id text, role text,
  content text, tool_name text, tool_args text, hits text, created real);
create index if not exists chat_messages_session on chat_messages(session_id, id);
"""


def mode():
    m = (config.env("ATLAS_DB", "") or "").strip().lower()
    if m in ("pg", "postgres", "tiger"):
        return "pg"
    if m == "sqlite":
        return "sqlite"
    return "pg" if config.env("DATABASE_URL") else "sqlite"


def pg_enabled():
    return mode() == "pg"


class SqliteBackend:
    """The original store: emails in data/mail.sqlite, vectors in data/index/<encoder>/emb.npy."""

    kind = "sqlite"

    def __init__(self, conn=None):
        self.conn = conn if conn is not None else store.connect()
        self.lock = threading.RLock()
        self._chat_ready = False

    def upsert_emails(self, rows):
        store.upsert_emails(self.conn, rows)
        return len(rows)

    def get_email(self, id):
        return store.get_email(self.conn, id)

    def all_ids(self):
        return store.all_ids(self.conn)

    def count(self):
        return self.conn.execute("select count(*) from emails").fetchone()[0]

    def email_meta(self):
        return {r[0]: (r[1], r[2], r[3]) for r in self.conn.execute("select id, date, from_addr, from_name from emails")}

    def fts(self, query, k=50):
        return store.fts_search(self.conn, query, k)

    def upsert_vectors(self, encoder, ids, E):
        """Vectors live in the file index (atlas.index); this is a no-op kept for interface parity."""
        return 0

    def vectors_matrix(self, encoder, reload=False):
        from atlas.index import load_index

        idx = load_index(encoder, reload=reload)
        return list(idx.ids), idx.E

    def knn(self, query_vec, k=10, encoder=None, filters=None):
        ids, E = self.vectors_matrix(encoder or config.ENCODER)
        if not len(ids):
            return []
        cos = E @ np.asarray(query_vec, np.float32)
        f = {kk: vv for kk, vv in (filters or {}).items() if vv not in (None, "", [])}
        if f:
            from atlas.search.hybrid import _ts

            meta, after, before, frm = self.email_meta(), _ts(f.get("after")), _ts(f.get("before"), end=True), f.get("from")
            for i, eid in enumerate(ids):
                date, addr, name = meta.get(eid, (None, "", ""))
                if (after and (date or 0) < after) or (before and (date or 0) >= before) or \
                        (frm and frm.lower() not in f"{addr or ''} {name or ''}".lower()):
                    cos[i] = -np.inf
        order = np.argsort(-cos)[:k]
        return [(ids[i], float(cos[i])) for i in order if np.isfinite(cos[i])]

    # chat history, stored next to the emails
    def _chat(self):
        if not self._chat_ready:
            with self.lock:
                self.conn.executescript(CHAT_SCHEMA)
                self._chat_ready = True

    def new_session(self, channel=None, user_handle=None, session_id=None, title=None):
        self._chat()
        sid = session_id or uuid.uuid4().hex[:16]
        now = time.time()
        with self.lock:
            self.conn.execute("insert or ignore into chat_sessions values (?,?,?,?,?,?)",
                              (sid, channel, user_handle, title, now, now))
            self.conn.commit()
        return sid

    def append_message(self, session_id, role, content, tool_name=None, tool_args=None, hits=None,
                       channel=None, user_handle=None):
        title = (content or "")[:80] if role == "user" else None
        self.new_session(channel, user_handle, session_id, title=title)
        with self.lock:
            self.conn.execute("update chat_sessions set updated=?, title=coalesce(title, ?) where id=?",
                              (time.time(), title, session_id))
            cur = self.conn.execute(
                "insert into chat_messages(session_id, role, content, tool_name, tool_args, hits, created) values (?,?,?,?,?,?,?)",
                (session_id, role, content, tool_name, None if tool_args is None else json.dumps(tool_args),
                 None if hits is None else json.dumps(hits), time.time()))
            self.conn.commit()
        return cur.lastrowid

    def history(self, session_id, n=20, roles=("user", "assistant")):
        self._chat()
        q = ",".join("?" * len(roles))
        rows = self.conn.execute(f"select role, content, created, tool_name, tool_args, hits from chat_messages "
                                 f"where session_id=? and role in ({q}) order by id desc limit ?",
                                 (session_id, *roles, int(n))).fetchall()
        return [{"role": r[0], "content": r[1], "ts": r[2], "tool_name": r[3],
                 "tool_args": json.loads(r[4]) if r[4] else None, "hits": json.loads(r[5]) if r[5] else None}
                for r in reversed(rows)]

    def list_sessions(self, limit=50):
        self._chat()
        rows = self.conn.execute("select id, title, channel, user_handle, updated from chat_sessions "
                                 "order by updated desc limit ?", (int(limit),)).fetchall()
        return [{"id": r[0], "title": r[1], "channel": r[2], "user_handle": r[3], "updated": r[4]} for r in rows]

    def log_query(self, *a, **kw):
        pass


_pg = None
_pg_lock = threading.Lock()


def get_pg():
    """Shared PgStore (one connection per process). Raises when Postgres is unreachable."""
    global _pg
    with _pg_lock:
        if _pg is None:
            from atlas.db.pg import PgStore

            _pg = PgStore()
        return _pg


def get_backend(conn=None, force_sqlite=False):
    """Postgres when ATLAS_DB=pg (or DATABASE_URL is set), else SQLite. Falls back to SQLite,
    with a warning, when Postgres cannot be reached so the app still starts."""
    if not force_sqlite and pg_enabled():
        try:
            return get_pg()
        except Exception as e:
            log.warning("ATLAS_DB=pg but Postgres is unavailable (%s); using SQLite", e)
    return SqliteBackend(conn)


def mirror_to_pg(encoder_name, ids, E, conn=None):
    """Copy the given emails (from SQLite) and their vectors into Postgres. Used by index build/add."""
    db = get_pg()
    conn = conn if conn is not None else store.connect()
    rows = [r for r in (store.get_email(conn, i) for i in ids) if r]
    db.upsert_emails(rows)
    have = {r["id"] for r in rows}
    keep = [j for j, i in enumerate(ids) if i in have]
    E = np.asarray(E, np.float32)
    db.upsert_vectors(encoder_name, [ids[j] for j in keep], E[keep] if len(keep) else E[:0])
    return len(keep)
