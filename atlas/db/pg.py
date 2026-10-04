"""Postgres + pgvector storage (Tiger Data). Same data as the SQLite store plus chat history.

    from atlas.db.pg import PgStore
    db = PgStore()                       # DATABASE_URL from env
    db.knn(qvec, k=10, encoder="base")   # [(id, cosine)]

Local: docker-compose up -d (see docs/TIGER.md). Tiger Cloud: point DATABASE_URL at the service.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path

import numpy as np

from atlas import config

log = logging.getLogger("atlas.db")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
VEC_DIM = 768
EMAIL_COLS = ["id", "thread_id", "from_addr", "from_name", "to_addrs", "date", "subject", "body", "snippet",
              "labels", "source", "raw_path"]


def database_url():
    return config.env("DATABASE_URL", "")


def raw_path_for(row):
    if (row.get("source") or "") == "gmail":
        return f"data/raw/gmail/{row['id']}.eml"
    return row.get("raw_path")


def pad(v):
    """Zero-pad a (dim,) or (n, dim) float array to VEC_DIM. Cosine is unchanged by zero padding."""
    v = np.asarray(v, dtype=np.float32)
    d = v.shape[-1]
    if d > VEC_DIM:
        raise ValueError(f"embedding dim {d} > {VEC_DIM}")
    if d == VEC_DIM:
        return v
    out = np.zeros(v.shape[:-1] + (VEC_DIM,), np.float32)
    out[..., :d] = v
    return out


def _arr(v):
    """pgvector may hand back a Vector object or an ndarray depending on version."""
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32)


def _labels(v):
    if v is None or isinstance(v, (list, dict)):
        return json.dumps(v) if v is not None else None
    try:
        return json.dumps(json.loads(v))
    except Exception:
        return json.dumps([str(v)])


def _ts(v, end=False):
    from atlas.search.hybrid import _ts as hts

    return hts(v, end)


class PgStore:
    kind = "pg"

    def __init__(self, url=None, init=True):
        import psycopg
        from pgvector.psycopg import register_vector

        self.url = url or database_url()
        if not self.url:
            raise RuntimeError("DATABASE_URL not set")
        self._psycopg = psycopg
        self._register = register_vector
        self.lock = threading.RLock()
        self.conn = None
        self._mat = {}
        self.has_timescale = False
        self._connect()
        if init:
            self.init_db()

    # ---------- connection ----------

    def _connect(self):
        self.conn = self._psycopg.connect(self.url, autocommit=True, connect_timeout=10)
        self.conn.execute("create extension if not exists vector")
        self._register(self.conn)
        try:  # pgvector >= 0.8: keep scanning the HNSW graph when the encoder filter drops rows
            self.conn.execute("set hnsw.iterative_scan = relaxed_order")
            self.conn.execute("set hnsw.ef_search = 100")
        except Exception:
            pass

    def execute(self, sql, params=None):
        with self.lock:
            try:
                return self.conn.execute(sql, params)
            except (self._psycopg.OperationalError, self._psycopg.InterfaceError):
                log.warning("postgres connection lost, reconnecting")
                self._connect()
                return self.conn.execute(sql, params)

    def fetchall(self, sql, params=None):
        with self.lock:
            return self.execute(sql, params).fetchall()

    def fetchone(self, sql, params=None):
        with self.lock:
            return self.execute(sql, params).fetchone()

    def init_db(self):
        with self.lock:
            self.execute(SCHEMA_PATH.read_text())
            try:
                self.execute("create extension if not exists timescaledb")
                self.execute("select create_hypertable('query_log', 'ts', if_not_exists => true, migrate_data => true)")
                self.has_timescale = True
            except Exception as e:
                log.info("timescaledb not available, query_log stays a plain table: %s", e)

    def close(self):
        if self.conn:
            self.conn.close()

    # ---------- emails ----------

    def upsert_emails(self, rows):
        rows = list(rows)
        if not rows:
            return 0
        cols = ",".join(EMAIL_COLS)
        sets = ",".join(f"{c}=excluded.{c}" for c in EMAIL_COLS[1:])
        sql = f"insert into emails({cols}) values ({','.join(['%s'] * len(EMAIL_COLS))}) on conflict(id) do update set {sets}"
        data = []
        for r in rows:
            r = dict(r)
            r["labels"] = _labels(r.get("labels"))
            r["raw_path"] = raw_path_for(r)
            data.append([r.get(c) for c in EMAIL_COLS])
        with self.lock:
            with self.conn.cursor() as cur:
                cur.executemany(sql, data)
        return len(data)

    def get_email(self, id):
        r = self.fetchone(f"select {','.join(EMAIL_COLS)} from emails where id=%s", (id,))
        if not r:
            return None
        d = dict(zip(EMAIL_COLS, r))
        if d["labels"] is not None and not isinstance(d["labels"], str):
            d["labels"] = json.dumps(d["labels"])  # same shape as SQLite: JSON text
        return d

    def all_ids(self):
        return [r[0] for r in self.fetchall("select id from emails order by date")]

    def count(self):
        return self.fetchone("select count(*) from emails")[0]

    def email_meta(self):
        """id -> (date, from_addr, from_name, source), for date, sender and source filters."""
        return {r[0]: (r[1], r[2], r[3], r[4]) for r in self.fetchall("select id, date, from_addr, from_name, source from emails")}

    def fts(self, query, k=50):
        """OR over alphanumeric tokens like the SQLite fts_search, ranked by ts_rank_cd."""
        toks = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
        if not toks:
            return []
        q = " or ".join(toks)
        sql = ("select id, ts_rank_cd(tsv, q) s from emails, websearch_to_tsquery('english', %s) q "
               "where tsv @@ q order by s desc, date desc limit %s")
        return [(r[0], float(r[1])) for r in self.fetchall(sql, (q, k))]

    # ---------- vectors ----------

    def upsert_vectors(self, encoder, ids, E):
        ids = list(ids)
        E = np.asarray(E, dtype=np.float32).reshape(len(ids), -1)
        if not ids:
            return 0
        dim = int(E.shape[1])
        P = pad(E)
        sql = ("insert into email_vectors(email_id, encoder, dim, embedding) values (%s,%s,%s,%s) "
               "on conflict(email_id, encoder) do update set dim=excluded.dim, embedding=excluded.embedding")
        with self.lock:
            with self.conn.cursor() as cur:
                cur.executemany(sql, [(i, encoder, dim, P[j]) for j, i in enumerate(ids)])
        self._mat.pop(encoder, None)
        return len(ids)

    def knn(self, query_vec, k=10, encoder=None, filters=None):
        """Nearest emails by cosine via the HNSW index: [(id, cosine)]."""
        encoder = encoder or config.ENCODER
        q = pad(query_vec)
        where, params = ["v.encoder = %s"], [encoder]
        f = {kk: vv for kk, vv in (filters or {}).items() if vv not in (None, "", [])}
        join = ""
        if f:
            join = "join emails e on e.id = v.email_id"
            if f.get("after"):
                where.append("e.date >= %s")
                params.append(_ts(f["after"]))
            if f.get("before"):
                where.append("e.date < %s")
                params.append(_ts(f["before"], end=True))
            if f.get("from"):
                where.append("(coalesce(e.from_addr,'') || ' ' || coalesce(e.from_name,'')) ilike %s")
                params.append(f"%{f['from']}%")
        sql = (f"select v.email_id, 1 - (v.embedding <=> %s) cos from email_vectors v {join} "
               f"where {' and '.join(where)} order by v.embedding <=> %s limit %s")
        return [(r[0], float(r[1])) for r in self.fetchall(sql, [q] + params + [q, int(k)])]

    def vectors_matrix(self, encoder, reload=False):
        """(ids, E float32 (n, dim)) for one encoder, cached. The region scorer needs all of it."""
        if not reload and encoder in self._mat:
            return self._mat[encoder]
        rows = self.fetchall("select v.email_id, v.dim, v.embedding from email_vectors v join emails e on e.id = v.email_id "
                             "where v.encoder=%s order by e.date, v.email_id", (encoder,))
        if not rows:
            out = ([], np.zeros((0, 0), np.float32))
        else:
            dim = rows[0][1]
            out = ([r[0] for r in rows], np.stack([_arr(r[2])[:dim] for r in rows]))
        self._mat[encoder] = out
        return out

    def encoders(self):
        return [r[0] for r in self.fetchall("select distinct encoder from email_vectors order by 1")]

    # ---------- chunks ----------

    def upsert_chunks(self, rows, encoder=None, E=None):
        """rows: {id, doc_id, source, uri, heading, span_start, span_end, text}; optional vectors."""
        rows = list(rows)
        cols = ["id", "doc_id", "source", "uri", "heading", "span_start", "span_end", "text"]
        sql = (f"insert into chunks({','.join(cols)}) values ({','.join(['%s'] * len(cols))}) on conflict(id) do update set "
               + ",".join(f"{c}=excluded.{c}" for c in cols[1:]))
        with self.lock:
            with self.conn.cursor() as cur:
                cur.executemany(sql, [[r.get(c) for c in cols] for r in rows])
                if encoder is not None and E is not None:
                    E = np.asarray(E, np.float32).reshape(len(rows), -1)
                    P = pad(E)
                    cur.executemany("insert into chunk_vectors(chunk_id, encoder, dim, embedding) values (%s,%s,%s,%s) "
                                    "on conflict(chunk_id, encoder) do update set dim=excluded.dim, embedding=excluded.embedding",
                                    [(r["id"], encoder, int(E.shape[1]), P[j]) for j, r in enumerate(rows)])
        return len(rows)

    def knn_chunks(self, query_vec, k=10, encoder=None, source=None):
        encoder = encoder or config.ENCODER
        q = pad(query_vec)
        where, params = ["v.encoder=%s"], [encoder]
        if source:
            where.append("c.source=%s")
            params.append(source)
        sql = ("select c.id, c.doc_id, c.source, c.uri, c.heading, c.span_start, c.span_end, c.text, "
               "1 - (v.embedding <=> %s) cos from chunk_vectors v join chunks c on c.id = v.chunk_id "
               f"where {' and '.join(where)} order by v.embedding <=> %s limit %s")
        keys = ["id", "doc_id", "source", "uri", "heading", "span_start", "span_end", "text", "cos"]
        return [dict(zip(keys, r)) for r in self.fetchall(sql, [q] + params + [q, int(k)])]

    # ---------- events / watches mirrors ----------

    def upsert_events(self, rows):
        with self.lock:
            with self.conn.cursor() as cur:
                cur.executemany('insert into events(id, title, start, "end", location, source) values (%s,%s,%s,%s,%s,%s) '
                                'on conflict(id) do update set title=excluded.title, start=excluded.start, '
                                '"end"=excluded."end", location=excluded.location, source=excluded.source',
                                [[r["id"], r["title"], r["start"], r["end"], r.get("location"), r.get("source")] for r in rows])

    def upsert_watches(self, rows):
        with self.lock:
            with self.conn.cursor() as cur:
                cur.executemany("insert into watches(id, name, positive, negative, created, last_checked) values (%s,%s,%s,%s,%s,%s) "
                                "on conflict(id) do update set name=excluded.name, positive=excluded.positive, "
                                "negative=excluded.negative, last_checked=excluded.last_checked",
                                [[r["id"], r["name"], r.get("positive"), r.get("negative"), r.get("created"),
                                  r.get("last_checked")] for r in rows])

    # ---------- chat ----------

    def new_session(self, channel=None, user_handle=None, session_id=None, title=None):
        sid = session_id or uuid.uuid4().hex[:16]
        self.execute("insert into chat_sessions(id, channel, user_handle, title) values (%s,%s,%s,%s) "
                     "on conflict(id) do nothing", (sid, channel, user_handle, title))
        return sid

    def append_message(self, session_id, role, content, tool_name=None, tool_args=None, hits=None,
                       channel=None, user_handle=None):
        from psycopg.types.json import Jsonb

        self.new_session(channel, user_handle, session_id, title=(content or "")[:80] if role == "user" else None)
        with self.lock:
            self.execute("update chat_sessions set updated=now(), title=coalesce(title, %s) where id=%s",
                         ((content or "")[:80] if role == "user" else None, session_id))
            r = self.execute("insert into chat_messages(session_id, role, content, tool_name, tool_args, hits) "
                             "values (%s,%s,%s,%s,%s,%s) returning id",
                             (session_id, role, content, tool_name,
                              Jsonb(tool_args) if tool_args is not None else None,
                              Jsonb(hits) if hits is not None else None)).fetchone()
        return r[0]

    def history(self, session_id, n=20, roles=("user", "assistant")):
        """Last n messages of a session, oldest first: [{role, content, ts, tool_name, tool_args, hits}]."""
        rows = self.fetchall("select role, content, created, tool_name, tool_args, hits from chat_messages "
                             "where session_id=%s and role = any(%s) order by id desc limit %s",
                             (session_id, list(roles), int(n)))
        return [{"role": r[0], "content": r[1], "ts": r[2].timestamp(), "tool_name": r[3], "tool_args": r[4],
                 "hits": r[5]} for r in reversed(rows)]

    def list_sessions(self, limit=50):
        rows = self.fetchall("select id, title, channel, user_handle, updated from chat_sessions order by updated desc limit %s",
                             (int(limit),))
        return [{"id": r[0], "title": r[1], "channel": r[2], "user_handle": r[3], "updated": r[4].timestamp()} for r in rows]

    # ---------- metrics ----------

    def log_query(self, channel, query, n_facets=None, region_size=None, tokens_returned=None, latency_ms=None):
        self.execute("insert into query_log(channel, query, n_facets, region_size, tokens_returned, latency_ms) "
                     "values (%s,%s,%s,%s,%s,%s)", (channel, query, n_facets, region_size, tokens_returned, latency_ms))


