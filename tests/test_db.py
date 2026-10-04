"""Postgres backend. Offline parts always run; the pg-marked ones need DATABASE_URL (a scratch
schema is created and dropped, so a real database is left alone)."""

import os

import numpy as np
import pytest

from atlas import config, store
from atlas.db import backend as B
from atlas.db.pg import VEC_DIM, pad, raw_path_for
from atlas.model.encoder import load_encoder
from atlas.search import engine as engmod
from atlas.search import hybrid

TEST_SCHEMA = "atlas_test"


def test_mode_switch(monkeypatch):
    monkeypatch.setenv("ATLAS_DB", "sqlite")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x@localhost/none")
    assert B.mode() == "sqlite"
    monkeypatch.setenv("ATLAS_DB", "")
    assert B.mode() == "pg"
    monkeypatch.delenv("DATABASE_URL")
    assert B.mode() == "sqlite"


def test_pad_keeps_cosine():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 256)).astype(np.float32)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    pa, pb = pad(a), pad(b)
    assert pa.shape == (VEC_DIM,)
    assert abs(float(pa @ pb) - float(a @ b)) < 1e-6


def test_raw_path():
    assert raw_path_for({"id": "123", "source": "gmail"}) == "data/raw/gmail/123.eml"
    assert raw_path_for({"id": "x", "source": "fixture"}) is None


def test_sqlite_chat_history():
    conn = store.connect(":memory:")
    db = B.SqliteBackend(conn)
    sid = db.new_session("web", "me")
    db.append_message(sid, "user", "any hackathons?")
    db.append_message(sid, "tool", None, tool_name="search_region", tool_args={"positive": ["hackathon"]}, hits=["a"])
    db.append_message(sid, "assistant", "two of them", hits=["a", "b"])
    h = db.history(sid)
    assert [m["role"] for m in h] == ["user", "assistant"]
    assert h[1]["hits"] == ["a", "b"]
    assert db.list_sessions()[0]["title"] == "any hackathons?"


def test_ask_persists_turns(monkeypatch, tmp_path):
    from atlas.agent import grok

    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(engmod, "_try_load_index", lambda name: None)
    monkeypatch.setattr(config, "XAI_API_KEY", "")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    grok._exp_cache.clear()
    conn = store.connect(":memory:")
    store.load_fixture(conn)
    monkeypatch.setattr(engmod, "_engines", {})
    e = engmod.Engine("hash", conn)
    engmod.set_engine(e)
    r = grok.ask("hackathon", "web")
    sid = r["session_id"]
    r2 = grok.ask("and contests?", "web", session_id=sid)
    assert r2["session_id"] == sid
    h = e.backend.history(sid, 10)
    assert [m["role"] for m in h] == ["user", "assistant", "user", "assistant"]
    assert h[1]["hits"]


# ---------- live Postgres ----------

@pytest.fixture
def pg(monkeypatch):
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    import psycopg

    from atlas.db.pg import PgStore

    with psycopg.connect(url, autocommit=True) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"drop schema if exists {TEST_SCHEMA} cascade")
        c.execute(f"create schema {TEST_SCHEMA}")
    sep = "&" if "?" in url else "?"
    db = PgStore(f"{url}{sep}options=-csearch_path%3D{TEST_SCHEMA},public")
    yield db
    db.close()
    with psycopg.connect(url, autocommit=True) as c:
        c.execute(f"drop schema if exists {TEST_SCHEMA} cascade")


@pytest.mark.pg
def test_pg_matches_sqlite(pg, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(engmod, "_try_load_index", lambda name: None)
    monkeypatch.setattr(config, "XAI_API_KEY", "")
    conn = store.connect(":memory:")
    store.load_fixture(conn)
    enc = load_encoder("hash")
    pairs = store.texts_for_embedding(conn)
    ids = [i for i, _ in pairs]
    E = enc.encode_docs([t for _, t in pairs])
    pg.upsert_emails([store.get_email(conn, i) for i in ids])
    pg.upsert_vectors(enc.name, ids, E)

    got_ids, got_E = pg.vectors_matrix(enc.name)
    assert sorted(got_ids) == sorted(ids) and got_E.shape == E.shape

    q = enc.encode_queries(["hackathon"])[0]
    nn = pg.knn(q, 5, encoder=enc.name)
    cos = E @ q
    # hash vectors tie a lot, so compare scores rather than ids
    assert np.allclose([c for _, c in nn], np.sort(cos)[::-1][:5], atol=1e-4)
    assert all(abs(c - float(cos[ids.index(i)])) < 1e-4 for i, c in nn)

    assert pg.fts("hackathon", 5)
    assert pg.get_email(ids[0])["subject"] == store.get_email(conn, ids[0])["subject"]

    sqlite_eng = engmod.Engine("hash", conn)
    pg_eng = engmod.Engine("hash", conn)
    pg_eng.backend = pg
    pg_eng.reload()
    a = hybrid.search("coding competition", ["hackathon", "ICPC regional"], [], k=8, engine=sqlite_eng)
    b = hybrid.search("coding competition", ["hackathon", "ICPC regional"], [], k=8, engine=pg_eng)
    assert [h["id"] for h in a["hits"]] == [h["id"] for h in b["hits"]]
    assert a["region"]["size"] == b["region"]["size"]

    sid = pg.new_session("web", "me")
    pg.append_message(sid, "user", "hi")
    pg.append_message(sid, "tool", None, tool_name="search_region", tool_args={"positive": ["x"]}, hits=["a"])
    pg.append_message(sid, "assistant", "hello", hits=["a"])
    assert [m["role"] for m in pg.history(sid)] == ["user", "assistant"]
    assert pg.list_sessions()[0]["id"] == sid
