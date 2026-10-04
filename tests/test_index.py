import json

import numpy as np
import pytest

from atlas import config, store
from atlas.index import build_index, load_index, load_probes
from atlas.model.encoder import load_encoder


@pytest.fixture
def fixture_conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    conn = store.connect(":memory:")
    rows = store.load_fixture(conn)
    return conn, rows


def test_probes():
    p = load_probes()
    assert len(p) == 300 and len(set(p)) == 300


def test_build_and_load_index(fixture_conn, tmp_path):
    conn, rows = fixture_conn
    enc = load_encoder("hash")
    idx = build_index(enc, conn)
    d = tmp_path / "index" / "hash"
    for f in ("emb.npy", "ids.json", "meta.json", "hub.npz", "map.json"):
        assert (d / f).exists(), f
    assert np.load(d / "emb.npy").dtype == np.float16
    meta = json.loads((d / "meta.json").read_text())
    assert meta["encoder"] == "hash" and meta["dim"] == 256 and "built_at" in meta
    m = json.loads((d / "map.json").read_text())
    assert len(m["points"]) == len(rows) and m["clusters"]
    assert {"id", "x", "y", "cluster"} <= set(m["points"][0]) and {"id", "label", "size"} <= set(m["clusters"][0])
    assert sum(c["size"] for c in m["clusters"]) == sum(p["cluster"] >= 0 for p in m["points"])

    li = load_index("hash", reload=True)
    assert li.ids == idx.ids and li.E.dtype == np.float32 and li.E.shape == (len(rows), 256)
    assert li.mu.shape == (len(rows),) and (li.sigma > 0).all()
    assert np.allclose(np.linalg.norm(li.E, axis=1), 1, atol=1e-2)

    # incremental add
    v = enc.encode_docs(["Codeforces round tomorrow, 6 problems"])
    li.add(["new1"], v)
    again = load_index("hash", reload=True)
    assert again.ids[-1] == "new1" and again.E.shape[0] == len(rows) + 1 and len(again.mu) == len(rows) + 1
    assert any(p["id"] == "new1" for p in again.map["points"])
    li.add(["new1"], v)  # re-adding replaces, no duplicate
    assert load_index("hash", reload=True).ids.count("new1") == 1


def test_load_missing_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "nothing")
    with pytest.raises(FileNotFoundError):
        load_index("hash", reload=True)


@pytest.mark.slow
def test_umap_path_runs():
    """The big-corpus branch (UMAP + HDBSCAN) on synthetic blobs."""
    from atlas.index import mapping

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 32))
    E = np.vstack([c + 0.05 * rng.normal(size=(60, 32)) for c in centers]).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    xy, lab = mapping.layout(E)
    assert xy.shape == (240, 2) and len(set(lab) - {-1}) >= 3


@pytest.mark.slow
def test_base_encoder_coding_competition(fixture_conn):
    conn, rows = fixture_conn
    enc = load_encoder("base")
    idx = build_index(enc, conn)
    topic = {r["id"]: r["topic"] for r in rows}
    q = enc.encode_queries(["coding competition"])[0]
    s = idx.E @ q
    z = (s - idx.mu) / idx.sigma
    order = [idx.ids[i] for i in np.argsort(-z)]
    top4 = [topic[i] for i in order[:4]]
    assert sum(t in ("hackathon", "contest") for t in top4) >= 3, top4
