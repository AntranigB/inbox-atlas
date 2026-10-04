"""Loads the store, encoder and index once. Uses atlas.index when it exists, else builds in memory."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atlas import config, store
from atlas.model.encoder import load_encoder

log = logging.getLogger("atlas.search")

PROBES_FILE = Path(__file__).with_name("probes.json")


@dataclass
class MemIndex:
    ids: list
    E: np.ndarray
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    map: dict | None = None
    built: bool = False  # False when map.json came from the fallback, not from atlas.index
    pos: dict = field(default_factory=dict)


def probe_phrases() -> list[str]:
    return json.loads(PROBES_FILE.read_text())


def hub_stats(E: np.ndarray, enc) -> tuple[np.ndarray, np.ndarray]:
    """Per-email mean and std of cosine to 300 unrelated probe topics (hubness)."""
    Q = enc.encode_queries(probe_phrases())
    R = E @ Q.T
    return R.mean(1).astype(np.float32), R.std(1).astype(np.float32) + 1e-3


def fallback_map(E: np.ndarray, ids: list, conn, n_clusters: int | None = None) -> dict:
    """PCA 2D + tiny k-means, labeled by most common sender. Only when map.json is missing."""
    n = len(ids)
    if n == 0:
        return {"points": [], "clusters": [], "fallback": True}
    X = E - E.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    xy = U[:, :2] * S[:2] if n > 2 else np.zeros((n, 2))
    k = n_clusters or max(2, min(12, int(np.sqrt(n / 2))))
    k = min(k, n)
    rng = np.random.default_rng(0)
    C = E[rng.choice(n, k, replace=False)]
    for _ in range(15):
        lab = (E @ C.T).argmax(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c = E[m].mean(0)
                C[j] = c / max(np.linalg.norm(c), 1e-8)
    lab = (E @ C.T).argmax(1)
    names = {}
    for i, eid in enumerate(ids):
        r = conn.execute("select from_name from emails where id=?", (eid,)).fetchone()
        names.setdefault(int(lab[i]), []).append((r[0] if r else "") or "?")
    clusters = []
    for j in range(k):
        ns = names.get(j, [])
        if not ns:
            continue
        top = sorted(set(ns), key=lambda s: -ns.count(s))[:2]
        clusters.append({"id": j, "label": " / ".join(top), "size": len(ns)})
    lo, hi = xy.min(0), xy.max(0)
    xy = (xy - lo) / np.maximum(hi - lo, 1e-8) * 2 - 1
    pts = [{"id": eid, "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4), "cluster": int(lab[i])}
           for i, eid in enumerate(ids)]
    return {"points": pts, "clusters": clusters, "fallback": True}


def build_mem_index(conn, enc) -> MemIndex:
    rows = store.texts_for_embedding(conn)
    ids = [r[0] for r in rows]
    E = enc.encode_docs([r[1] for r in rows]) if rows else np.zeros((0, enc.dim), np.float32)
    return MemIndex(ids=ids, E=np.asarray(E, np.float32))


def _try_load_index(name):
    try:
        from atlas.index import load_index
    except Exception:
        return None
    try:
        return load_index(name)
    except Exception as e:
        log.info("atlas.index.load_index(%s) failed: %s", name, e)
        return None


class Engine:
    def __init__(self, spec: str | None = None, conn=None):
        self.spec = spec or config.ENCODER
        self.conn = conn or store.connect()
        self.enc = load_encoder(self.spec)
        self.name = getattr(self.enc, "name", self.spec)
        self.lock = threading.Lock()
        self.reload()

    def reload(self):
        idx = _try_load_index(self.name)
        if idx is not None and len(getattr(idx, "ids", [])):
            self.index = MemIndex(ids=list(idx.ids), E=np.asarray(idx.E, np.float32),
                                  mu=getattr(idx, "mu", None), sigma=getattr(idx, "sigma", None),
                                  map=getattr(idx, "map", None), built=getattr(idx, "map", None) is not None)
        else:
            self.index = build_mem_index(self.conn, self.enc)
        ix = self.index
        if ix.mu is None or ix.sigma is None:
            ix.mu, ix.sigma = hub_stats(ix.E, self.enc)
        ix.pos = {eid: i for i, eid in enumerate(ix.ids)}

    def get_map(self, fallback=True):
        ix = self.index
        if ix.map is None and fallback and len(ix.ids):
            ix.map = fallback_map(ix.E, ix.ids, self.conn)
        return ix.map

    def ensure_ids(self, new_ids):
        """Embed emails that arrived after the index was built (for watches)."""
        missing = [i for i in new_ids if i not in self.index.pos]
        if not missing:
            return
        rows = [store.get_email(self.conn, i) for i in missing]
        rows = [r for r in rows if r]
        if not rows:
            return
        with self.lock:
            V = self.enc.encode_docs([store.embed_text(r) for r in rows])
            ix = self.index
            Q = self.enc.encode_queries(probe_phrases())
            R = V @ Q.T
            ix.E = np.vstack([ix.E, V]).astype(np.float32)
            ix.mu = np.concatenate([ix.mu, R.mean(1)]).astype(np.float32)
            ix.sigma = np.concatenate([ix.sigma, R.std(1) + 1e-3]).astype(np.float32)
            for r in rows:
                ix.pos[r["id"]] = len(ix.ids)
                ix.ids.append(r["id"])


_engines: dict = {}
_lock = threading.Lock()


def resolve_spec(name: str | None) -> str:
    """UI passes index dir names ("base", "atlas-embed"); map them to load_encoder specs."""
    if not name or name in ("hash", "base") or "/" in name:
        return name or config.ENCODER
    for cand in (config.MODELS / name, config.MODELS / name.split("-")[0] / "", config.MODELS / "atlas-embed"):
        if cand.exists():
            return str(cand)
    return name


def get_engine(spec: str | None = None) -> Engine:
    spec = resolve_spec(spec) if spec else config.ENCODER
    with _lock:
        if spec not in _engines:
            _engines[spec] = Engine(spec)
        return _engines[spec]


def set_engine(engine: Engine, spec: str | None = None):
    """Tests inject an engine built on an in-memory DB."""
    _engines[spec or config.ENCODER] = engine


def available_encoders() -> list[str]:
    out = []
    if config.INDEX_DIR.exists():
        out = sorted(p.name for p in config.INDEX_DIR.iterdir() if (p / "emb.npy").exists())
    return out
