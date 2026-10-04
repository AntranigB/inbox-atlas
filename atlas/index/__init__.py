"""Embedding index: data/index/<encoder_name>/ with emb.npy, ids.json, meta.json, hub.npz, map.json.

    from atlas.index import build_index, load_index
    build_index(load_encoder("base"))
    idx = load_index("base"); idx.E, idx.ids, idx.mu, idx.sigma, idx.map
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from atlas import config, store

PROBES_PATH = Path(__file__).with_name("probes.txt")
EMBED_BATCH = 256
_lock = threading.Lock()
log = logging.getLogger("atlas.index")


def _mirror_pg(name, ids, E, conn=None):
    """When ATLAS_DB=pg, copy these emails and vectors into Postgres too. Never fails the build."""
    try:
        from atlas.db.backend import mirror_to_pg, pg_enabled

        if pg_enabled() and len(ids):
            mirror_to_pg(name, list(ids), E, conn)
    except Exception as e:
        log.warning("postgres mirror failed for %s: %s", name, e)


def index_dir(name):
    return config.INDEX_DIR / name


def default_name():
    spec = config.ENCODER
    if spec in ("hash", "base"):
        return spec if (spec == "hash" or not config.DIM) else f"{spec}-{config.DIM}"
    from atlas.model.encoder import load_encoder

    return load_encoder(spec).name


def load_probes():
    return [l.strip() for l in PROBES_PATH.read_text().splitlines() if l.strip() and not l.startswith("#")]


def hub_stats(E, P):
    """Per-email mean and std of similarity to the probe topics (hubness correction)."""
    R = E @ P.T
    return R.mean(1).astype(np.float32), np.maximum(R.std(1), 1e-4).astype(np.float32)


def _save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _save_npy(path, arr):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _save_npz(path, **arrs):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def embed_texts(encoder, texts, batch=EMBED_BATCH, verbose=False):
    out, t0 = [], time.time()
    for i in range(0, len(texts), batch):
        out.append(np.asarray(encoder.encode_docs(texts[i:i + batch]), dtype=np.float32))
        if verbose:
            done = min(i + batch, len(texts))
            sys.stdout.write(f"\r  embed: {done}/{len(texts)}  {done / max(time.time() - t0, 1e-6):.0f}/s   ")
            sys.stdout.flush()
    if verbose and texts:
        print()
    return np.concatenate(out) if out else np.zeros((0, encoder.dim), np.float32)


class Index:
    def __init__(self, name, ids, E, mu, sigma, P, map_):
        self.name = name
        self.ids = list(ids)
        self.E = np.asarray(E, dtype=np.float32)
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.P = P
        self.map = map_
        self.pos = {i: k for k, i in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    @property
    def dim(self):
        return self.E.shape[1]

    def vec(self, id):
        k = self.pos.get(id)
        return None if k is None else self.E[k]

    def add(self, ids, vecs, save=True):
        """Append (or replace) emails. Hub stats come from the stored probes; map position is the
        mean of the 5 nearest mapped points, cluster from the nearest one."""
        ids = list(ids)
        vecs = np.asarray(vecs, dtype=np.float32).reshape(len(ids), -1)
        if not ids:
            return
        with _lock:
            if self.P is not None and len(self.P):
                mu, sigma = hub_stats(vecs, self.P)
            else:
                mu, sigma = np.zeros(len(ids), np.float32), np.ones(len(ids), np.float32)
            pts = {p["id"]: p for p in self.map.get("points", [])} if self.map else {}
            if pts and len(self.ids):
                mapped = np.array([k for k, i in enumerate(self.ids) if i in pts])
                xy = np.array([[pts[self.ids[k]]["x"], pts[self.ids[k]]["y"]] for k in mapped]) if len(mapped) else None
                cl = [pts[self.ids[k]]["cluster"] for k in mapped]
                if len(mapped):
                    S = vecs @ self.E[mapped].T
                    for j, id in enumerate(ids):
                        nn = np.argsort(-S[j])[:5]
                        x, y = xy[nn].mean(0)
                        pts[id] = {"id": id, "x": round(float(x), 4), "y": round(float(y), 4), "cluster": cl[nn[0]]}
            new_rows = []
            for j, id in enumerate(ids):
                k = self.pos.get(id)
                if k is not None:
                    self.E[k], self.mu[k], self.sigma[k] = vecs[j], mu[j], sigma[j]
                else:
                    self.pos[id] = len(self.ids)
                    self.ids.append(id)
                    new_rows.append(j)
            if new_rows:
                self.E = np.vstack([self.E.reshape(-1, vecs.shape[1]), vecs[new_rows]])
                self.mu = np.concatenate([self.mu, mu[new_rows]])
                self.sigma = np.concatenate([self.sigma, sigma[new_rows]])
            if self.map and pts:
                self.map["points"] = list(pts.values())
                sizes = {}
                for p in self.map["points"]:
                    sizes[p["cluster"]] = sizes.get(p["cluster"], 0) + 1
                for c in self.map.get("clusters", []):
                    c["size"] = sizes.get(c["id"], 0)
            if save:
                self.save()
        _mirror_pg(self.name, ids, vecs)

    def save(self):
        d = index_dir(self.name)
        d.mkdir(parents=True, exist_ok=True)
        _save_npy(d / "emb.npy", self.E.astype(np.float16))
        _save_json(d / "ids.json", self.ids)
        _save_npz(d / "hub.npz", mu=self.mu, sigma=self.sigma,
                  P=self.P if self.P is not None else np.zeros((0, self.dim), np.float32))
        if self.map:
            _save_json(d / "map.json", self.map)
        meta_p = d / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"encoder": self.name}
        meta.update({"dim": int(self.dim), "n": len(self.ids), "updated_at": int(time.time())})
        _save_json(meta_p, meta)


def build_index(encoder, conn=None, with_map=True, grok=False, verbose=False):
    """Embed every email, compute hub stats and the 2D map, write data/index/<encoder.name>/."""
    from atlas.index.mapping import build_map

    conn = conn or store.connect()
    pairs = store.texts_for_embedding(conn)
    ids = [i for i, _ in pairs]
    if verbose:
        print(f"embedding {len(ids)} emails with {encoder.name} (dim {encoder.dim})")
    E = embed_texts(encoder, [t for _, t in pairs], verbose=verbose)
    P = np.asarray(encoder.encode_queries(load_probes()), dtype=np.float32)
    mu, sigma = hub_stats(E, P) if len(E) else (np.zeros(0, np.float32), np.zeros(0, np.float32))
    map_ = build_map(conn, ids, E, grok=grok, verbose=verbose) if with_map else {"points": [], "clusters": []}
    idx = Index(encoder.name, ids, E, mu, sigma, P, map_)
    d = index_dir(encoder.name)
    d.mkdir(parents=True, exist_ok=True)
    _save_json(d / "meta.json", {"encoder": encoder.name, "dim": int(E.shape[1]) if E.ndim == 2 else encoder.dim,
                                 "n": len(ids), "built_at": int(time.time())})
    idx.save()
    _mirror_pg(encoder.name, ids, E, conn)
    _cache.pop(encoder.name, None)
    return idx


_cache = {}


def load_index(encoder_name=None, reload=False):
    name = encoder_name or default_name()
    if not reload and name in _cache:
        return _cache[name]
    d = index_dir(name)
    if not (d / "emb.npy").exists():
        raise FileNotFoundError(f"no index at {d}. Build it: uv run python -m atlas.index.build --encoder {name}")
    E = np.load(d / "emb.npy").astype(np.float32)
    ids = json.loads((d / "ids.json").read_text())
    hub = np.load(d / "hub.npz")
    P = hub["P"] if "P" in hub.files and len(hub["P"]) else None
    map_p = d / "map.json"
    map_ = json.loads(map_p.read_text()) if map_p.exists() else {"points": [], "clusters": []}
    idx = Index(name, ids, E, hub["mu"], hub["sigma"], P, map_)
    _cache[name] = idx
    return idx
