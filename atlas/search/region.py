"""Query regions: a topic list compressed into a center, facets and anti-facets (PLAN 3b, 3c)."""

from __future__ import annotations

import numpy as np

from atlas import config

# related? thresholds. Z_MIN is in hub-corrected z units, FLOOR is raw region score.
# Tuned on the fixture: see tests/test_region.py and eval/run_eval.py.
Z_MIN = 3.0
FLOOR = {"hash": 0.30, "base": 0.50, "default": 0.55}

W_FACET, W_CORE, W_NEG = 0.6, 0.4, 1.0


def floor_for(encoder_name: str) -> float:
    name = (encoder_name or "").split("-")[0]
    return FLOOR.get(name, FLOOR["default"])


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        return v / max(float(np.linalg.norm(v)), 1e-8)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


class Region:
    """Interface shared with atlas/model/region_encoder.py (LearnedRegionModel)."""

    def score(self, E: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def prob(self, E: np.ndarray) -> np.ndarray | None:
        return None

    def describe(self) -> dict:
        return {}

    def facet_of(self, E: np.ndarray) -> np.ndarray | None:
        """Index of the facet each row matched best, if the region has facets."""
        return None


class HeuristicRegion(Region):
    def __init__(self, pos_vecs, neg_vecs=None, labels=None, neg_labels=None):
        self.P = _norm(np.atleast_2d(pos_vecs))
        nv = np.asarray(neg_vecs if neg_vecs is not None else np.zeros((0, self.P.shape[1])), dtype=np.float32)
        self.N = _norm(nv.reshape(-1, self.P.shape[1])) if nv.size else np.zeros((0, self.P.shape[1]), np.float32)
        self.c = _norm(self.P.mean(0))
        self.labels = list(labels) if labels else [f"facet {i}" for i in range(len(self.P))]
        self.neg_labels = list(neg_labels) if neg_labels else [f"anti {i}" for i in range(len(self.N))]

    def _parts(self, E):
        S = E @ self.P.T
        s_facet = S.max(1)
        s_core = E @ self.c
        s_neg = (E @ self.N.T).max(1) if len(self.N) else np.zeros(len(E), np.float32)
        return S, s_facet, s_core, s_neg

    def score(self, E):
        _, s_facet, s_core, s_neg = self._parts(E)
        return W_FACET * s_facet + W_CORE * s_core - W_NEG * np.maximum(0, s_neg - s_facet)

    def facet_of(self, E):
        return (E @ self.P.T).argmax(1)

    def describe(self, E=None, member_mask=None):
        """Region compression: center, facets, anti-facets, spread, and per-facet hit counts."""
        sims = self.P @ self.c
        d = {
            "kind": "heuristic",
            "center_dim": int(self.c.shape[0]),
            "facets": [{"label": l, "to_center": round(float(s), 3)} for l, s in zip(self.labels, sims)],
            "anti_facets": self.neg_labels,
            "spread": round(float(1 - sims.mean()), 3),
        }
        if E is not None and member_mask is not None:
            f = self.facet_of(E)
            counts = {l: 0 for l in self.labels}
            for i in np.flatnonzero(member_mask):
                counts[self.labels[f[i]]] += 1
            d["facet_hits"] = counts
        return d


def _learned_path():
    return config.MODELS / "atlas-embed" / "region.pt"


_learned = None


def build_region(pos_vecs, neg_vecs=None, kind="heuristic", labels=None, neg_labels=None) -> Region:
    global _learned
    if kind in ("learned", "auto") and _learned_path().exists():
        try:
            if _learned is None:
                from atlas.model.region_encoder import LearnedRegionModel

                _learned = LearnedRegionModel.load(str(_learned_path()))
            r = _learned.build(pos_vecs, neg_vecs)
            # keep facet labels around for describe/facet_of
            r._heur = HeuristicRegion(pos_vecs, neg_vecs, labels, neg_labels)
            return r
        except Exception:
            if kind == "learned":
                raise
    return HeuristicRegion(pos_vecs, neg_vecs, labels, neg_labels)


def hub_z(raw: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (raw - mu) / np.maximum(sigma, 1e-4)


def related_verdict(raw, z, encoder_name, z_min=Z_MIN):
    """related? = top z >= z_min and its raw >= floor. Confidence is a squashed margin."""
    floor = floor_for(encoder_name)
    member = (z >= z_min) & (raw >= floor)
    if len(z) == 0:
        return {"related": False, "confidence": 0.5, "count": 0, "max_z": 0.0, "top_raw": 0.0, "floor": floor, "member": member}
    top = int(np.argmax(np.where(raw >= floor, z, -np.inf))) if member.any() else int(np.argmax(z))
    mz, mr = float(z[top]), float(raw[top])
    related = bool(member.any())
    margin = min(mz - z_min, (mr - floor) * 20) if related else max(mz - z_min, (mr - floor) * 20)
    conf = 1 / (1 + np.exp(-abs(margin)))
    return {"related": related, "confidence": round(float(conf), 3), "count": int(member.sum()),
            "max_z": round(mz, 2), "top_raw": round(mr, 3), "floor": floor, "member": member}
