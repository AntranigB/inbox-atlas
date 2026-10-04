"""Retrieval and calibration metrics (numpy only)."""

import numpy as np


def topk_idx(S, k):
    k = min(k, S.shape[1])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(S, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def retrieval(S, rel, k=10):
    """S: (nq, nd) scores. rel: list of sets of relevant column indices.
    Recall@k is |rel in top k| / min(|rel|, k) so large topics are not capped near zero."""
    top = topk_idx(S, k)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rs, ns = [], []
    for i, r in enumerate(rel):
        if not r:
            continue
        hits = np.array([1.0 if j in r else 0.0 for j in top[i]])
        rs.append(hits.sum() / min(len(r), k))
        ideal = disc[: min(len(r), k)].sum()
        ns.append((hits * disc[: len(hits)]).sum() / ideal)
    return {"recall@10": float(np.mean(rs)) if rs else 0.0, "ndcg@10": float(np.mean(ns)) if ns else 0.0, "n": len(rs)}


def auroc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ece(y, p, bins=15):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_platt(y, s, iters=500):
    """Fit sigmoid(a*s + c) by Newton steps. Returns (a, c)."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    a, c = 1.0, 0.0
    for _ in range(iters):
        z = np.clip(a * s + c, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = np.array([((p - y) * s).mean(), (p - y).mean()])
        w = p * (1 - p) + 1e-9
        H = np.array([[(w * s * s).mean(), (w * s).mean()], [(w * s).mean(), w.mean()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, c = a - step[0], c - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(c)
