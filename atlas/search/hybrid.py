"""Keyword (BM25), plain embedding cosine, region, and RRF fusion over one inbox."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np

from atlas import config

from atlas import store
from atlas.search import region as R
from atlas.search.engine import Engine, get_engine

RRF_K = 60
MODES = ("keyword", "embed", "region", "hybrid")


def _ts(v, end=False):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    d = dt.date.fromisoformat(str(v)[:10])
    t = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()
    return int(t + (86400 if end else 0))


def filter_mask(eng: Engine, filters: dict | None) -> np.ndarray:
    ids = eng.index.ids
    mask = np.ones(len(ids), bool)
    f = {k: v for k, v in (filters or {}).items() if v not in (None, "", [])}
    if not f:
        return mask
    after, before, frm = _ts(f.get("after")), _ts(f.get("before"), end=True), f.get("from")
    srcs = f.get("sources") or f.get("source")
    srcs = {srcs} if isinstance(srcs, str) else set(srcs or [])
    meta = eng.email_meta()
    for i, eid in enumerate(ids):
        date, addr, name, src = meta.get(eid, (None, "", "", None))
        if srcs and source_kind(src) not in srcs:
            mask[i] = False
        elif after and (date or 0) < after:
            mask[i] = False
        elif before and (date or 0) >= before:
            mask[i] = False
        elif frm and frm.lower() not in f"{addr or ''} {name or ''}".lower():
            mask[i] = False
    return mask


def source_kind(src) -> str:
    """'obsidian' for vault notes, 'gmail' for every mail source (gmail, imap, fixture)."""
    return "obsidian" if src == "obsidian" else "gmail"


def rrf(rank_lists, k=RRF_K):
    score = {}
    for ranks in rank_lists:
        for r, eid in enumerate(ranks):
            score[eid] = score.get(eid, 0.0) + 1.0 / (k + r + 1)
    return sorted(score, key=lambda e: -score[e]), score


def _hit(eng, eid, **extra):
    r = eng.get_email(eid) or {}
    h = {"id": eid, "from": r.get("from_name") or r.get("from_addr"), "from_addr": r.get("from_addr"),
         "date": r.get("date"), "subject": r.get("subject"), "snippet": (r.get("snippet") or (r.get("body") or "")[:200]),
         "source": source_kind(r.get("source"))}
    h.update({k: v for k, v in extra.items() if v is not None})
    return h


def keyword_ranks(eng, text, mask, n=200):
    pos = eng.index.pos
    out = []
    for eid, s in eng.fts(text, n):
        i = pos.get(eid)
        if i is not None and mask[i]:
            out.append((eid, s))
    return out


def region_scores(eng: Engine, query: str, positive=None, negative=None, kind=None):
    """Returns (region, raw, z) over every email in the index.

    Heuristic by default: hub z is calibrated against the heuristic formula, and the learned region
    (opt in with ATLAS_REGION=learned) scores on a different scale and did not rank better on held-out topics.
    """
    kind = kind or config.env("ATLAS_REGION", "heuristic")
    pos_labels = list(dict.fromkeys([query] + list(positive or []) if query else list(positive or [])))
    neg_labels = [n for n in (negative or []) if n]
    P = eng.enc.encode_queries(pos_labels)
    N = eng.enc.encode_queries(neg_labels) if neg_labels else None
    reg = R.build_region(P, N, kind=kind, labels=pos_labels, neg_labels=neg_labels)
    E = eng.index.E
    raw = reg.score(E) if len(E) else np.zeros(0, np.float32)
    mu, sigma = eng.null_stats(len(pos_labels)) if len(E) else (0, 1)
    z = R.hub_z(raw, mu, sigma) if len(E) else raw
    return reg, raw, z


def nearest_clusters(eng, member_ids, top=3):
    m = eng.get_map()
    if not m or not member_ids:
        return []
    cl = {p["id"]: p["cluster"] for p in m.get("points", [])}
    labels = {c["id"]: c.get("label", str(c["id"])) for c in m.get("clusters", [])}
    counts = {}
    for eid in member_ids:
        c = cl.get(eid)
        if c is not None and c != -1:
            counts[c] = counts.get(c, 0) + 1
    return [labels.get(c, str(c)) for c in sorted(counts, key=lambda c: -counts[c])[:top]]


def search(query: str, positive=None, negative=None, filters=None, k=10, mode="region",
           engine: Engine | None = None, encoder: str | None = None) -> dict:
    eng = engine or get_engine(encoder)
    mask = filter_mask(eng, filters)
    ix = eng.index
    out = {"mode": mode, "encoder": eng.name, "query": query, "facets": {"positive": list(positive or []),
           "negative": list(negative or [])}, "filters": filters or {}}
    if mode == "keyword":
        kw = keyword_ranks(eng, " ".join([query] + list(positive or [])), mask)
        out["hits"] = [_hit(eng, eid, bm25=round(float(s), 3)) for eid, s in kw[:k]]
        out["region"] = None
        return out
    if mode == "embed":
        q = eng.enc.encode_queries([query])[0]
        if getattr(eng.backend, "kind", "") == "pg":  # pgvector HNSW knn
            try:
                nn = eng.backend.knn(q, k, encoder=eng.name, filters=filters)
                out["hits"] = [_hit(eng, eid, cos=round(c, 3)) for eid, c in nn]
                out["region"] = None
                return out
            except Exception:
                pass
        cos = ix.E @ q if len(ix.E) else np.zeros(0)
        cos = np.where(mask, cos, -np.inf)
        order = np.argsort(-cos)[:k]
        out["hits"] = [_hit(eng, ix.ids[i], cos=round(float(cos[i]), 3)) for i in order if np.isfinite(cos[i])]
        out["region"] = None
        return out

    reg, raw, z = region_scores(eng, query, positive, negative)
    verdict = R.related_verdict(np.where(mask, raw, -1.0), np.where(mask, z, -99.0), eng.name)
    member = verdict.pop("member") & mask
    prob = reg.prob(ix.E) if len(ix.E) else None
    heur = getattr(reg, "_heur", reg)
    facet_idx = heur.facet_of(ix.E) if len(ix.E) else None
    labels = heur.labels
    zm = np.where(mask, z, -np.inf)
    order = list(np.argsort(-(zm + 1e-3 * raw)))
    if mode == "hybrid":
        kw = [eid for eid, _ in keyword_ranks(eng, " ".join([query] + list(positive or [])), mask)]
        fused, _ = rrf([[ix.ids[i] for i in order[:200] if np.isfinite(zm[i])], kw])
        order = [ix.pos[e] for e in fused]
    hits = []
    for i in order[:k]:
        if not np.isfinite(zm[i]):
            continue
        hits.append(_hit(eng, ix.ids[i], z=round(float(z[i]), 2), raw=round(float(raw[i]), 3),
                         prob=None if prob is None else round(float(prob[i]), 3),
                         facet=labels[int(facet_idx[i])] if facet_idx is not None else None,
                         member=bool(member[i])))
    member_ids = [ix.ids[i] for i in np.flatnonzero(member)]
    desc = heur.describe(ix.E, member) if hasattr(heur, "describe") else {}
    if reg is not heur and hasattr(reg, "describe"):
        try:
            desc["learned"] = reg.describe()
        except Exception:
            pass
    out["hits"] = hits
    out["region"] = {"size": int(member.sum()), "tau_z": R.Z_MIN, "facet_hits": desc.get("facet_hits", {}),
                     "nearest_clusters": nearest_clusters(eng, member_ids), "member_ids": member_ids[:200],
                     "describe": {k2: v for k2, v in desc.items() if k2 != "facet_hits"}, **verdict}
    return out


def is_related(topic: str, positive=None, negative=None, engine=None, encoder=None) -> dict:
    res = search(topic, positive, negative, k=5, mode="region", engine=engine, encoder=encoder)
    rg = res["region"]
    return {"related": rg["related"], "confidence": rg["confidence"], "count": rg["count"], "max_z": rg["max_z"],
            "examples": [{"id": h["id"], "from": h["from"], "subject": h["subject"], "z": h["z"]}
                         for h in res["hits"] if h.get("member")][:3]}


def dumps(o):
    return json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x))
