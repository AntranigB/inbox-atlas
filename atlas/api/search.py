"""Search, related, map, agenda, watches and chat routes (owned by search-agent)."""

from __future__ import annotations

import logging
import threading

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atlas import config, store
from atlas.db import qlog
from atlas.agent import grok, tools
from atlas.search import hybrid
from atlas.search.engine import available_encoders, get_engine

log = logging.getLogger("atlas.api")
router = APIRouter()


class AskBody(BaseModel):
    text: str
    channel: str = "web"
    history: list | None = None
    session_id: str | None = None
    user_handle: str | None = None


class SearchBody(BaseModel):
    query: str = ""
    mode: str = "region"
    k: int = 20
    positive: list[str] | None = None  # None = ask Grok to expand
    negative: list[str] | None = None
    filters: dict | None = None
    encoder: str | None = None


class WatchBody(BaseModel):
    name: str
    positive: list[str] | None = None
    negative: list[str] | None = None


def _facet_points(eng, labels):
    """Place each facet on the 2D map at the weighted mean of its 5 nearest emails."""
    m = eng.get_map()
    if not m or not labels or not len(eng.index.E):
        return []
    xy = {p["id"]: (p["x"], p["y"]) for p in m.get("points", [])}
    F = eng.enc.encode_queries(labels)
    S = F @ eng.index.E.T
    out = []
    for j, lab in enumerate(labels):
        top = np.argsort(-S[j])[:5]
        pts = [(xy[eng.index.ids[i]], max(float(S[j, i]), 1e-3)) for i in top if eng.index.ids[i] in xy]
        if not pts:
            continue
        w = sum(p[1] for p in pts)
        out.append({"label": lab, "x": sum(p[0][0] * p[1] for p in pts) / w, "y": sum(p[0][1] * p[1] for p in pts) / w})
    return out


@router.post("/api/search")
def api_search(b: SearchBody):
    mode = b.mode if b.mode in hybrid.MODES else "region"
    eng = get_engine(b.encoder) if b.encoder else get_engine()
    exp = None
    pos, neg, filters = b.positive, b.negative, dict(b.filters or {})
    if mode in ("region", "hybrid") and pos is None and b.query.strip():
        exp = grok.expand(b.query)
        pos, neg = exp["positive"], exp["negative"]
        for kf, v in (exp.get("filters") or {}).items():
            filters.setdefault(kf, v)
    with qlog.timed(f"search_{mode}", b.query) as row:
        res = hybrid.search(b.query, pos or [], neg or [], filters, k=b.k, mode=mode, engine=eng)
        row.update(n_facets=len(pos or []) or None, region_size=(res.get("region") or {}).get("size"))
    res["expansion"] = exp
    if res.get("region"):
        res["facet_points"] = _facet_points(eng, [b.query] + list(pos or []) if b.query else list(pos or []))
    return res


@router.get("/api/related")
def api_related(topic: str, encoder: str | None = None):
    exp = grok.expand(topic)
    eng = get_engine(encoder) if encoder else get_engine()
    out = hybrid.is_related(topic, exp["positive"], exp["negative"], engine=eng)
    out["facets"] = {"positive": exp["positive"], "negative": exp["negative"]}
    return out


@router.post("/api/ask")
def api_ask(b: AskBody):
    return grok.ask(b.text, b.channel, b.history, session_id=b.session_id, user_handle=b.user_handle)


@router.get("/api/map")
def api_map(encoder: str | None = None):
    eng = get_engine(encoder) if encoder else get_engine()
    m = eng.get_map()
    if not m:
        return {"points": [], "clusters": [], "built": False}
    return {**m, "built": bool(m.get("points"))}


@router.get("/api/email/{eid}")
def api_email(eid: str):
    r = get_engine().get_email(eid)
    if not r:
        raise HTTPException(404, "no such email")
    return r


@router.get("/api/agenda")
def api_agenda(date: str | None = None):
    return tools.todays_agenda(date)


@router.get("/api/watches")
def api_watches():
    return tools.list_watches()


@router.post("/api/watches")
def api_add_watch(b: WatchBody):
    return tools.add_watch(b.name, b.positive or [b.name], b.negative or [])


@router.delete("/api/watches/{wid}")
def api_del_watch(wid: str):
    return tools.delete_watch(wid)


@router.get("/api/encoders")
def api_encoders():
    return {"default": config.ENCODER, "available": available_encoders()}


def on_startup():
    def warm():
        try:
            get_engine()
        except Exception as e:
            log.warning("engine warmup failed: %s", e)
    threading.Thread(target=warm, daemon=True).start()
