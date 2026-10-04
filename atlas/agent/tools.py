"""Grok tools shared by web chat, the voice agent and iMessage. See CONTRACT.md."""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
import uuid
from zoneinfo import ZoneInfo

import numpy as np

from atlas import config, store
from atlas.search import hybrid
from atlas.search import region as R
from atlas.search.engine import get_engine

log = logging.getLogger("atlas.agent")


def _fmt_date(ts):
    if not ts:
        return None
    return dt.datetime.fromtimestamp(int(ts), ZoneInfo(config.TIMEZONE)).strftime("%a %b %d %Y %H:%M")


def _slim(h):
    return {k: h.get(k) for k in ("id", "from", "subject", "snippet", "z", "prob", "facet")} | {"date": _fmt_date(h.get("date"))}


def search_region(positive=None, negative=None, after=None, before=None, query=None, k=10, **kw):
    positive = [p for p in (positive or []) if p]
    q = query or (positive[0] if positive else "")
    res = hybrid.search(q, positive, negative or [], {"after": after, "before": before, "from": kw.get("from")},
                        k=int(k or 10), mode="region")
    rg = res["region"]
    # Only region members are answers. Near misses (z 2 to 3) go in a separate list so the model can
    # mention them when asked, without presenting them as matches (e.g. a job assessment for "coding competition").
    hits = [_slim(h) for h in res["hits"] if h.get("member")]
    borderline = [_slim(h) for h in res["hits"] if not h.get("member") and h.get("z", 0) >= 2.0][:3]
    return {"region": {"size": rg["size"], "facet_hits": rg["facet_hits"], "nearest_clusters": rg["nearest_clusters"],
                       "related": rg["related"], "max_z": rg["max_z"]},
            "hits": hits, "borderline": borderline, "_full": res}


def is_related(topic, positive=None, negative=None, **_):
    return hybrid.is_related(topic, positive or [], negative or [])


def get_email(id, **_):
    r = get_engine().get_email(id)
    if not r:
        return {"error": f"no email {id}"}
    return {"id": r["id"], "from": r["from_name"] or r["from_addr"], "from_addr": r["from_addr"],
            "date": _fmt_date(r["date"]), "subject": r["subject"], "body": (r["body"] or "")[:2000]}


def list_clusters(**_):
    m = get_engine().get_map() or {}
    return [{"id": c["id"], "label": c.get("label"), "size": c.get("size")} for c in m.get("clusters", [])]


def _date_patterns(d: dt.date):
    mon, Mon = d.strftime("%b"), d.strftime("%B")
    day = d.day
    return [f"{mon} {day}", f"{Mon} {day}", f"{d.month}/{day}", d.isoformat(), f"{mon}. {day}"]


def _day_bounds(d: dt.date):
    tz = ZoneInfo(config.TIMEZONE)
    t0 = int(dt.datetime(d.year, d.month, d.day, tzinfo=tz).timestamp())
    return t0, t0 + 86400


def todays_agenda(date=None, **_):
    tz = ZoneInfo(config.TIMEZONE)
    d = dt.date.fromisoformat(date) if date else dt.datetime.now(tz).date()
    try:
        from atlas import calendar as cal  # ingest branch

        for fn in ("todays_agenda", "agenda", "agenda_for"):
            if hasattr(cal, fn):
                out = getattr(cal, fn)(d.isoformat())
                if isinstance(out, dict):
                    return out
    except Exception as e:
        log.debug("atlas.calendar unavailable: %s", e)
    conn = get_engine().conn
    t0, t1 = _day_bounds(d)
    events = [{"title": e["title"], "start": _fmt_date(e["start"]), "end": _fmt_date(e["end"]),
               "location": e.get("location")} for e in store.events_between(conn, t0, t1)]
    pats = _date_patterns(d)
    weekday = d.strftime("%A")
    emails, seen = [], set()
    rows = conn.execute("select id, from_name, from_addr, date, subject, body from emails order by date desc limit 3000").fetchall()
    recent = int(time.time()) - 21 * 86400
    for r in rows:
        text = f"{r['subject'] or ''} {r['body'] or ''}"
        hit = any(p in text for p in pats)
        # "Sunday" in a recent email very likely means this coming Sunday
        if not hit and (r["date"] or 0) >= min(recent, t0 - 14 * 86400) and (r["date"] or 0) <= t1 and weekday in text:
            hit = True
        if hit and r["id"] not in seen:
            seen.add(r["id"])
            emails.append({"id": r["id"], "from": r["from_name"] or r["from_addr"], "date": _fmt_date(r["date"]),
                           "subject": r["subject"], "snippet": (r["body"] or "")[:200]})
    return {"date": d.isoformat(), "events": events, "emails": emails[:15]}


def add_watch(name, positive=None, negative=None, **_):
    conn = get_engine().conn
    wid = uuid.uuid4().hex[:12]
    now = int(time.time())
    conn.execute("insert into watches values (?,?,?,?,?,?)",
                 (wid, name, json.dumps(list(positive or [name])), json.dumps(list(negative or [])), now, now))
    conn.commit()
    return {"id": wid, "name": name}


def list_watches(**_):
    conn = get_engine().conn
    return [{"id": r["id"], "name": r["name"], "positive": json.loads(r["positive"] or "[]"),
             "negative": json.loads(r["negative"] or "[]")}
            for r in conn.execute("select * from watches order by created")]


def delete_watch(id_or_name):
    conn = get_engine().conn
    n = conn.execute("delete from watches where id=? or name=?", (id_or_name, id_or_name)).rowcount
    conn.commit()
    return {"deleted": n}


def check_watches(new_ids):
    """For each watch, which of the new emails fall inside its region. Returns a list of dicts
    {watch_name, watch, id, from, subject, date, snippet, z}."""
    eng = get_engine()
    new_ids = list(new_ids or [])
    if not new_ids:
        return []
    eng.ensure_ids(new_ids)
    rows = [eng.index.pos[i] for i in new_ids if i in eng.index.pos]
    if not rows:
        return []
    out = []
    for w in list_watches():
        reg, raw, z = hybrid.region_scores(eng, w["name"], w["positive"], w["negative"])
        floor = R.floor_for(eng.name)
        for i in rows:
            if z[i] >= R.Z_MIN and raw[i] >= floor:
                e = eng.get_email(eng.index.ids[i]) or {}
                out.append({"watch_name": w["name"], "name": w["name"], "watch": w, "id": e.get("id"),
                            "from": e.get("from_name") or e.get("from_addr"), "subject": e.get("subject"),
                            "date": _fmt_date(e.get("date")), "snippet": e.get("snippet"), "z": round(float(z[i]), 2)})
    now = int(time.time())
    eng.conn.execute("update watches set last_checked=?", (now,))
    eng.conn.commit()
    return out


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


_arr = {"type": "array", "items": {"type": "string"}}

TOOL_SCHEMAS = [
    _fn("search_region", "Search the inbox with a topic region. Facets are short phrases in the words emails use. "
        "Returns region stats (size, facet_hits incl. zero-hit facets, nearest_clusters) and hits.",
        {"positive": {**_arr, "description": "facets that define the topic"},
         "negative": {**_arr, "description": "near-miss facets to exclude"},
         "after": {"type": "string", "description": "YYYY-MM-DD"}, "before": {"type": "string", "description": "YYYY-MM-DD"},
         "from": {"type": "string", "description": "sender name or domain substring"},
         "k": {"type": "integer", "default": 10}}, ["positive"]),
    _fn("is_related", "Calibrated yes/no: does the inbox contain anything about this topic?",
        {"topic": {"type": "string"}, "positive": {**_arr, "description": "optional facets"}}, ["topic"]),
    _fn("get_email", "Full cleaned body of one email (max 2000 chars).", {"id": {"type": "string"}}, ["id"]),
    _fn("list_clusters", "Clusters on the inbox map with labels and sizes.", {}),
    _fn("todays_agenda", "Calendar events plus emails mentioning a date. Default today.",
        {"date": {"type": "string", "description": "YYYY-MM-DD, default today"}}),
    _fn("add_watch", "Create a standing region; new mail inside it triggers a text to the user.",
        {"name": {"type": "string"}, "positive": _arr, "negative": _arr}, ["name", "positive"]),
    _fn("list_watches", "List the user's watches.", {}),
]

_TOOLS = {"search_region": search_region, "is_related": is_related, "get_email": get_email,
          "list_clusters": list_clusters, "todays_agenda": todays_agenda, "add_watch": add_watch,
          "list_watches": list_watches}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def run_tool(name, args=None, raw=False):
    """Dispatch a tool call. Returns a JSON-safe dict/list (raw=True keeps private keys like _full)."""
    if isinstance(args, str):
        args = json.loads(args or "{}")
    fn = _TOOLS.get(name)
    if not fn:
        return {"error": f"unknown tool {name}"}
    try:
        out = fn(**(args or {}))
    except Exception as e:
        log.exception("tool %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}
    return out if raw else _clean(out)
