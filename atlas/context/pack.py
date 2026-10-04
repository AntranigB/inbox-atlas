"""Token-budgeted context packs for LLM agents.

    from atlas.context import build_context
    pack = build_context("when is my flight to Japan?", budget_tokens=800)
    pack["context"]  # paste this into the agent prompt

Pipeline: Grok facets (cached) -> region scoring over emails and vault notes -> calibrated
related? verdict -> for each hit keep only the sentences that score highest against the same
region -> greedy packing under the budget. When the region is empty the pack says so with zero
items, which is the agent's signal to stop searching.

Tokens are counted with tiktoken cl100k_base (falls back to chars / 4 if tiktoken is missing).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from functools import lru_cache

import numpy as np

from atlas import store
from atlas.search import hybrid
from atlas.search import region as R
from atlas.search.engine import get_engine

SOURCES = ("gmail", "obsidian")
MAX_SENTS_PER_ITEM = 4
WHOLE_BODY_TOKENS = 70  # bodies this short are kept whole: cheaper than fragmenting them
SENT_MIN_CHARS = 12


@lru_cache(maxsize=1)
def _enc():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    e = _enc()
    return len(e.encode(text, disallowed_special=())) if e else max(1, len(text) // 4)


def truncate_tokens(text: str, max_tokens: int) -> str:
    e = _enc()
    if not e:
        return text[: max_tokens * 4]
    ids = e.encode(text, disallowed_special=())
    return text if len(ids) <= max_tokens else e.decode(ids[:max_tokens]) + " ..."


# ---------- documents and uris ----------

def uri_of(row: dict) -> str:
    rid = row.get("id") or ""
    return rid[4:] if rid.startswith("obs:") else f"gmail:{rid}"


def _fmt_date(ts):
    if not ts:
        return None
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


def doc_text(row: dict) -> str:
    """What an agent would read if it opened the item whole."""
    if row.get("source") == "obsidian":
        return f"# {row.get('subject') or ''}\n{row.get('body') or ''}"
    who = row.get("from_name") or row.get("from_addr") or ""
    return f"Subject: {row.get('subject') or ''}\nFrom: {who}\nDate: {_fmt_date(row.get('date'))}\n\n{row.get('body') or ''}"


def full_note_text(conn, thread_id: str) -> str:
    rows = store.thread_rows(conn, thread_id)
    return "\n\n".join(f"## {r['subject']}\n{r['body']}" for r in rows)


def get_doc(uri: str, max_tokens: int = 1500, conn=None) -> dict:
    """Full text of one email, one note section (path#anchor) or a whole note (path), capped."""
    conn = conn or get_engine().conn
    uri = (uri or "").strip()
    if uri.startswith("gmail:"):
        row = store.get_email(conn, uri[6:])
        text = doc_text(row) if row else None
    elif "#" in uri:
        row = store.get_email(conn, f"obs:{uri}")
        text = doc_text(row) if row else None
    else:
        rows = store.thread_rows(conn, f"obs:{uri}")
        row = rows[0] if rows else None
        text = full_note_text(conn, f"obs:{uri}") if rows else None
    if not row:
        return {"uri": uri, "error": "not found"}
    full = count_tokens(text)
    text = truncate_tokens(text, max_tokens)
    return {"uri": uri, "source": hybrid.source_kind(row.get("source")), "title": row.get("subject"),
            "date": _fmt_date(row.get("date")), "text": text, "tokens": count_tokens(text), "full_tokens": full,
            "truncated": full > max_tokens}


# ---------- sentence extraction ----------

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(*])|\n+")


def sentences(text: str) -> list[str]:
    out = []
    for s in _SPLIT.split(text or ""):
        s = re.sub(r"\s+", " ", s).strip(" -*>\t")
        if len(s) >= SENT_MIN_CHARS and not s.startswith("---"):
            out.append(s)
    return out


def _header(item) -> str:
    bits = [f"[{item['source']}] {item['title']}"]
    meta = ", ".join(x for x in (item.get("date"), item.get("from")) if x)
    if meta:
        bits.append(f"({meta})")
    bits.append(f"<{item['uri']}>")
    return " ".join(bits)


def render(items) -> str:
    return "\n\n".join(f"{_header(it)}\n{it['excerpt']}" for it in items)


def _tier(budget: int, k: int):
    """Budget is the recall knob: (relative z cut, sentences per item, near-member z, max items).
    Tight budgets keep the core of the region and the single best sentences; looser budgets add the
    rest of the region, then near members (z >= 2) and more sentences per item."""
    if budget <= 400:
        return 0.6, 2, None, k
    if budget <= 1000:
        return 0.35, MAX_SENTS_PER_ITEM, None, k
    return 0.0, MAX_SENTS_PER_ITEM + 2, 2.0, int(k * 1.5)


def _excerpt(ss, idxs) -> str:
    """Selected sentences in reading order, '...' where text was skipped."""
    out, prev = [], None
    for i in sorted(idxs):
        if prev is not None and i - prev > 1:
            out.append("...")
        out.append(ss[i])
        prev = i
    return " ".join(out)


def _not_found(sources, verdict) -> str:
    return (f"Nothing about this in {' or '.join(sources)}: best match z={verdict['max_z']:.1f} "
            f"is under the {R.Z_MIN:.1f} threshold. Stop searching.")


def build_context(question: str, budget_tokens: int = 800, sources=SOURCES, k: int = 8,
                  engine=None, facets: dict | None = None, use_grok: bool = True) -> dict:
    """Minimal context that answers `question`, packed under `budget_tokens`.

    Returns {answerable, confidence, reason, region, items, context, tokens, tokens_saved_vs_naive}.
    tokens counts the rendered `context` string. tokens_saved_vs_naive compares it with reading the
    same top-k documents whole (full email, or the full note for a vault hit).
    """
    from atlas.agent import grok

    eng = engine or get_engine()
    sources = tuple(s for s in (sources or SOURCES) if s in SOURCES) or SOURCES
    exp = facets or (grok.expand(question) if use_grok else {"positive": [], "negative": []})
    pos, neg = list(exp.get("positive") or []), list(exp.get("negative") or [])
    res = hybrid.search(question, pos, neg, {"sources": list(sources)}, k=max(k * 3, 12), mode="region", engine=eng)
    rg = res["region"]
    region = {"facets": [question] + [p for p in pos if p != question], "anti_facets": neg, "size": rg["size"],
              "max_z": rg["max_z"], "related": rg["related"],
              "facet_hits": {f: n for f, n in (rg.get("facet_hits") or {}).items() if n},
              "nearest_clusters": rg.get("nearest_clusters", [])}
    base = {"question": question, "confidence": rg["confidence"], "region": region, "budget_tokens": budget_tokens,
            "sources": list(sources)}

    rel_z, max_sents, near_z, k = _tier(int(budget_tokens), k)
    hits = [h for h in res["hits"] if h.get("member") or (near_z is not None and h.get("z", -99) >= near_z)]
    if hits:  # small budgets keep only the core of the region; bigger budgets buy recall
        top_z = max(h["z"] for h in hits)
        hits = [h for h in hits if h["z"] >= rel_z * top_z][:k]
    if not rg["related"] or not hits:
        reason = _not_found(sources, rg)
        return {**base, "answerable": False, "reason": reason, "items": [], "context": reason,
                "tokens": count_tokens(reason), "tokens_saved_vs_naive": 0}

    rows = [store.get_email(eng.conn, h["id"]) or {} for h in hits]
    # score every candidate sentence against the same region the hits came from
    labels = region["facets"]
    P = eng.enc.encode_queries(labels)
    Nv = eng.enc.encode_queries(neg) if neg else None
    reg = R.HeuristicRegion(P, Nv, labels, neg)
    per = []
    flat = []
    for j, r in enumerate(rows):
        ss = sentences(r.get("body") or "")
        per.append(ss)
        flat += [(j, i, s) for i, s in enumerate(ss)]
    S = reg.score(eng.enc.encode_docs([f"{rows[j].get('subject') or ''}: {s}" for j, _, s in flat])) if flat else np.zeros(0)
    scores = {}
    for (j, i, _), v in zip(flat, S):
        scores.setdefault(j, {})[i] = float(v)
    cut = float(S.mean() + 0.5 * S.std()) if len(S) else 0.0

    items, used, naive = [], 0, 0
    budget = int(budget_tokens)
    for j, (h, r) in enumerate(zip(hits, rows)):
        naive += count_tokens(full_note_text(eng.conn, r["thread_id"]) if r.get("source") == "obsidian"
                              else doc_text(r))
        item = {"source": hybrid.source_kind(r.get("source")), "uri": uri_of(r), "title": r.get("subject") or "",
                "date": _fmt_date(r.get("date")),
                "from": None if r.get("source") == "obsidian" else (r.get("from_name") or r.get("from_addr")),
                "z": h.get("z"), "facet": h.get("facet")}
        ss, sc = per[j], scores.get(j, {})
        body = " ".join((r.get("body") or "").split())
        if count_tokens(body) <= WHOLE_BODY_TOKENS or not sc:
            keep = None
            item["excerpt"] = body
        else:
            ranked = sorted(sc, key=lambda i: -sc[i])
            keep = [ranked[0]] + [i for i in ranked[1:max_sents] if sc[i] >= cut]
            item["excerpt"] = _excerpt(ss, keep)
        cost = count_tokens(_header(item) + "\n" + item["excerpt"]) + (2 if items else 0)
        # drop the weakest sentences until the item fits what is left of the budget
        while keep and used + cost > budget and len(keep) > 1:
            keep.remove(min(keep, key=lambda i: sc[i]))
            item["excerpt"] = _excerpt(ss, keep)
            cost = count_tokens(_header(item) + "\n" + item["excerpt"]) + (2 if items else 0)
        if used + cost > budget:
            if items:
                continue  # a later, shorter item may still fit
            item["excerpt"] = truncate_tokens(item["excerpt"], max(budget - count_tokens(_header(item)) - 4, 8))
            cost = count_tokens(_header(item) + "\n" + item["excerpt"])
        items.append(item)
        used += cost
    context = render(items)
    tokens = count_tokens(context)
    return {**base, "answerable": True, "reason": f"{len(items)} items from a region of {rg['size']}",
            "items": items, "context": context, "tokens": tokens, "naive_tokens": naive,
            "tokens_saved_vs_naive": max(naive - tokens, 0)}


def dumps(pack: dict) -> str:
    return json.dumps(pack, default=lambda x: x.item() if hasattr(x, "item") else str(x))


__all__ = ["build_context", "count_tokens", "get_doc", "sentences", "truncate_tokens", "SOURCES"]
