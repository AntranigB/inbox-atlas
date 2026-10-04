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
    max_tokens = max(0, int(max_tokens))
    if not max_tokens:
        return ""
    e = _enc()
    if not e:
        return text[: max_tokens * 4]
    ids = e.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    # The ellipsis is part of the budget, too. Re-encoding guards against token
    # boundaries changing when the suffix is appended (and split Unicode tokens).
    suffix = " ..." if max_tokens >= 2 else ""
    keep = max_tokens - len(e.encode(suffix))
    while keep >= 0:
        cut = e.decode(ids[:keep]).rstrip("\ufffd") + suffix
        if count_tokens(cut) <= max_tokens:
            return cut
        keep -= 1
    return ""


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

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(*])")


def sentences(text: str) -> list[str]:
    out = []
    # Email commonly wraps prose at 72 columns and puts event title/date/location
    # on successive lines. A single newline is not a sentence boundary. Retain
    # those relationships; blank lines still delimit paragraphs.
    paragraphs = re.split(r"\n\s*\n", text or "")
    spans = [s for p in paragraphs for s in _SPLIT.split(re.sub(r"\s+", " ", p))]
    for s in spans:
        s = re.sub(r"\s+", " ", s).strip(" -*>\t")
        # A short line can be the entire answer: "Oct 6, 8am", "$42", "K7QW2P".
        # Remove separators, not facts. Keep adjacent label/value lines together
        # so the extractor cannot select "Confirmation:" without its code.
        if any(c.isalnum() for c in s):
            if out and (out[-1].endswith(":") or len(s) < 12 or len(out[-1]) < 12):
                out[-1] += " " + s
                continue
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
    return f"No confident match in {' or '.join(sources)}. Relevant information may still exist."


def build_context(question: str, budget_tokens: int = 800, sources=SOURCES, k: int = 8,
                  engine=None, facets: dict | None = None, use_grok: bool = True) -> dict:
    """Minimal context that answers `question`, packed under `budget_tokens`.

    Returns {answerable, confidence, reason, region, items, context, tokens, tokens_saved_vs_naive}.
    tokens counts the rendered `context` string. tokens_saved_vs_naive compares it with reading the
    same top-k documents whole (full email, or the full note for a vault hit).
    """
    from atlas.agent import grok

    budget_tokens = int(budget_tokens)
    if budget_tokens < 0:
        raise ValueError("budget_tokens must be nonnegative")
    if k < 1:
        raise ValueError("k must be positive")
    eng = engine or get_engine()
    sources = tuple(s for s in (sources or SOURCES) if s in SOURCES) or SOURCES
    exp = facets or (grok.expand(question) if use_grok else {"positive": [], "negative": []})
    pos, neg = list(exp.get("positive") or []), list(exp.get("negative") or [])
    filters = {**(exp.get("filters") or {}), "sources": list(sources)}
    res = hybrid.search(question, pos, neg, filters, k=max(k * 3, 12), mode="region", engine=eng)
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
        context = truncate_tokens(reason, budget_tokens)
        return {**base, "answerable": False, "status": "no_confident_match", "reason": reason,
                "items": [], "context": context, "tokens": count_tokens(context), "tokens_saved_vs_naive": 0}

    pairs = [(h, eng.get_email(h["id"])) for h in hits]
    pairs = [(h, r) for h, r in pairs if r]
    hits, rows = [h for h, _ in pairs], [r for _, r in pairs]
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
    items, naive, seen_docs, seen_excerpts = [], 0, set(), set()
    budget = int(budget_tokens)
    for j, (h, r) in enumerate(zip(hits, rows)):
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
        # Identical text under different senders, subjects, or dates can mean
        # different things (e.g. "tomorrow"). Deduplicate only matching metadata.
        fingerprint = (item["source"], item["title"], item["from"], item["date"],
                       " ".join(item["excerpt"].casefold().split()))
        if fingerprint in seen_excerpts:
            continue
        # drop the weakest sentences until the item fits what is left of the budget
        while keep and count_tokens(render(items + [item])) > budget and len(keep) > 1:
            keep.remove(min(keep, key=lambda i: sc[i]))
            item["excerpt"] = _excerpt(ss, keep)
        if count_tokens(render(items + [item])) > budget:
            if items:
                continue  # a later, shorter item may still fit
            room = budget - count_tokens(_header(item) + "\n")
            while room > 0:
                item["excerpt"] = truncate_tokens(item["excerpt"], room)
                if count_tokens(render([item])) <= budget:
                    break
                room -= 1
            if room <= 0 or not item["excerpt"].strip(" ."):
                continue
        items.append(item)
        seen_excerpts.add(fingerprint)
        doc_id = r["thread_id"] if r.get("source") == "obsidian" else r["id"]
        if doc_id not in seen_docs:
            naive += count_tokens(full_note_text(eng.conn, doc_id) if r.get("source") == "obsidian" else doc_text(r))
            seen_docs.add(doc_id)
    context = render(items)
    tokens = count_tokens(context)
    return {**base, "answerable": bool(items), "status": "ok" if items else "insufficient_budget",
            "reason": f"{len(items)} items from a region of {rg['size']}" if items else "Relevant matches did not fit the context budget.",
            "items": items, "context": context, "tokens": tokens, "naive_tokens": naive,
            "tokens_saved_vs_naive": max(naive - tokens, 0)}


def tool_payload(p: dict) -> dict:
    """Compact agent response: citations/excerpts appear once in context, not twice.

    The budget covers context only. response_tokens counts the whole compact JSON
    payload, excluding this counter and transport-specific MCP framing.
    """
    out = {k: p.get(k) for k in ("answerable", "status", "confidence", "reason", "context", "tokens")}
    out["response_tokens"] = count_tokens(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return out


def dumps(pack: dict) -> str:
    return json.dumps(pack, default=lambda x: x.item() if hasattr(x, "item") else str(x))


__all__ = ["build_context", "count_tokens", "get_doc", "sentences", "truncate_tokens", "SOURCES"]
