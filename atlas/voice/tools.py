"""Tool bridge for the voice agent.

Uses atlas.agent.tools (TOOL_SCHEMAS, run_tool) when the search-agent branch is present and
falls back to a small keyword-search stub so the voice agent works on its own.
"""

import asyncio
import inspect
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger("atlas.voice")


def to_realtime(schema: dict) -> dict:
    """Chat-completions tool schema -> flat Realtime function schema."""
    if schema.get("type") == "function" and "function" in schema:
        f = schema["function"]
        return {"type": "function", "name": f["name"], "description": f.get("description", ""),
                "parameters": f.get("parameters", {"type": "object", "properties": {}})}
    if "name" in schema:
        return {"type": "function", **{k: v for k, v in schema.items() if k != "type"}}
    return schema


STUB_SCHEMAS = [
    {"type": "function", "name": "search_region",
     "description": "Search the user's email by topic. Returns matching emails with sender, date, subject, snippet.",
     "parameters": {"type": "object", "properties": {
         "positive": {"type": "array", "items": {"type": "string"},
                      "description": "phrases describing what to find"},
         "negative": {"type": "array", "items": {"type": "string"},
                      "description": "phrases describing what to exclude"},
         "k": {"type": "integer"}},
         "required": ["positive"]}},
    {"type": "function", "name": "get_email",
     "description": "Fetch one email by id with its body.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]


def _fmt_date(ts):
    try:
        return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def stub_run_tool(name: str, args: dict):
    from atlas import store
    conn = store.connect()
    if name == "search_region":
        q = " ".join(args.get("positive") or [])
        neg = {w.lower() for p in (args.get("negative") or []) for w in p.split()}
        hits = []
        for id_, score in store.fts_search(conn, q, int(args.get("k") or 10) * 2):
            e = store.get_email(conn, id_)
            text = f"{e['subject']} {e['body']}".lower()
            if neg and any(w in text for w in neg):
                continue
            hits.append({"id": e["id"], "from": e["from_name"] or e["from_addr"], "date": _fmt_date(e["date"]),
                         "subject": e["subject"], "snippet": e["snippet"], "z": round(score, 3), "prob": None})
        hits = hits[: int(args.get("k") or 10)]
        return {"region": {"size": len(hits), "facet_hits": {}, "nearest_clusters": []}, "hits": hits}
    if name == "get_email":
        e = store.get_email(conn, args.get("id", ""))
        if not e:
            return {"error": "not found"}
        return {"id": e["id"], "from": e["from_name"] or e["from_addr"], "date": _fmt_date(e["date"]),
                "subject": e["subject"], "body": (e["body"] or "")[:2000]}
    return {"error": f"unknown tool {name}"}


def load_tools():
    """Returns (realtime_schemas, run_tool). Imported lazily so a missing agent module is fine."""
    try:
        from atlas.agent import tools as agent_tools
        schemas = [to_realtime(s) for s in agent_tools.TOOL_SCHEMAS]
        return schemas, agent_tools.run_tool
    except (ImportError, AttributeError) as e:
        log.info("atlas.agent.tools unavailable (%s), using voice stub tools", e)
        return STUB_SCHEMAS, stub_run_tool


async def call_tool(run_tool, name: str, arguments: str | dict) -> dict:
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError:
        args = {}
    try:
        if inspect.iscoroutinefunction(run_tool):
            out = await run_tool(name, args)
        else:
            out = await asyncio.to_thread(run_tool, name, args)
            if inspect.isawaitable(out):
                out = await out
    except Exception as e:  # tool errors go back to the model, not up the socket
        log.exception("tool %s failed", name)
        out = {"error": str(e)[:300]}
    return out if isinstance(out, dict) else {"result": out}


def compact_output(result: dict, limit: int = 6000) -> str:
    """JSON for the model, trimmed so a big hit list does not blow up the turn."""
    r = dict(result)
    if isinstance(r.get("hits"), list):
        r["hits"] = [{k: v for k, v in h.items() if k in ("id", "from", "date", "subject", "snippet", "prob")}
                     for h in r["hits"][:8]]
    s = json.dumps(r, default=str)
    return s if len(s) <= limit else s[:limit]
