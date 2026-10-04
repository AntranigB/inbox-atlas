"""Grok client: query expansion and the tool-calling inbox agent."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
from zoneinfo import ZoneInfo

import httpx

from atlas import config
from atlas.agent import prompts, tools

log = logging.getLogger("atlas.agent")

MAX_TOOL_CALLS = 6
MAX_SEARCHES = 3  # first search + 2 refinement rounds
CHAT_URL = f"{config.XAI_BASE}/chat/completions"


class GrokError(RuntimeError):
    pass


def _key():
    return config.env("XAI_API_KEY") or config.XAI_API_KEY


def chat(messages, tools_=None, json_mode=False, model=None, temperature=0.2, timeout=60):
    if not _key():
        raise GrokError("XAI_API_KEY not set")
    body = {"model": model or config.GROK_MODEL, "messages": messages, "temperature": temperature}
    if tools_:
        body["tools"] = tools_
        body["tool_choice"] = "auto"
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(CHAT_URL, json=body, timeout=timeout, headers={"Authorization": f"Bearer {_key()}"})
    if r.status_code >= 400:
        raise GrokError(f"grok {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]


def _today():
    return dt.datetime.now(ZoneInfo(config.TIMEZONE)).date().isoformat()


def _parse_json(s):
    s = (s or "").strip()
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0) if m else s)


# ---------- expansion ----------

_exp_cache: dict = {}
_exp_lock = threading.Lock()


def _cache_file():
    return config.DATA / "cache" / "expand.json"


def _load_cache():
    if not _exp_cache and _cache_file().exists():
        try:
            _exp_cache.update(json.loads(_cache_file().read_text()))
        except Exception:
            pass


def expand(query: str) -> dict:
    """{positive, negative, filters, intent}. Falls back to the bare query when Grok is unavailable."""
    q = (query or "").strip()
    fallback = {"positive": [], "negative": [], "filters": {}, "intent": "find", "source": "fallback"}
    if not q:
        return fallback
    with _exp_lock:
        _load_cache()
        if q.lower() in _exp_cache:
            return _exp_cache[q.lower()]
    try:
        msg = chat([{"role": "system", "content": prompts.EXPAND.replace("{today}", _today())},
                    {"role": "user", "content": q}], json_mode=True, temperature=0)
        d = _parse_json(msg.get("content"))
    except Exception as e:
        log.warning("expand failed: %s", e)
        return fallback
    out = {"positive": [str(p) for p in d.get("positive", []) if p][:8],
           "negative": [str(n) for n in d.get("negative", []) if n][:4],
           "filters": {k: v for k, v in (d.get("filters") or {}).items() if v},
           "intent": d.get("intent") or "find", "source": "grok"}
    with _exp_lock:
        _exp_cache[q.lower()] = out
        try:
            _cache_file().parent.mkdir(parents=True, exist_ok=True)
            _cache_file().write_text(json.dumps(_exp_cache, indent=1))
        except Exception:
            pass
    return out


# ---------- agent ----------

def _trim(text, channel):
    text = (text or "").strip()
    if channel in ("imessage", "voice"):
        text = re.sub(r"[*_`#]+", "", text)
        text = re.sub(r"^\s*[-]\s+", "", text, flags=re.M)
    text = text.replace(chr(0x2014), ", ").replace(chr(0x2013), " to ")
    if channel == "imessage" and len(text) > 600:
        text = text[:597].rsplit(" ", 1)[0] + "..."
    return text


def _fallback_answer(text, channel):
    """No Grok: run a plain region search and summarize it."""
    exp = expand(text)
    res = tools.run_tool("search_region", {"positive": exp["positive"] or [text], "query": text,
                                           "negative": exp["negative"], "k": 5}, raw=True)
    full = res.get("_full", {})
    hits = full.get("hits", [])
    mem = [h for h in hits if h.get("member")]
    if not mem:
        reply = f"Nothing in your inbox about {text}."
    else:
        parts = [f"{h['from']}: {h['subject']}" for h in mem[:3]]
        reply = f"Found {len(mem)} emails about {text}. " + "; ".join(parts)
    return {"reply": _trim(reply, channel), "hits": hits, "region": full.get("region")}


def ask(text: str, channel: str = "web", history=None) -> dict:
    """Answer a question about the inbox with Grok + tools. Returns {reply, hits, region}."""
    channel = channel if channel in prompts.STYLE else "web"
    if not _key():
        return _fallback_answer(text, channel)
    sys = prompts.SYSTEM.format(today=dt.datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%A %Y-%m-%d"),
                                tz=config.TIMEZONE, style=prompts.STYLE[channel])
    msgs = [{"role": "system", "content": sys}]
    for h in (history or [])[-12:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": str(h["content"])})
    msgs.append({"role": "user", "content": text})

    last_full, calls, searches, trace = None, 0, 0, []
    try:
        while True:
            allow = calls < MAX_TOOL_CALLS
            msg = chat(msgs, tools.TOOL_SCHEMAS if allow else None)
            tcs = msg.get("tool_calls") or []
            if not tcs or not allow:
                reply = msg.get("content") or ""
                break
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls += 1
                if name == "search_region":
                    searches += 1
                    if searches > MAX_SEARCHES:
                        out = {"error": "refinement limit reached, answer with what you have"}
                    else:
                        out = tools.run_tool(name, args, raw=True)
                        if "_full" in out:
                            last_full = out.pop("_full")
                elif calls > MAX_TOOL_CALLS:
                    out = {"error": "tool call limit reached"}
                else:
                    out = tools.run_tool(name, args)
                trace.append({"tool": name, "args": args})
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(tools._clean(out))[:6000]})
    except GrokError as e:
        log.warning("ask failed: %s", e)
        return _fallback_answer(text, channel)
    return {"reply": _trim(reply, channel), "hits": (last_full or {}).get("hits", []),
            "region": (last_full or {}).get("region"), "facets": (last_full or {}).get("facets"), "trace": trace}


def check_watches(new_ids):
    return tools.check_watches(new_ids)
