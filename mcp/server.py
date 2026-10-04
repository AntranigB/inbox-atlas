"""Inbox Atlas as an MCP server: token-budgeted context from your inbox and Obsidian vault.

    uv run python mcp/server.py                         # stdio (Claude Code, Hermes, OpenClaw)
    uv run python mcp/server.py --http --port 8767      # streamable HTTP at http://127.0.0.1:8767/mcp

Tools:
  atlas_context(question, budget_tokens, sources)  minimal packed context, or a calibrated "nothing here"
  atlas_search(query, k, sources)                   ranked hits with one-line snippets
  atlas_related(topic, sources)                     yes/no: is this topic in the data at all?
  atlas_get(uri, max_tokens)                        one email, note section or whole note, capped
  atlas_points_of_interest(question, within)        which vault folders and notes a question lives in
  atlas_related_folders(path, k)                    folders near a folder, with shared tags and links

See docs/mcp.md for registering it with Claude Code, Hermes Agent and OpenClaw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# this directory is named mcp/, so keep it from shadowing the mcp SDK package
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from atlas.context import pack, tree  # noqa: E402

INSTRUCTIONS = (
    "Inbox Atlas searches the user's email and Obsidian notes by meaning. Call atlas_context first: it returns "
    "only the sentences that answer the question, under a token budget. If it returns answerable=false, the "
    "topic is not in the data: stop searching instead of grepping more files. Use atlas_get(uri) only when an "
    "excerpt is not enough. For a large or nested vault, atlas_points_of_interest(question) first shows which "
    "folders and notes the question lives in; pass within=<folder> to drill down, then atlas_get the note.")

server = _Server("inbox-atlas", instructions=INSTRUCTIONS)


def _sources(sources):
    if not sources:
        return pack.SOURCES
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",")]
    return tuple(s for s in sources if s in pack.SOURCES) or pack.SOURCES


@server.tool()
def atlas_context(question: str, budget_tokens: int = 800, sources: list[str] | None = None) -> dict:
    """Minimal context that answers a question from the user's email (gmail) and notes (obsidian).

    Returns answerable, confidence, the packed context string, items with uri/title/date/excerpt, and
    token counts. answerable=false means nothing relevant exists: stop searching."""
    p = pack.build_context(question, budget_tokens=int(budget_tokens), sources=_sources(sources))
    return {k: p.get(k) for k in ("answerable", "confidence", "reason", "context", "items", "tokens",
                                   "tokens_saved_vs_naive")} | {"region": {k: p["region"].get(k) for k in
                                                                          ("facets", "size", "max_z")}}


@server.tool()
def atlas_search(query: str, k: int = 10, sources: list[str] | None = None) -> dict:
    """Ranked hits (uri, title, date, snippet, z) for a topic across email and notes."""
    from atlas.agent import grok
    from atlas.search import hybrid

    exp = grok.expand(query)
    res = hybrid.search(query, exp.get("positive"), exp.get("negative"), {"sources": list(_sources(sources))},
                        k=int(k), mode="region")
    rg = res["region"]
    hits = []
    for h in res["hits"]:
        uri = h["id"][4:] if h["id"].startswith("obs:") else f"gmail:{h['id']}"
        hits.append({"uri": uri, "source": h.get("source"), "title": h.get("subject"), "from": h.get("from"),
                     "date": pack._fmt_date(h.get("date")), "snippet": (h.get("snippet") or "")[:160],
                     "z": h.get("z"), "member": h.get("member")})
    return {"related": rg["related"], "confidence": rg["confidence"], "region_size": rg["size"], "hits": hits}


@server.tool()
def atlas_related(topic: str, sources: list[str] | None = None) -> dict:
    """Calibrated yes/no: does the user's email or vault contain anything about this topic?"""
    from atlas.agent import grok
    from atlas.search import hybrid

    exp = grok.expand(topic)
    res = hybrid.search(topic, exp.get("positive"), exp.get("negative"), {"sources": list(_sources(sources))},
                        k=5, mode="region")
    rg = res["region"]
    return {"related": rg["related"], "confidence": rg["confidence"], "count": rg["count"], "max_z": rg["max_z"],
            "examples": [{"uri": h["id"][4:] if h["id"].startswith("obs:") else f"gmail:{h['id']}",
                          "title": h.get("subject")} for h in res["hits"] if h.get("member")][:3]}


@server.tool()
def atlas_get(uri: str, max_tokens: int = 1500) -> dict:
    """Full text of one item by uri: gmail:<id>, <note path>#<heading anchor>, or <note path> for a whole note."""
    return pack.get_doc(uri, max_tokens=int(max_tokens))


@server.tool()
def atlas_points_of_interest(question: str, k_folders: int = 5, k_notes: int = 3, within: str | None = None) -> dict:
    """Which folders and notes of the Obsidian vault a question lives in, before reading any note.

    Returns related (is it in the vault at all), ranked folders (path, hit notes / notes, score, top tags,
    best note uris, one excerpt), a compact context string and its token count. within=<folder path>
    restricts to that subtree so you can go folder, then subfolder, then note. related=false: stop."""
    p = tree.points_of_interest(question, k_folders=int(k_folders), k_notes=int(k_notes), within=within)
    keep = ("path", "n_notes", "hit_notes", "score", "last_modified", "top_tags", "notes", "title", "excerpt")
    return {k: p.get(k) for k in ("related", "confidence", "within", "reason", "context", "tokens")} | {
        "folders": [{k: f.get(k) for k in keep} for f in p["folders"]]}


@server.tool()
def atlas_related_folders(path: str, k: int = 5) -> dict:
    """Vault folders closest to `path` by meaning (ancestors and descendants excluded), with the cosine and
    the tags and wikilinks they share."""
    return tree.related_folders(path, k=int(k))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    a = ap.parse_args(argv)
    if a.http:
        try:
            server.run("streamable-http", host=a.host, port=a.port)
        except TypeError:  # mcp 1.x takes host/port in settings
            server.settings.host, server.settings.port = a.host, a.port
            server.run("streamable-http")
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
