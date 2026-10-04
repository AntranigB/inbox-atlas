"""POST /api/context: the same token-budgeted context pack the MCP server returns.

POST /api/context/folders and GET /api/context/related_folders?path= navigate the vault by folder."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from atlas.context import pack, tree
from atlas.db import qlog

router = APIRouter()


class ContextBody(BaseModel):
    question: str
    budget_tokens: int = 800
    sources: list[str] | None = None
    k: int = 8


class FoldersBody(BaseModel):
    question: str
    k_folders: int = 5
    k_notes: int = 3
    within: str | None = None


class GetBody(BaseModel):
    uri: str
    max_tokens: int = 1500


@router.post("/api/context")
def api_context(b: ContextBody):
    with qlog.timed("agent_context", b.question) as row:
        p = pack.build_context(b.question, budget_tokens=b.budget_tokens, sources=tuple(b.sources or pack.SOURCES), k=b.k)
        row.update(region_size=len(p.get("items") or []), tokens_returned=p.get("tokens"))
    return pack.json.loads(pack.dumps(p))


@router.post("/api/context/get")
def api_context_get(b: GetBody):
    return pack.get_doc(b.uri, max_tokens=b.max_tokens)


@router.post("/api/context/folders")
def api_context_folders(b: FoldersBody):
    with qlog.timed("agent_folders", b.question) as row:
        p = tree.points_of_interest(b.question, k_folders=b.k_folders, k_notes=b.k_notes, within=b.within)
        row.update(region_size=len(p.get("folders") or []), tokens_returned=p.get("tokens"))
    return pack.json.loads(pack.dumps(p))


@router.get("/api/context/related_folders")
def api_related_folders(path: str, k: int = 5):
    return pack.json.loads(pack.dumps(tree.related_folders(path, k=k)))
