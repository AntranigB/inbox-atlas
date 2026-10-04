"""POST /api/context: the same token-budgeted context pack the MCP server returns."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from atlas.context import pack

router = APIRouter()


class ContextBody(BaseModel):
    question: str
    budget_tokens: int = 800
    sources: list[str] | None = None
    k: int = 8


class GetBody(BaseModel):
    uri: str
    max_tokens: int = 1500


@router.post("/api/context")
def api_context(b: ContextBody):
    p = pack.build_context(b.question, budget_tokens=b.budget_tokens, sources=tuple(b.sources or pack.SOURCES), k=b.k)
    return pack.json.loads(pack.dumps(p))


@router.post("/api/context/get")
def api_context_get(b: GetBody):
    return pack.get_doc(b.uri, max_tokens=b.max_tokens)
