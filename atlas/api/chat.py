"""Grok chat sessions: POST /api/chat keeps per-session history (atlas/chat.py)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atlas import chat

router = APIRouter()


class ChatBody(BaseModel):
    text: str
    session_id: str | None = None
    channel: str = "web"


@router.post("/api/chat")
def post_chat(body: ChatBody):
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    return chat.chat(body.text, body.session_id, body.channel)


@router.get("/api/chat/sessions")
def sessions(limit: int = 50):
    return chat.get_store().list_sessions(limit=limit)


@router.get("/api/chat/sessions/{session_id}")
def session(session_id: str, limit: int = 200):
    return {"id": session_id, "messages": chat.get_store().get_history(session_id, limit=limit)}


@router.delete("/api/chat/sessions/{session_id}")
def delete_session(session_id: str):
    return {"removed": chat.get_store().delete_session(session_id)}
