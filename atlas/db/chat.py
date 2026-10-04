"""Chat history helpers over the active backend (Postgres when ATLAS_DB=pg, else SQLite).

    from atlas.db import chat
    chat.add_message(session_id, "user", "any hackathons?", channel="web")
    chat.get_history(session_id, limit=20)   # [{role, content, ts}]
    chat.list_sessions()                      # [{id, title, channel, updated}]
"""

from __future__ import annotations

from atlas.db.backend import get_backend

_db = None


def _backend():
    global _db
    if _db is None:
        _db = get_backend()
    return _db


def new_session(channel=None, user_handle=None):
    return _backend().new_session(channel, user_handle)


def add_message(session_id, role, content, channel=None, tool_name=None, tool_args=None, hits=None, user_handle=None):
    return _backend().append_message(session_id, role, content, tool_name=tool_name, tool_args=tool_args, hits=hits,
                                     channel=channel, user_handle=user_handle)


def get_history(session_id, limit=20):
    return [{"role": m["role"], "content": m["content"], "ts": m["ts"]} for m in _backend().history(session_id, limit)]


def list_sessions(limit=50):
    return [{"id": s["id"], "title": s["title"], "channel": s["channel"], "updated": s["updated"]}
            for s in _backend().list_sessions(limit)]
