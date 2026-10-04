"""Grok chat sessions with per-session history.

Storage: the tiger branch's `atlas.db.chat` when it imports (Postgres with ATLAS_DB=pg, its own
SQLite tables otherwise), else a small SQLite file (data/chat.sqlite). ATLAS_DB=local forces the
latter. Both expose the same functions below.

Session ids are free strings. The web UI makes random ones, iMessage uses
`imessage:<handle>` (and `telegram:<id>`, `whatsapp_business:<id>` on other providers).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid

from atlas import config

log = logging.getLogger("atlas.chat")

HISTORY_TURNS = 12  # messages sent to Grok as context

SCHEMA = """
create table if not exists chat_sessions(
  id text primary key, title text, channel text, created integer, updated integer
);
create table if not exists chat_messages(
  id integer primary key autoincrement, session_id text, role text, content text, ts integer
);
create index if not exists chat_messages_session on chat_messages(session_id, id);
"""


class SqliteChat:
    kind = "sqlite"

    def __init__(self, path=None):
        self.path = path or (config.DATA / "chat.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def ensure_session(self, session_id, channel="web", title=None):
        now = int(time.time())
        with self.lock:
            self.conn.execute("insert or ignore into chat_sessions values (?,?,?,?,?)",
                              (session_id, (title or "")[:80], channel, now, now))
            if title:
                self.conn.execute("update chat_sessions set title=? where id=? and (title is null or title='')",
                                  (title[:80], session_id))
            self.conn.commit()

    def add_message(self, session_id, role, content, channel=None):
        now = int(time.time())
        with self.lock:
            self.conn.execute("insert into chat_messages(session_id, role, content, ts) values (?,?,?,?)",
                              (session_id, role, content, now))
            self.conn.execute("update chat_sessions set updated=? where id=?", (now, session_id))
            self.conn.commit()

    def get_history(self, session_id, limit=HISTORY_TURNS):
        with self.lock:
            rows = self.conn.execute(
                "select role, content, ts from chat_messages where session_id=? order by id desc limit ?",
                (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def list_sessions(self, limit=50):
        with self.lock:
            rows = self.conn.execute(
                "select s.*, (select count(*) from chat_messages m where m.session_id=s.id) n "
                "from chat_sessions s order by updated desc limit ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id):
        with self.lock:
            self.conn.execute("delete from chat_messages where session_id=?", (session_id,))
            cur = self.conn.execute("delete from chat_sessions where id=?", (session_id,))
            self.conn.commit()
        return cur.rowcount


class PgChat:
    """Adapter over the tiger branch's atlas.db chat functions (duck typed, names may vary)."""

    kind = "pg"

    def __init__(self, mod):
        self.m = mod

    def _fn(self, *names):
        for n in names:
            f = getattr(self.m, n, None)
            if callable(f):
                return f
        raise AttributeError(f"atlas.db has none of {names}")

    def ensure_session(self, session_id, channel="web", title=None):
        f = None
        for n in ("ensure_session", "create_session", "upsert_session"):
            f = getattr(self.m, n, None)
            if callable(f):
                break
        if f:
            try:
                f(session_id, channel=channel, title=title)
            except TypeError:
                f(session_id)

    def add_message(self, session_id, role, content, channel=None):
        f = self._fn("add_message", "append_message", "save_message")
        try:
            f(session_id, role, content, channel=channel)
        except TypeError:
            f(session_id, role, content)

    def get_history(self, session_id, limit=HISTORY_TURNS):
        rows = self._fn("get_history", "history", "get_messages")(session_id, limit=limit)
        return [dict(r) if not isinstance(r, dict) else r for r in rows]

    def list_sessions(self, limit=50):
        return [dict(r) if not isinstance(r, dict) else r for r in self._fn("list_sessions", "sessions")(limit=limit)]

    def delete_session(self, session_id):
        try:
            return self._fn("delete_session", "clear_session")(session_id) or 0
        except AttributeError:
            return 0


_store = None
_store_lock = threading.Lock()


def get_store():
    global _store
    with _store_lock:
        if _store is None:
            _store = _open()
        return _store


def _open():
    """tiger's atlas.db.chat when present (it picks Postgres or SQLite itself from ATLAS_DB /
    DATABASE_URL), so web, iMessage and grok.ask share one history. Else data/chat.sqlite."""
    if (config.env("ATLAS_DB") or "").lower() != "local":
        try:
            import importlib
            mod = importlib.import_module("atlas.db.chat")
            s = PgChat(mod)
            s.list_sessions(limit=1)
            log.info("chat history via atlas.db.chat")
            return s
        except ModuleNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001 - fall back to sqlite, never block chat
            log.warning("atlas.db chat history unavailable (%s), using data/chat.sqlite", e)
    return SqliteChat()


def reset_store(store=None):
    global _store
    _store = store


def new_session_id() -> str:
    return "web:" + uuid.uuid4().hex[:12]


def chat(text: str, session_id: str | None = None, channel: str = "web", ask=None, store=None) -> dict:
    """One turn: load history, ask Grok with it, save both messages."""
    store = store or get_store()
    session_id = (session_id or "").strip() or new_session_id()
    if ask is None:
        from atlas.agent.grok import ask
    store.ensure_session(session_id, channel, title=text.strip().split("\n")[0])
    history = [{"role": h["role"], "content": h["content"]} for h in store.get_history(session_id)]
    # history is passed explicitly; a grok.ask that persists turns itself (tiger branch) is told
    # not to, so nothing is double written.
    kw = {}
    try:
        import inspect
        if "persist" in inspect.signature(ask).parameters:
            kw["persist"] = False
    except (TypeError, ValueError):
        pass
    from atlas.db import qlog

    with qlog.timed(f"chat_{channel}", text) as row:
        res = ask(text, channel=channel, history=history, **kw)
        if not isinstance(res, dict):
            res = {"reply": str(res)}
        row.update(region_size=(res.get("region") or {}).get("size"), tokens_returned=len(res.get("reply") or "") // 4)
    store.add_message(session_id, "user", text, channel)
    store.add_message(session_id, "assistant", res.get("reply") or "", channel)
    return {**res, "session_id": session_id, "history_used": len(history)}
