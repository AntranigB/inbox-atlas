"""iMessage routes: send a text to the owner, notifier status/brief, and watch management for the sidecar."""

import json
import logging
import os
import sys
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atlas import config, notify

log = logging.getLogger("atlas.messaging")
router = APIRouter()


class NotifyBody(BaseModel):
    text: str
    to: str | None = None


class BriefBody(BaseModel):
    enabled: bool
    time: str | None = None


class WatchBody(BaseModel):
    name: str
    topic: str | None = None
    positive: list[str] | None = None
    negative: list[str] | None = None


@router.post("/api/notify")
def post_notify(body: NotifyBody):
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    try:
        return notify.send_text(body.text, body.to)
    except notify.NotifyError as e:
        raise HTTPException(502, str(e)) from e


@router.get("/api/notify/status")
def notify_status():
    return notify.get_notifier().status()


@router.post("/api/notify/brief")
def set_brief(body: BriefBody):
    return notify.get_notifier().set_brief(body.enabled, body.time)


@router.post("/api/notify/brief/send")
def send_brief_now():
    """Send the morning brief right now (demo button)."""
    n = notify.get_notifier()
    text = n.build_brief()
    try:
        notify.send_text(text)
    except notify.NotifyError as e:
        raise HTTPException(502, str(e)) from e
    return {"ok": True, "text": text}


@router.post("/api/notify/poll")
def poll_now():
    return {"sent": notify.get_notifier().poll()}


@router.get("/api/notify/agenda")
def agenda(date: str | None = None):
    """todays_agenda from the search-agent tools when present, else the local fallback."""
    from datetime import date as Date
    try:
        from atlas.agent.tools import todays_agenda  # search-agent branch
        return todays_agenda(date=date) if date else todays_agenda()
    except ImportError:
        return notify.local_agenda(Date.fromisoformat(date) if date else None)


def _conn():
    from atlas import store
    return store.connect()


@router.get("/api/notify/watches")
def list_watches():
    rows = _conn().execute("select id, name from watches order by created").fetchall()
    return [{"id": r["id"], "name": r["name"]} for r in rows]


@router.post("/api/notify/watches")
def add_watch(body: WatchBody):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    positive = body.positive or [body.topic or name]
    negative = body.negative or []
    try:
        from atlas.agent.tools import add_watch as agent_add  # search-agent branch
    except ImportError:
        agent_add = None
    if agent_add:
        r = agent_add(name=name, positive=positive, negative=negative)
        return {"id": r.get("id"), "name": r.get("name", name)}
    conn = _conn()
    wid = uuid.uuid4().hex[:12]
    conn.execute("insert into watches values (?,?,?,?,?,?)",
                 (wid, name, json.dumps(positive), json.dumps(negative), int(time.time()), int(time.time())))
    conn.commit()
    return {"id": wid, "name": name}


@router.delete("/api/notify/watches/{name}")
def delete_watch(name: str):
    conn = _conn()
    cur = conn.execute("delete from watches where id=? or lower(name)=lower(?)", (name, name))
    conn.commit()
    return {"removed": cur.rowcount}


def _enabled():
    if os.getenv("NOTIFY", "1") == "0":
        return False
    return "pytest" not in sys.modules and "PYTEST_CURRENT_TEST" not in os.environ


def on_startup():
    if not _enabled():
        log.info("notifier disabled")
        return
    notify.get_notifier().start()
    log.info("notifier on, texting via %s", config.env("SIDECAR_URL", config.SIDECAR_URL))
