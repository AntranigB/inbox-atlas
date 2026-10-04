"""iMessage notifications: morning brief, new mail in watched regions, and send_text.

Runs as a background thread inside the API server (started from atlas/api/messaging.py).
Texts go out through the Node sidecar (imessage/, POST SIDECAR_URL/send).

Env:
  NOTIFY=0             disable the scheduler
  BRIEF_TIME=08:00     morning brief, local TIMEZONE
  BRIEF=0              brief off by default (can be toggled by texting "brief on/off")
  POLL_SECONDS=120     new mail poll interval
  QUIET_HOURS=22-7     watch alerts are held during these local hours and sent after
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from atlas import config

log = logging.getLogger("atlas.notify")

MAX_SEEN = 5000
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_WORDS = re.compile(r"\b(today|tonight|this (morning|afternoon|evening)|my day|schedule|agenda|calendar|tomorrow)\b", re.I)


class NotifyError(RuntimeError):
    pass


def tz():
    return ZoneInfo(config.env("TIMEZONE", config.TIMEZONE) or "America/New_York")


def send_text(text: str, to: str | None = None, timeout: float = 30.0) -> dict:
    """Text the owner through the sidecar. Raises NotifyError with a readable message."""
    url = config.env("SIDECAR_URL", config.SIDECAR_URL).rstrip("/") + "/send"
    body = {"text": text}
    if to:
        body["to"] = to
    try:
        r = httpx.post(url, json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise NotifyError(f"iMessage sidecar not reachable at {url} ({e}). Start it: cd imessage && npm start") from e
    try:
        data = r.json()
    except ValueError:
        data = {"error": r.text[:200]}
    if r.status_code >= 400 or not data.get("ok", False):
        raise NotifyError(data.get("error") or f"sidecar returned {r.status_code}")
    return data


def agenda_prompt(text: str = "What do I have today?", now: datetime | None = None) -> str:
    """Same hint the sidecar adds, so the agent calls todays_agenda for the local date."""
    if not DAY_WORDS.search(text):
        return text
    now = now or datetime.now(tz())
    tomorrow = bool(re.search(r"\btomorrow\b", text, re.I))
    d = (now + timedelta(days=1 if tomorrow else 0)).date().isoformat()
    which = "tomorrow is" if tomorrow else "today is"
    return (f"{text}\n[context: the user's local date {which} {d} ({now.tzinfo}). "
            f"Use the todays_agenda tool with date={d} and answer in short plain text.]")


def _mentions_day(row: dict, day: date, zone) -> bool:
    """Does this email talk about `day`? Explicit dates, or a weekday name within the week after it was sent."""
    text = f"{row.get('subject') or ''} {row.get('body') or ''}".lower()
    sent = datetime.fromtimestamp(row.get("date") or 0, zone).date()
    if sent > day or (day - sent).days > 30:
        return False
    mon_full, mon_abbr = day.strftime("%B").lower(), day.strftime("%b").lower()
    pats = [rf"\b({mon_full}|{mon_abbr})\.? {day.day}(st|nd|rd|th)?\b", rf"\b{day.month}/{day.day}\b"]
    if any(re.search(p, text) for p in pats):
        return True
    delta = (day - sent).days
    if delta == 0 and re.search(r"\b(today|tonight)\b", text):
        return True
    if delta == 1 and re.search(r"\btomorrow\b", text):
        return True
    return 0 < delta <= 6 and re.search(rf"\b{WEEKDAYS[day.weekday()]}\b", text) is not None


def local_agenda(day: date | None = None, conn=None) -> dict:
    """Fallback for todays_agenda when the search-agent branch is not present."""
    from atlas import store

    zone = tz()
    day = day or datetime.now(zone).date()
    conn = conn or store.connect()
    t0 = int(datetime.combine(day, datetime.min.time(), zone).timestamp())
    events = store.events_between(conn, t0, t0 + 86400)
    lo = t0 - 30 * 86400
    rows = [dict(r) for r in conn.execute("select * from emails where date >= ? and date < ? order by date", (lo, t0 + 86400))]
    emails = [{"id": r["id"], "from": r["from_name"] or r["from_addr"], "subject": r["subject"], "snippet": r["snippet"]}
              for r in rows if _mentions_day(r, day, zone)]
    return {"date": day.isoformat(), "events": events, "emails": emails}


def format_agenda(a: dict) -> str:
    zone = tz()
    lines = []
    for e in a.get("events") or []:
        t = datetime.fromtimestamp(e["start"], zone).strftime("%-I:%M%p").lower() if isinstance(e.get("start"), int) else ""
        lines.append(f"- {t} {e.get('title') or 'event'}{' @ ' + e['location'] if e.get('location') else ''}".replace("-  ", "- "))
    for m in a.get("emails") or []:
        lines.append(f"- {m.get('subject')} ({m.get('from')}): {m.get('snippet') or ''}"[:220])
    return "\n".join(lines) if lines else "Nothing on the calendar and no emails mention today."


def _agent_ask(text: str, history=None) -> str:
    from atlas.agent.grok import ask  # search-agent branch

    r = ask(text, channel="imessage", history=history or [])
    return r.get("reply", "") if isinstance(r, dict) else str(r)


def _hit_fields(hit) -> tuple[str, str, str, str]:
    """(watch name, email id, sender, subject) from whatever shape check_watches returns."""
    if isinstance(hit, (list, tuple)) and len(hit) == 2:
        w, e = hit
        hit = {"watch": w, "email": e}
    w = hit.get("watch")
    wname = hit.get("watch_name") or (w.get("name") if isinstance(w, dict) else w) or hit.get("name") or "a watch"
    e = hit.get("email") if isinstance(hit.get("email"), dict) else hit
    eid = e.get("email_id") or e.get("id") or hit.get("email_id") or ""
    sender = e.get("from_name") or e.get("from") or e.get("from_addr") or "someone"
    subject = e.get("subject") or "(no subject)"
    return str(wname), str(eid), str(sender), str(subject)


class Notifier:
    def __init__(self, send=send_text, ask=None, sync=None, check=None, now=None, state_path=None, conn=None):
        self.send = send
        self._ask = ask
        self._sync = sync
        self._check = check
        self._now = now or (lambda: datetime.now(tz()))
        self.state_path = state_path or (config.DATA / "notify_state.json")
        self.conn = conn
        self.lock = threading.RLock()
        self.thread = None
        self.stop_event = threading.Event()
        self.poll_seconds = int(config.env("POLL_SECONDS", "120") or 120)
        self.brief_time = config.env("BRIEF_TIME", "08:00") or "08:00"
        self.quiet = self._parse_quiet(config.env("QUIET_HOURS", "22-7"))
        self.state = {"brief_enabled": config.env("BRIEF", "1") != "0", "last_brief": None, "seen": [],
                      "pending": [], "last_poll": 0, "last_error": None, "sent": 0}
        self._load()

    # state
    def _load(self):
        try:
            self.state.update(json.loads(self.state_path.read_text()))
        except (OSError, ValueError):
            pass

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state["seen"] = self.state["seen"][-MAX_SEEN:]
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.state_path)
        except OSError as e:
            log.warning("could not save notify state: %s", e)

    @staticmethod
    def _parse_quiet(spec):
        m = re.fullmatch(r"\s*(\d{1,2})\s*-\s*(\d{1,2})\s*", spec or "")
        return (int(m.group(1)), int(m.group(2))) if m else None

    def is_quiet(self, now=None):
        if not self.quiet:
            return False
        h = (now or self._now()).hour
        a, b = self.quiet
        return a <= h < b if a < b else (h >= a or h < b)

    def set_brief(self, enabled: bool, at: str | None = None):
        with self.lock:
            self.state["brief_enabled"] = bool(enabled)
            if at and re.fullmatch(r"\d{1,2}:\d{2}", at):
                self.brief_time = at
            self._save()
        return {"enabled": self.state["brief_enabled"], "time": self.brief_time}

    def status(self):
        return {"running": bool(self.thread and self.thread.is_alive()), "brief_enabled": self.state["brief_enabled"],
                "brief_time": self.brief_time, "timezone": str(tz()), "last_brief": self.state["last_brief"],
                "poll_seconds": self.poll_seconds, "last_poll": self.state["last_poll"], "quiet_hours": self.quiet,
                "pending": len(self.state["pending"]), "sent": self.state["sent"], "last_error": self.state["last_error"],
                "sidecar": config.env("SIDECAR_URL", config.SIDECAR_URL)}

    def _text(self, text):
        try:
            self.send(text)
            self.state["sent"] += 1
            return True
        except Exception as e:  # noqa: BLE001 - a failed text must never kill the loop
            self.state["last_error"] = f"{type(e).__name__}: {e}"
            log.warning("send_text failed: %s", e)
            return False

    # morning brief
    def build_brief(self, now=None) -> str:
        now = now or self._now()
        prompt = agenda_prompt("What do I have today?", now)
        try:
            reply = (self._ask or _agent_ask)(prompt)
        except Exception as e:  # noqa: BLE001 - agent missing or down, use the local agenda
            log.info("agent unavailable for brief (%s), using local agenda", e)
            reply = None
        if not reply:
            try:
                from atlas.agent.tools import todays_agenda  # search-agent branch
                a = todays_agenda(date=now.date().isoformat())
            except Exception:  # noqa: BLE001
                a = local_agenda(now.date(), self.conn)
            reply = format_agenda(a)
        return f"Good morning. {now.strftime('%A %b')} {now.day}:\n{reply}"

    def brief_due(self, now):
        if not self.state["brief_enabled"]:
            return False
        today = now.date().isoformat()
        if self.state["last_brief"] == today:
            return False
        h, m = (int(x) for x in self.brief_time.split(":"))
        start = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return start <= now < start + timedelta(hours=3)

    def send_brief(self, now=None):
        now = now or self._now()
        with self.lock:
            ok = self._text(self.build_brief(now))
            if ok:
                self.state["last_brief"] = now.date().isoformat()
            self._save()
            return ok

    # new mail
    def _sync_fn(self):
        if self._sync:
            return self._sync
        if not (config.env("GMAIL_USER") and config.env("GMAIL_APP_PASSWORD")):
            return None
        try:
            from atlas.ingest.sync import sync_new  # ingest branch
        except ImportError:
            return None
        return sync_new

    def _check_fn(self):
        if self._check:
            return self._check
        try:
            from atlas.agent.grok import check_watches  # search-agent branch
        except ImportError:
            return None
        return check_watches

    def poll(self, now=None):
        now = now or self._now()
        with self.lock:
            self.state["last_poll"] = int(time.time())
            sync, check = self._sync_fn(), self._check_fn()
            if not sync:
                self._save()
                return []
            try:
                new = sync() or []
                if isinstance(new, dict):
                    new = new.get("new_ids") or new.get("ids") or []
                hits = check(list(new)) if (check and new) else []
            except Exception as e:  # noqa: BLE001
                self.state["last_error"] = f"poll: {type(e).__name__}: {e}"
                log.warning("mail poll failed: %s", e)
                self._save()
                return []
            msgs = self.queue_hits(hits or [])
            self.flush(now)
            return msgs

    def queue_hits(self, hits):
        seen = set(self.state["seen"])
        msgs = []
        for hit in hits:
            wname, eid, sender, subject = _hit_fields(hit)
            key = f"{wname}|{eid or subject}"
            if key in seen:
                continue
            seen.add(key)
            self.state["seen"].append(key)
            msgs.append(f"New email in {wname}: {sender}: {subject}")
        self.state["pending"].extend(msgs)
        return msgs

    def flush(self, now=None):
        now = now or self._now()
        with self.lock:
            pending = self.state["pending"]
            if not pending or self.is_quiet(now):
                self._save()
                return 0
            text = pending[0] if len(pending) == 1 else f"{len(pending)} new emails in your watches:\n" + "\n".join(pending)
            if self._text(text):
                self.state["pending"] = []
            self._save()
            return len(pending)

    # loop
    def tick(self, now=None):
        now = now or self._now()
        if self.brief_due(now):
            self.send_brief(now)
        if time.time() - (self.state["last_poll"] or 0) >= self.poll_seconds:
            self.poll(now)
        else:
            self.flush(now)

    def _run(self):
        log.info("notifier started (brief %s, poll every %ss)", self.brief_time, self.poll_seconds)
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("notifier tick failed")
            self.stop_event.wait(20)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="atlas-notify", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()


_notifier: Notifier | None = None


def get_notifier() -> Notifier:
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier
