"""Google Calendar (secret iCal URL) sync plus "emails that mention a date".

    uv run python -m atlas.calendar            # sync next 14 days into the events table
"""

import datetime as dt
import hashlib
import re
import sys
from zoneinfo import ZoneInfo

from atlas import config, store

DAYS_AHEAD = 14


def tz():
    return ZoneInfo(config.TIMEZONE)


def _to_dt(v, zone):
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=zone)
    return dt.datetime.combine(v, dt.time(0), tzinfo=zone)  # all day event


def parse_ics(text, start=None, days=DAYS_AHEAD, source="gcal"):
    """Expand recurring events in [start, start+days) and return events rows."""
    import icalendar
    import recurring_ical_events

    zone = tz()
    start = start or dt.datetime.now(zone)
    if start.tzinfo is None:
        start = start.replace(tzinfo=zone)
    end = start + dt.timedelta(days=days)
    cal = icalendar.Calendar.from_ical(text)
    rows = []
    for ev in recurring_ical_events.of(cal).between(start, end):
        s = _to_dt(ev.get("DTSTART").dt, zone)
        e_raw = ev.get("DTEND")
        e = _to_dt(e_raw.dt, zone) if e_raw else s + dt.timedelta(hours=1)
        uid = str(ev.get("UID", ""))
        title = str(ev.get("SUMMARY", "")).strip() or "(no title)"
        rid = hashlib.sha1(f"{uid}|{int(s.timestamp())}".encode()).hexdigest()[:16]
        rows.append({"id": rid, "title": title, "start": int(s.timestamp()), "end": int(e.timestamp()),
                     "location": str(ev.get("LOCATION", "") or "") or None, "source": source})
    rows.sort(key=lambda r: r["start"])
    return rows


def fetch_ics(url=None):
    import httpx

    url = url or config.env("GCAL_ICS_URL")
    if not url:
        raise SystemExit("GCAL_ICS_URL is not set. Google Calendar > Settings > your calendar > "
                         "'Secret address in iCal format', paste it into .env as GCAL_ICS_URL=...")
    url = url.replace("webcal://", "https://")
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r.text


def sync_calendar(conn=None, url=None, days=DAYS_AHEAD):
    conn = conn or store.connect()
    rows = parse_ics(fetch_ics(url), days=days)
    store.upsert_events(conn, rows)
    return rows


# ---------- emails that mention a date ----------

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_RE_MON_DAY = re.compile(_MON + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(\d{4}))?", re.I)
_RE_DAY_MON = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MON + r"\b(?:,?\s*(\d{4}))?", re.I)
_RE_NUM = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/])")
_RE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_WEEKDAY = re.compile(r"\b(?:(this|next)\s+)?(" + "|".join(WEEKDAYS) + r"|mon|tues?|wed|thurs?|fri|sat|sun)\b", re.I)
_RE_REL = re.compile(r"\b(today|tonight|tomorrow|this morning|this afternoon|this evening)\b", re.I)
_WD_ABBR = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6}


def _mon_idx(s):
    return MONTHS.index(s[:3].lower()) + 1


def _resolve_year(month, day, ref):
    """Year for a month/day with no year: the occurrence closest after ref (within ~10 months)."""
    for y in (ref.year, ref.year + 1, ref.year - 1):
        try:
            d = dt.date(y, month, day)
        except ValueError:
            continue
        if -60 <= (d - ref).days <= 300:
            return d
    return None


def mentioned_dates(text, sent):
    """All calendar dates a text refers to, relative to the sent date (a datetime.date)."""
    out = set()

    def add(y, m, d):
        try:
            if y:
                y = int(y)
                y = y + 2000 if y < 100 else y
                out.add(dt.date(y, int(m), int(d)))
            else:
                r = _resolve_year(int(m), int(d), sent)
                if r:
                    out.add(r)
        except ValueError:
            pass

    for m in _RE_MON_DAY.finditer(text):
        add(m.group(3), _mon_idx(m.group(1)), m.group(2))
    for m in _RE_DAY_MON.finditer(text):
        add(m.group(3), _mon_idx(m.group(2)), m.group(1))
    for m in _RE_NUM.finditer(text):
        mo, d = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            add(m.group(3), mo, d)
    for m in _RE_ISO.finditer(text):
        add(m.group(1), m.group(2), m.group(3))
    for m in _RE_REL.finditer(text):
        w = m.group(1).lower()
        out.add(sent + dt.timedelta(days=1 if w == "tomorrow" else 0))
    for m in _RE_WEEKDAY.finditer(text):
        w = m.group(2).lower()
        wd = WEEKDAYS.index(w) if w in WEEKDAYS else _WD_ABBR.get(w)
        if wd is None:
            continue
        if len(w) <= 4 and not m.group(2)[0].isupper():
            continue  # bare lowercase "sun", "wed", "sat" are usually words, not days
        delta = (wd - sent.weekday()) % 7
        if m.group(1) and m.group(1).lower() == "next" and delta == 0:
            delta = 7
        out.add(sent + dt.timedelta(days=delta))
    return out


def extract_event_mentions(conn, date=None, lookback_days=30, limit=20):
    """Emails (sent within lookback_days before date) that mention `date`.

    `date` is a datetime.date or 'YYYY-MM-DD' (default today in TIMEZONE). Returns email dicts
    with an extra `mention` field (the matched snippet), newest first.
    """
    zone = tz()
    if date is None:
        date = dt.datetime.now(zone).date()
    elif isinstance(date, str):
        date = dt.date.fromisoformat(date)
    t1 = int(dt.datetime.combine(date + dt.timedelta(days=1), dt.time(0), tzinfo=zone).timestamp())
    t0 = t1 - (lookback_days + 1) * 86400
    rows = conn.execute("select * from emails where date >= ? and date < ? order by date desc", (t0, t1)).fetchall()
    hits = []
    for r in rows:
        r = dict(r)
        sent = dt.datetime.fromtimestamp(r["date"], zone).date()
        text = f"{r.get('subject') or ''}\n{r.get('body') or ''}"
        if date in mentioned_dates(text, sent):
            r["mention"] = _mention_snippet(text, date, sent)
            hits.append(r)
            if len(hits) >= limit:
                break
    return hits


def _mention_snippet(text, date, sent):
    for line in re.split(r"(?<=[.!?\n])\s+", text):
        if date in mentioned_dates(line, sent):
            return line.strip()[:200]
    return text[:200]


def agenda(conn, date=None):
    """Calendar events plus mentioning emails for one local day (backs todays_agenda)."""
    zone = tz()
    if date is None:
        date = dt.datetime.now(zone).date()
    elif isinstance(date, str):
        date = dt.date.fromisoformat(date)
    t0 = int(dt.datetime.combine(date, dt.time(0), tzinfo=zone).timestamp())
    return {"date": date.isoformat(), "events": store.events_between(conn, t0, t0 + 86400),
            "emails": extract_event_mentions(conn, date)}


if __name__ == "__main__":
    rows = sync_calendar()
    zone = tz()
    print(f"synced {len(rows)} events for the next {DAYS_AHEAD} days")
    for r in rows[:30]:
        print(f"  {dt.datetime.fromtimestamp(r['start'], zone):%a %b %d %H:%M}  {r['title']}")
    sys.exit(0)
