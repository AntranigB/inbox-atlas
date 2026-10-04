import datetime as dt
from zoneinfo import ZoneInfo

from atlas import calendar, store

NY = ZoneInfo("America/New_York")

ICS = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//test//EN
BEGIN:VEVENT
UID:weekly-1
SUMMARY:CS 4780 Lecture
DTSTART;TZID=America/New_York:20260929T101000
DTEND;TZID=America/New_York:20260929T112500
RRULE:FREQ=WEEKLY;BYDAY=TU,TH
LOCATION:Gates G01
END:VEVENT
BEGIN:VEVENT
UID:once-1
SUMMARY:BigRed Hacks demo
DTSTART:20261004T130000Z
DTEND:20261004T150000Z
END:VEVENT
BEGIN:VEVENT
UID:allday-1
SUMMARY:Mom birthday
DTSTART;VALUE=DATE:20261010
DTEND;VALUE=DATE:20261011
END:VEVENT
BEGIN:VEVENT
UID:old-1
SUMMARY:Way in the past
DTSTART:20250101T130000Z
DTEND:20250101T140000Z
END:VEVENT
END:VCALENDAR
"""


def test_parse_ics_expands_recurring():
    start = dt.datetime(2026, 10, 3, 0, 0, tzinfo=NY)
    rows = calendar.parse_ics(ICS, start=start, days=14)
    titles = [r["title"] for r in rows]
    assert titles.count("CS 4780 Lecture") == 4  # Oct 6, 8, 13, 15
    assert "BigRed Hacks demo" in titles and "Mom birthday" in titles
    assert "Way in the past" not in titles
    lec = next(r for r in rows if r["title"] == "CS 4780 Lecture")
    s = dt.datetime.fromtimestamp(lec["start"], NY)
    assert (s.month, s.day, s.hour, s.minute) == (10, 6, 10, 10) and lec["location"] == "Gates G01"
    assert len({r["id"] for r in rows}) == len(rows)
    conn = store.connect(":memory:")
    store.upsert_events(conn, rows)
    t0 = int(dt.datetime(2026, 10, 4, tzinfo=NY).timestamp())
    assert [e["title"] for e in store.events_between(conn, t0, t0 + 86400)] == ["BigRed Hacks demo"]


def test_mentioned_dates():
    sent = dt.date(2026, 10, 1)  # a Thursday
    d = calendar.mentioned_dates
    oct4 = dt.date(2026, 10, 4)
    assert oct4 in d("Demo is on Oct 4 at 1pm", sent)
    assert oct4 in d("see you October 4th", sent)
    assert oct4 in d("due 10/4", sent)
    assert oct4 in d("due 2026-10-04", sent)
    assert oct4 in d("happening this Sunday", sent)
    assert oct4 in d("on the 4th of October", sent)
    assert dt.date(2026, 10, 2) in d("tomorrow works", sent)
    assert dt.date(2026, 10, 1) in d("call me today", sent)
    assert d("the sun is out", sent) == set()
    assert dt.date(2027, 1, 5) in d("Jan 5 kickoff", dt.date(2026, 12, 20))


def test_extract_event_mentions_on_fixture():
    conn = store.connect(":memory:")
    store.load_fixture(conn)
    zone = NY
    base = int(dt.datetime(2026, 10, 2, 9, tzinfo=zone).timestamp())
    store.upsert_emails(conn, [
        {"id": "m1", "subject": "Demo time", "body": "Judging starts Sunday at 10am in Klarman.", "date": base,
         "from_name": "Hacks", "source": "test"},
        {"id": "m2", "subject": "Dinner", "body": "Are we still on for tomorrow?", "date": base + 86400,
         "from_name": "Sam", "source": "test"},
        {"id": "m3", "subject": "Unrelated", "body": "Next week maybe.", "date": base, "source": "test"},
    ])
    hits = calendar.extract_event_mentions(conn, "2026-10-04")
    ids = [h["id"] for h in hits]
    assert "m1" in ids and "m2" in ids and "m3" not in ids  # m2 sent Oct 3 says tomorrow
    assert "Sunday" in next(h for h in hits if h["id"] == "m1")["mention"]
    assert "m2" not in [h["id"] for h in calendar.extract_event_mentions(conn, dt.date(2026, 10, 5))]
    ag = calendar.agenda(conn, "2026-10-04")
    assert ag["date"] == "2026-10-04" and "m1" in [e["id"] for e in ag["emails"]]
