import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from atlas import config, notify, store

NY = ZoneInfo("America/New_York")
DEMO_SUNDAY = datetime(2026, 10, 4, 8, 0, tzinfo=NY)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "mail.sqlite")
    monkeypatch.setenv("TIMEZONE", "America/New_York")
    conn = store.connect(tmp_path / "mail.sqlite")
    store.load_fixture(conn)
    return conn


def make(tmp_path, **kw):
    sent = []
    kw.setdefault("send", lambda text: sent.append(text))
    n = notify.Notifier(state_path=tmp_path / "state.json", now=lambda: DEMO_SUNDAY, **kw)
    return n, sent


def test_agenda_prompt_asks_for_todays_agenda():
    p = notify.agenda_prompt("What do I have today?", DEMO_SUNDAY)
    assert "todays_agenda" in p and "2026-10-04" in p
    assert "2026-10-05" in notify.agenda_prompt("anything tomorrow?", DEMO_SUNDAY)
    assert notify.agenda_prompt("internship emails?", DEMO_SUNDAY) == "internship emails?"


def test_local_agenda_finds_spin_and_demo_day(db):
    a = notify.local_agenda(DEMO_SUNDAY.date(), db)
    subjects = [e["subject"] for e in a["emails"]]
    assert "Gym class booking confirmed" in subjects
    assert "Reminder: demo day Sunday 9am" in subjects
    assert "Dinner Sunday?" in subjects
    assert "Re: lab meeting notes" not in subjects  # Monday
    assert "CS 4780 prelim moved" not in subjects  # Oct 13


def test_brief_uses_agent_with_today_prompt(db, tmp_path):
    prompts = []

    def fake_ask(text):
        prompts.append(text)
        return "Demo day judging 9am in PSB, spin at 5pm at Helen Newman."

    n, sent = make(tmp_path, ask=fake_ask, conn=db)
    assert n.brief_due(DEMO_SUNDAY)
    n.tick()
    assert "todays_agenda" in prompts[0] and "2026-10-04" in prompts[0]
    assert len(sent) == 1 and sent[0].startswith("Good morning. Sunday Oct 4")
    assert "spin at 5pm" in sent[0]
    assert not n.brief_due(DEMO_SUNDAY)  # once a day
    n.tick()
    assert len(sent) == 1


def test_brief_falls_back_to_fixture_agenda_when_agent_missing(db, tmp_path):
    def broken(text):
        raise ImportError("atlas.agent.grok")

    n, sent = make(tmp_path, ask=broken, conn=db)
    text = n.build_brief(DEMO_SUNDAY)
    assert "Spin, Sunday Oct 4 at 5:00pm" in text
    assert "demo day Sunday 9am" in text


def test_brief_window_and_toggle(tmp_path):
    n, _ = make(tmp_path)
    assert not n.brief_due(DEMO_SUNDAY.replace(hour=7, minute=59))
    assert not n.brief_due(DEMO_SUNDAY.replace(hour=23))
    n.set_brief(False)
    assert not n.brief_due(DEMO_SUNDAY)
    assert n.set_brief(True, "07:30") == {"enabled": True, "time": "07:30"}


def test_poll_texts_watch_hits_deduped_and_respects_quiet_hours(tmp_path):
    hit = {"watch_name": "internships", "name": "internships", "watch": {"id": "w1", "name": "internships"},
           "id": "e1", "from": "Jane Recruiter", "subject": "Interview invite", "z": 4.2}
    checked = []
    n, sent = make(tmp_path, sync=lambda: ["e1"], check=lambda ids: checked.append(ids) or [hit])
    n.set_brief(False)
    noon = DEMO_SUNDAY.replace(hour=12)
    assert n.poll(noon) == ["New email in internships: Jane Recruiter: Interview invite"]
    assert checked == [["e1"]]
    assert sent == ["New email in internships: Jane Recruiter: Interview invite"]
    n.poll(noon)
    assert len(sent) == 1  # deduped

    hit2 = dict(hit, id="e2", subject="Offer letter")
    hit3 = dict(hit, id="e3", subject="Team match")
    n._check = lambda ids: [hit2, hit3]
    n.poll(DEMO_SUNDAY.replace(hour=23))  # quiet hours, held
    assert len(sent) == 1 and n.status()["pending"] == 2
    n.flush(DEMO_SUNDAY.replace(hour=7, minute=30))
    assert len(sent) == 2 and sent[1].startswith("2 new emails in your watches")
    # state survives a restart
    n2, _ = make(tmp_path, sync=lambda: ["e1"], check=lambda ids: [hit])
    assert n2.poll(noon) == []


def test_poll_skips_without_ingest(tmp_path, monkeypatch):
    monkeypatch.delenv("GMAIL_USER", raising=False)
    n, sent = make(tmp_path)
    assert n.poll() == [] and sent == []


def test_send_text_posts_to_sidecar(monkeypatch):
    calls = []

    class R:
        status_code = 200

        def json(self):
            return {"ok": True, "to": "+15555550100"}

    monkeypatch.setattr(notify.httpx, "post", lambda url, json, timeout: calls.append((url, json)) or R())
    monkeypatch.setenv("SIDECAR_URL", "http://localhost:8766")
    assert notify.send_text("hi")["ok"]
    assert calls == [("http://localhost:8766/send", {"text": "hi"})]


def test_send_text_unreachable_sidecar(monkeypatch):
    monkeypatch.setenv("SIDECAR_URL", "http://127.0.0.1:9")
    with pytest.raises(notify.NotifyError, match="sidecar not reachable"):
        notify.send_text("hi", timeout=1)


def test_api_routes(db, tmp_path, monkeypatch):
    import server

    sent = []
    monkeypatch.setitem(sys.modules, "atlas.agent.tools", None)  # exercise the direct DB path
    monkeypatch.setitem(sys.modules, "atlas.agent.grok", None)  # offline: no facet expansion
    monkeypatch.setattr(notify, "send_text", lambda text, to=None: sent.append(text) or {"ok": True})
    monkeypatch.setattr(notify, "_notifier", notify.Notifier(state_path=tmp_path / "s.json", conn=db))
    c = TestClient(server.app)
    assert c.post("/api/notify", json={"text": "hello"}).json() == {"ok": True}
    assert sent == ["hello"]
    st = c.get("/api/notify/status").json()
    assert st["running"] is False and "brief_time" in st
    assert c.post("/api/notify/brief", json={"enabled": False}).json()["enabled"] is False
    w = c.post("/api/notify/watches", json={"name": "internships", "topic": "internships"}).json()
    assert w["name"] == "internships"
    assert [x["name"] for x in c.get("/api/notify/watches").json()] == ["internships"]
    assert c.delete("/api/notify/watches/Internships").json()["removed"] == 1
    assert c.get("/api/notify/watches").json() == []


def test_notify_route_reports_sidecar_error(monkeypatch):
    import server

    def boom(text, to=None):
        raise notify.NotifyError("cannot start a chat, text the line first")

    monkeypatch.setattr(notify, "send_text", boom)
    r = TestClient(server.app).post("/api/notify", json={"text": "x"})
    assert r.status_code == 502 and "text the line first" in r.json()["detail"]


def test_agenda_route_on_fixture(db):
    import server

    a = TestClient(server.app).get("/api/notify/agenda", params={"date": "2026-10-04"}).json()
    text = str(a)
    assert "Spin" in text and "demo day" in text
