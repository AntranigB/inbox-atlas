"""Chat sessions, token auth, daemon supervisor and LaunchAgent plist (offline)."""

import json
import plistlib
import sys
import time

import pytest
from fastapi.testclient import TestClient

from atlas import chat, cli, config, daemon


@pytest.fixture
def store(tmp_path):
    s = chat.SqliteChat(tmp_path / "chat.sqlite")
    chat.reset_store(s)
    yield s
    chat.reset_store(None)


def test_chat_keeps_history_per_session(store):
    seen = []

    def ask(text, channel, history):
        seen.append(list(history))
        return {"reply": f"answer to {text}", "hits": []}

    a = chat.chat("first", None, "web", ask=ask)
    sid = a["session_id"]
    assert sid.startswith("web:") and a["history_used"] == 0
    b = chat.chat("second", sid, "web", ask=ask)
    assert b["history_used"] == 2
    assert seen[1] == [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer to first"}]
    chat.chat("other", "imessage:+15555550100", "imessage", ask=ask)
    assert seen[2] == []
    sessions = {s["id"]: s for s in store.list_sessions()}
    assert sessions[sid]["n"] == 4 and sessions[sid]["title"] == "first"
    assert sessions["imessage:+15555550100"]["channel"] == "imessage"
    assert store.delete_session(sid) == 1
    assert store.get_history(sid) == []


def _client(monkeypatch, token=""):
    monkeypatch.setenv("ATLAS_TOKEN", token)
    monkeypatch.setenv("NOTIFY", "0")
    import server
    return TestClient(server.app)


def test_chat_endpoint(monkeypatch, store):
    from atlas.agent import grok
    monkeypatch.setattr(grok, "ask", lambda text, channel="web", history=None: {"reply": f"{len(history or [])}:{text}"})
    c = _client(monkeypatch)
    r1 = c.post("/api/chat", json={"text": "hi"}).json()
    r2 = c.post("/api/chat", json={"text": "again", "session_id": r1["session_id"]}).json()
    assert r2["reply"] == "2:again" and r2["session_id"] == r1["session_id"]
    ss = c.get("/api/chat/sessions").json()
    assert ss[0]["id"] == r1["session_id"]
    msgs = c.get(f"/api/chat/sessions/{r1['session_id']}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert c.post("/api/chat", json={"text": " "}).status_code == 400


def test_token_guards_api_and_ws(monkeypatch, store):
    c = _client(monkeypatch, "s3cret")
    assert c.get("/api/chat/sessions").status_code == 401
    assert c.get("/api/chat/sessions", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/api/chat/sessions", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/api/chat/sessions", headers={"X-Atlas-Token": "s3cret"}).status_code == 200
    assert c.get("/api/chat/sessions?token=s3cret").status_code == 200
    h = c.get("/api/health").json()
    assert h == {"ok": True, "auth_required": True}
    assert "n_emails" in c.get("/api/health?token=s3cret").json()
    assert c.get("/").status_code == 200  # the UI itself loads, then asks for the token
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/voice") as ws:
            ws.receive_text()


def test_no_token_is_open_and_localhost(monkeypatch, store):
    c = _client(monkeypatch, "")
    assert c.get("/api/chat/sessions").status_code == 200
    assert c.get("/api/health").json()["auth_required"] is False


def test_service_restarts_with_backoff(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "LOGS", tmp_path)
    s = daemon.Service("boom", [sys.executable, "-c", "import sys; print('Error: kaboom'); sys.exit(3)"], tmp_path)
    s.start()
    s.proc.wait(10)
    assert s.poll() is False  # died, waiting out the backoff
    assert s.restarts == 1 and s.last_exit == 3 and "kaboom" in s.last_error
    assert s.next_start > time.time()
    s.next_start = 0
    assert s.poll() is True  # restarted
    s.proc.wait(10)
    s.poll()
    assert s.restarts == 2 and s.next_start - time.time() > 1  # 2s backoff after the second failure
    s.stop()


def test_service_stop_kills_group(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "LOGS", tmp_path)
    s = daemon.Service("sleepy", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path)
    s.start()
    pid = s.proc.pid
    assert daemon.alive(pid)
    s.stop(timeout=5)
    assert not daemon.alive(pid)


def test_sidecar_only_with_credentials(monkeypatch):
    for k in ("SPECTRUM_PROJECT_ID", "SPECTRUM_PROJECT_SECRET", "ATLAS_SIDECAR"):
        monkeypatch.setenv(k, "")
    svcs, skipped = daemon.build_services()
    assert [s.name for s in svcs] == ["api"] and "SPECTRUM" in skipped["imessage"]
    monkeypatch.setenv("ATLAS_SIDECAR", "mock")
    monkeypatch.setenv("PORT", "9911")
    monkeypatch.setenv("SIDECAR_PORT", "9912")
    svcs, _ = daemon.build_services()
    side = [s for s in svcs if s.name == "imessage"]
    if side:  # node installed
        assert "--dry-run" in side[0].cmd and side[0].env["ATLAS_URL"] == "http://127.0.0.1:9911"
    assert svcs[0].env["SIDECAR_URL"] == "http://127.0.0.1:9912"


def test_launch_agent_plist(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PLIST", tmp_path / "com.inboxatlas.daemon.plist")
    assert cli.main(["install-service"]) == 0
    p = plistlib.loads(cli.PLIST.read_bytes())
    assert p["Label"] == "com.inboxatlas.daemon" and p["KeepAlive"] is True and p["RunAtLoad"] is True
    assert p["ProgramArguments"][-3:] == ["atlas.cli", "up", "--foreground"]
    assert p["WorkingDirectory"] == str(config.ROOT)
    assert "launchctl bootstrap" in capsys.readouterr().out
    assert cli.main(["uninstall-service"]) == 0
    assert not cli.PLIST.exists()


def test_status_when_stopped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(daemon, "PID_FILE", tmp_path / "daemon.pid")
    monkeypatch.setattr(daemon, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setenv("PORT", "1")
    monkeypatch.setenv("SIDECAR_PORT", "2")
    assert daemon.status() == 0
    out = capsys.readouterr().out
    assert "DOWN" in out and "imessage" in out
    assert daemon.status(as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["daemon"]["running"] is False
