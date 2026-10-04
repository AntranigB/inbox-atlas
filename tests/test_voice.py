import asyncio
import json

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlas import config, store
from atlas.api import voice as voice_api
from atlas.voice import cleanup as cl
from atlas.voice import realtime
from atlas.voice import tools as vtools

STT = "https://api.x.ai/v1/stt"
CHAT = "https://api.x.ai/v1/chat/completions"
RAW = "So like find me a emails about, no wait, the coding competition."


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setattr(config, "XAI_API_KEY", "test-key")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(voice_api.router)
    return TestClient(app)


def chat_reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})


def test_needs_llm():
    assert not cl.needs_llm("Coding competition emails")
    assert not cl.needs_llm("")
    assert cl.needs_llm("um the hackathon")
    assert cl.needs_llm("meet at 5, no wait, 6")
    assert cl.needs_llm("the the slides")


@respx.mock
def test_short_clean_text_skips_llm():
    route = respx.post(CHAT)
    assert asyncio.run(cl.cleanup("coding competition emails.", "query")) == "Coding competition emails"
    assert not route.called


@respx.mock
def test_cleanup_falls_back_on_error():
    respx.post(CHAT).mock(return_value=httpx.Response(500))
    assert asyncio.run(cl.cleanup("um  the hackathon")) == "Um the hackathon"


@respx.mock
def test_dictate(client):
    stt = respx.post(STT).mock(return_value=httpx.Response(200, json={"text": RAW, "language": "en", "words": []}))
    chat = respx.post(CHAT).mock(return_value=chat_reply("Find me the emails about the coding competition."))
    r = client.post("/api/dictate", files={"audio": ("a.webm", b"x" * 2000, "audio/webm")}, data={"style": "query"})
    assert r.status_code == 200
    j = r.json()
    assert j["raw"] == RAW and j["text"] == "Find me the emails about the coding competition."
    assert "ms" in j
    sent = stt.calls[0].request
    assert b"grok-voice-transcribe-2.0" in sent.content and b'filename="a.webm"' in sent.content
    body = json.loads(chat.calls[0].request.content)
    assert body["temperature"] == 0 and "search query" in body["messages"][0]["content"]


@respx.mock
def test_stt_route_and_error(client):
    respx.post(STT).mock(side_effect=[httpx.Response(200, json={"text": "hi"}), httpx.Response(400, text="bad audio")])
    r = client.post("/api/stt", files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert r.json()["text"] == "hi"
    r = client.post("/api/stt", files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 502


def test_to_realtime():
    chat = {"type": "function", "function": {"name": "get_email", "description": "d",
                                             "parameters": {"type": "object", "properties": {}}}}
    rt = vtools.to_realtime(chat)
    assert rt == {"type": "function", "name": "get_email", "description": "d",
                  "parameters": {"type": "object", "properties": {}}}
    assert vtools.to_realtime(rt) == rt


def test_stub_search(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "mail.sqlite")
    store.load_fixture(store.connect())
    out = vtools.stub_run_tool("search_region", {"positive": ["hackathon"], "k": 5})
    assert out["hits"] and all({"id", "from", "date", "subject"} <= h.keys() for h in out["hits"])
    one = vtools.stub_run_tool("get_email", {"id": out["hits"][0]["id"]})
    assert one["subject"] == out["hits"][0]["subject"]


def test_call_tool_handles_errors_and_async():
    def boom(name, args):
        raise ValueError("nope")

    async def aok(name, args):
        return {"hits": [{"id": "1"}], "args": args}

    assert "nope" in asyncio.run(vtools.call_tool(boom, "x", "{}"))["error"]
    assert asyncio.run(vtools.call_tool(aok, "x", '{"a":1}'))["args"] == {"a": 1}


class FakeUpstream:
    """Scripted Grok realtime: asks for one tool call, then answers with audio after the output."""

    def __init__(self):
        self.sent = []
        self.q = asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, raw):
        ev = json.loads(raw)
        self.sent.append(ev)
        t = ev["type"]
        if t == "input_audio_buffer.append":
            await self.q.put({"type": "response.function_call_arguments.done", "call_id": "c1",
                              "name": "search_region", "arguments": '{"positive":["hackathon"]}'})
            await self.q.put({"type": "response.done"})
        elif t == "response.create":
            await self.q.put({"type": "response.output_audio.delta", "delta": "AAAA"})
            await self.q.put({"type": "response.done"})

    def __aiter__(self):
        return self

    async def __anext__(self):
        return json.dumps(await self.q.get())


def test_ws_voice_runs_tools(client, monkeypatch):
    up = FakeUpstream()
    monkeypatch.setattr(realtime, "connect_upstream", lambda: up)
    calls = []

    def run_tool(name, args):
        calls.append((name, args))
        return {"region": {"size": 1}, "hits": [{"id": "e1", "from": "BigRed Hacks", "date": "2026-10-02",
                                                  "subject": "you're in", "snippet": "s", "body": "long"}]}

    monkeypatch.setattr(vtools, "load_tools", lambda: (vtools.STUB_SCHEMAS, run_tool))
    with client.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["type"] == "atlas.ready"
        ws.send_bytes(b"\x00\x01" * 100)
        seen = []
        while True:
            ev = ws.receive_json()
            seen.append(ev)
            if ev["type"] == "response.output_audio.delta":
                break
    types = [e["type"] for e in seen]
    assert "atlas.hits" in types
    hits = next(e for e in seen if e["type"] == "atlas.hits")
    assert hits["hits"][0]["id"] == "e1" and hits["args"] == {"positive": ["hackathon"]}
    assert calls == [("search_region", {"positive": ["hackathon"]})]
    sess = up.sent[0]
    assert sess["type"] == "session.update"
    assert sess["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert [t["name"] for t in sess["session"]["tools"]] == ["search_region", "get_email"]
    out = next(e for e in up.sent if e["type"] == "conversation.item.create")
    assert out["item"]["type"] == "function_call_output" and out["item"]["call_id"] == "c1"
    assert "body" not in json.loads(out["item"]["output"])["hits"][0]
    assert up.sent[-1]["type"] == "response.create"
