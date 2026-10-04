import json

import httpx
import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient

from atlas import config, store
from atlas.agent import grok, tools
from atlas.search import engine as engmod
from atlas.search import hybrid, region


@pytest.fixture
def eng(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(engmod, "_try_load_index", lambda name: None)
    monkeypatch.setattr(config, "XAI_API_KEY", "")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    grok._exp_cache.clear()
    conn = store.connect(":memory:")
    rows = store.load_fixture(conn)
    e = engmod.Engine("hash", conn)
    e.topic = {r["id"]: r["topic"] for r in rows}
    monkeypatch.setattr(engmod, "_engines", {})
    engmod.set_engine(e)
    return e


CODING = (["hackathon", "Codeforces round", "ICPC regional", "LeetCode weekly contest", "DevPost submission",
           "Kaggle competition"], ["online assessment for internship"])


def test_region_interface():
    P = np.eye(4, dtype=np.float32)[:2]
    r = region.build_region(P, np.eye(4, dtype=np.float32)[2:3], labels=["a", "b"])
    E = np.eye(4, dtype=np.float32)
    s = r.score(E)
    assert s[0] > s[3] and s[2] < s[3]  # negative penalty pushes row 2 down
    d = r.describe(E, s > 0.5)
    assert set(d["facet_hits"]) == {"a", "b"} and "spread" in d


def test_coding_competition_region(eng):
    res = hybrid.search("coding competition", *CODING, k=10, mode="region", engine=eng)
    topics = [eng.topic[h["id"]] for h in res["hits"]]
    assert res["region"]["related"]
    assert topics[0] in ("hackathon", "contest")
    hr = next(i for i, h in enumerate(res["hits"]) if eng.topic[h["id"]] == "jobs" and "assessment" in h["subject"].lower()) \
        if any("assessment" in h["subject"].lower() for h in res["hits"]) else 99
    good = [i for i, t in enumerate(topics) if t in ("hackathon", "contest")]
    assert len(good) >= 3 and sorted(good)[2] < hr
    assert "Kaggle competition" in res["region"]["facet_hits"]


def test_absent_topic_not_related(eng):
    out = hybrid.is_related("yacht maintenance", ["boat hull cleaning", "marina slip rental"], engine=eng)
    assert out["related"] is False and out["count"] == 0


def test_modes_and_filters(eng):
    kw = hybrid.search("Codeforces", mode="keyword", engine=eng)
    assert kw["hits"][0]["from"] == "Codeforces"
    em = hybrid.search("Codeforces round", mode="embed", engine=eng)
    assert em["hits"][0]["from"] == "Codeforces"
    hy = hybrid.search("coding competition", *CODING, mode="hybrid", engine=eng)
    assert hy["hits"]
    f = hybrid.search("coding competition", *CODING, filters={"from": "leetcode"}, mode="region", engine=eng)
    assert {h["from"] for h in f["hits"]} == {"LeetCode"}
    late = hybrid.search("coding competition", *CODING, filters={"after": "2026-09-20"}, mode="region", engine=eng)
    assert all(h["date"] >= 1789862400 for h in late["hits"])


def test_rrf():
    order, _ = hybrid.rrf([["a", "b", "c"], ["b", "c", "a"]])
    assert order[0] == "b"


def test_tools_and_watches(eng):
    names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert names == {"search_region", "is_related", "get_email", "list_clusters", "todays_agenda", "add_watch", "list_watches"}
    out = tools.run_tool("search_region", {"positive": CODING[0], "negative": CODING[1]})
    assert out["region"]["facet_hits"] and "_full" not in out
    json.dumps(out)
    assert tools.run_tool("get_email", {"id": out["hits"][0]["id"]})["body"]
    assert tools.run_tool("list_clusters")
    ag = tools.run_tool("todays_agenda", {"date": "2026-10-04"})
    assert any("Spin" in (e["snippet"] or "") for e in ag["emails"])
    w = tools.run_tool("add_watch", {"name": "coding contests", "positive": ["Codeforces round", "LeetCode weekly contest"]})
    assert w["id"] and tools.list_watches()[0]["name"] == "coding contests"
    store.upsert_emails(eng.conn, [{"id": "new1", "from_name": "Codeforces", "from_addr": "noreply@codeforces.com",
                                    "subject": "Codeforces Round 1050 (Div. 1) registration open",
                                    "body": "Codeforces Round 1050 starts Saturday. Register now.", "date": 1791000000},
                                   {"id": "new2", "from_name": "Amazon", "subject": "Your order has shipped",
                                    "body": "Socks arriving Monday.", "date": 1791000000}])
    hits = grok.check_watches(["new1", "new2"])
    assert [h["id"] for h in hits] == ["new1"] and hits[0]["watch_name"] == "coding contests"
    assert tools.delete_watch("coding contests")["deleted"] == 1


def _grok_reply(content=None, tool_calls=None):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content,
                                                                "tool_calls": tool_calls}}]})


@respx.mock
def test_ask_tool_loop(eng, monkeypatch):
    monkeypatch.setattr(config, "XAI_API_KEY", "test")
    calls = iter([
        _grok_reply(tool_calls=[{"id": "t1", "type": "function", "function": {
            "name": "search_region", "arguments": json.dumps({"positive": CODING[0], "negative": CODING[1]})}}]),
        _grok_reply(content="**Codeforces** round 1043 starts Sunday — good luck."),
    ])
    route = respx.post(grok.CHAT_URL).mock(side_effect=lambda req: next(calls))
    out = grok.ask("any coding competitions?", channel="imessage")
    assert route.call_count == 2
    assert "*" not in out["reply"] and "—" not in out["reply"]
    assert out["hits"] and out["region"]["size"] >= 1
    sent = json.loads(route.calls[0].request.content)
    assert len(sent["tools"]) == 7


@respx.mock
def test_ask_caps_tool_calls(eng, monkeypatch):
    monkeypatch.setattr(config, "XAI_API_KEY", "test")
    tc = [{"id": "t", "type": "function", "function": {"name": "list_watches", "arguments": "{}"}}]
    route = respx.post(grok.CHAT_URL).mock(side_effect=lambda req: _grok_reply(
        content="done" if "tools" not in json.loads(req.content) else None,
        tool_calls=None if "tools" not in json.loads(req.content) else tc))
    out = grok.ask("loop forever", channel="voice")
    assert out["reply"] == "done" and route.call_count == grok.MAX_TOOL_CALLS + 1


@respx.mock
def test_expand_and_api(eng, monkeypatch):
    monkeypatch.setattr(config, "XAI_API_KEY", "test")
    respx.post(grok.CHAT_URL).mock(return_value=_grok_reply(content=json.dumps(
        {"positive": CODING[0], "negative": CODING[1], "filters": {"after": None}, "intent": "find"})))
    import server
    c = TestClient(server.app)
    r = c.post("/api/search", json={"query": "coding competition", "mode": "region", "k": 5}).json()
    assert r["expansion"]["positive"] == CODING[0] and r["region"]["related"] and r["facet_points"]
    assert c.post("/api/search", json={"query": "Codeforces", "mode": "keyword"}).json()["hits"]
    assert c.get("/api/related", params={"topic": "coding competition"}).json()["related"]
    m = c.get("/api/map").json()
    assert m["points"] and m["clusters"]
    assert c.get(f"/api/email/{r['hits'][0]['id']}").json()["body"]
    assert c.post("/api/watches", json={"name": "rent", "positive": ["lease"]}).json()["id"]
    assert c.get("/api/watches").json()[0]["name"] == "rent"
    assert "events" in c.get("/api/agenda", params={"date": "2026-10-04"}).json()


@pytest.mark.slow
def test_coding_competition_base_encoder(monkeypatch):
    monkeypatch.setattr(engmod, "_try_load_index", lambda name: None)
    conn = store.connect(":memory:")
    rows = store.load_fixture(conn)
    topic = {r["id"]: r["topic"] for r in rows}
    e = engmod.Engine("base", conn)
    pos = CODING[0] + ["Google Code Jam", "AtCoder contest"]
    neg = ["online coding assessment", "job coding interview", "take-home coding test"]
    res = hybrid.search("coding competition", pos, neg, k=24, mode="region", engine=e)
    order = [topic[h["id"]] for h in res["hits"]]
    hr = next(i for i, h in enumerate(res["hits"]) if "assessment" in h["subject"].lower())
    assert all(i < hr for i, t in enumerate(order) if t in ("hackathon", "contest"))
    assert not hybrid.is_related("yacht maintenance", ["boat hull cleaning", "marina slip rental"], engine=e)["related"]


def test_ask_without_key_falls_back(eng):
    out = grok.ask("Codeforces round", channel="imessage")
    assert out["reply"] and len(out["reply"]) <= 600
