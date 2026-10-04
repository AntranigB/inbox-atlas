"""Obsidian ingest, token-budgeted context packs, /api/context and the MCP server (offline, hash encoder)."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atlas import config, store
from atlas.agent import grok
from atlas.context import pack
from atlas.ingest import obsidian
from atlas.search import engine as engmod
from atlas.search import hybrid

VAULT = Path(__file__).parent / "fixtures" / "vault"
CODING = {"positive": ["hackathon", "Codeforces round", "ICPC regional", "LeetCode weekly contest",
                       "DevPost submission", "Kaggle competition"], "negative": ["online assessment for internship"]}
SOURDOUGH = {"positive": ["sourdough baking", "starter feeding", "bulk ferment", "bake covered Dutch oven"],
             "negative": []}
YACHT = {"positive": ["boat hull cleaning", "marina slip rental"], "negative": []}


@pytest.fixture
def eng(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(engmod, "_try_load_index", lambda name: None)
    monkeypatch.setattr(config, "XAI_API_KEY", "")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    grok._exp_cache.clear()
    conn = store.connect(":memory:")
    store.load_fixture(conn)
    obsidian.load_vault(conn, VAULT)
    e = engmod.Engine("hash", conn)
    monkeypatch.setattr(engmod, "_engines", {})
    engmod.set_engine(e)
    return e


def test_chunk_and_frontmatter():
    fm, body = obsidian.split_frontmatter("---\ntitle: X\ntags: [a, b]\n---\n# Top\nhello world this is text\n")
    meta = obsidian.parse_frontmatter(fm)
    assert meta["title"] == "X" and meta["tags"] == ["a", "b"]
    text = "intro line that is long enough\n## Plan\n- dentist at 3:40 on Thursday\n```\n# not a heading\n```\n## Log\nwent to the gym today and then did laundry\n"
    secs = obsidian.chunk_note(text, "Day")
    assert [s[0] for s in secs] == ["Day", "Plan", "Log"]
    assert "# not a heading" in secs[1][2]
    assert obsidian.anchor("2026-02-20 - PhD Apps!") == "2026-02-20---phd-apps"


def test_vault_ingest_skips_and_links(eng):
    rows = [dict(r) for r in eng.conn.execute("select * from emails where source='obsidian'")]
    assert rows
    blob = " ".join(r["body"] for r in rows)
    assert "SKIPPED_TEMPLATE_TEXT" not in blob and "SKIPPED_ARCHIVE_TEXT" not in blob
    assert not any(r["id"].startswith("obs:templates/") or r["id"].startswith("obs:archive/") for r in rows)
    japan = [r for r in rows if r["thread_id"] == "obs:projects/Japan trip.md"]
    assert len(japan) > 1 and all("#" in r["id"] for r in japan)
    assert any(json.loads(r["to_addrs"]) for r in rows)  # wikilinks kept
    assert any("travel" in json.loads(r["labels"]) for r in japan)  # frontmatter tags kept
    # re-ingest replaces instead of duplicating
    n = len(rows)
    obsidian.load_vault(eng.conn, VAULT)
    assert eng.conn.execute("select count(*) from emails where source='obsidian'").fetchone()[0] == n


def test_source_filter(eng):
    res = hybrid.search("sourdough", SOURDOUGH["positive"], [], {"sources": ["obsidian"]}, k=5, engine=eng)
    assert res["hits"] and all(h["source"] == "obsidian" for h in res["hits"])
    res = hybrid.search("coding competition", CODING["positive"], [], {"sources": ["gmail"]}, k=5, engine=eng)
    assert res["hits"] and all(h["source"] == "gmail" for h in res["hits"])


def test_pack_under_budget(eng):
    for budget in (120, 300, 800):
        p = pack.build_context("coding competition", budget_tokens=budget, engine=eng, facets=CODING)
        assert p["answerable"] and p["items"]
        assert p["tokens"] <= budget
        assert p["tokens"] == pack.count_tokens(p["context"])
        assert all(it["source"] == "gmail" and it["uri"].startswith("gmail:") for it in p["items"])
    small = pack.build_context("coding competition", budget_tokens=120, engine=eng, facets=CODING)
    big = pack.build_context("coding competition", budget_tokens=1500, engine=eng, facets=CODING)
    assert len(big["items"]) >= len(small["items"])


def test_pack_extracts_sentences_from_notes(eng):
    p = pack.build_context("sourdough baking temperatures", budget_tokens=400, engine=eng, facets=SOURDOUGH,
                           sources=("obsidian",))
    assert p["answerable"] and p["items"]
    assert all(it["source"] == "obsidian" and "#" in it["uri"] for it in p["items"])
    assert p["naive_tokens"] > p["tokens"] and p["tokens_saved_vs_naive"] == p["naive_tokens"] - p["tokens"]


def test_pack_nothing_here(eng):
    p = pack.build_context("yacht maintenance", engine=eng, facets=YACHT)
    assert p["answerable"] is False and p["items"] == []
    assert p["status"] == "no_confident_match" and p["tokens"] < 40
    assert "may still exist" in p["context"]


def test_get_doc(eng):
    whole = pack.get_doc("projects/Japan trip.md", max_tokens=5000, conn=eng.conn)
    assert "K7QW2P" in whole["text"] and not whole["truncated"]
    cut = pack.get_doc("projects/Japan trip.md", max_tokens=50, conn=eng.conn)
    assert cut["truncated"] and cut["tokens"] <= 55
    sec = pack.get_doc("projects/Japan trip.md#flight-details", conn=eng.conn)
    assert "K7QW2P" in sec["text"]
    eid = eng.conn.execute("select id from emails where source!='obsidian'").fetchone()[0]
    assert pack.get_doc(f"gmail:{eid}", conn=eng.conn)["source"] == "gmail"
    assert pack.get_doc("nope.md", conn=eng.conn)["error"] == "not found"


def test_api_context(eng):
    import server

    c = TestClient(server.app)
    r = c.post("/api/context", json={"question": "yacht maintenance", "budget_tokens": 300})
    assert r.status_code == 200 and r.json()["answerable"] in (True, False)
    r = c.post("/api/context/get", json={"uri": "projects/Japan trip.md#flight-details"})
    assert "K7QW2P" in r.json()["text"]


def test_mcp_server_tools(eng):
    spec = importlib.util.spec_from_file_location("atlas_mcp_server", config.ROOT / "mcp" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tools = asyncio.run(mod.server.list_tools())
    assert {t.name for t in tools} == {"atlas_context", "atlas_search", "atlas_related", "atlas_get",
                                         "atlas_points_of_interest", "atlas_related_folders"}
    out = mod.atlas_get("projects/Japan trip.md#flight-details", 200)
    assert "K7QW2P" in out["text"]
    ctx = mod.atlas_context("coding competition", 300)
    assert "items" not in ctx and "context" in ctx
    assert ctx["response_tokens"] > ctx["tokens"]


@pytest.mark.parametrize("budget", [0, 1, 2, 8, 15, 40])
def test_truncation_includes_ellipsis_and_unicode(budget):
    text = "確認番号 🍊 résumé " * 50
    cut = pack.truncate_tokens(text, budget)
    assert pack.count_tokens(cut) <= budget
    assert not cut.endswith("\ufffd")


def test_short_values_stay_with_labels():
    ss = pack.sentences("Your booking reference:\nK7QW2P\nDeparture time:\n8am\n---\n$42")
    assert any("reference:" in s and "K7QW2P" in s for s in ss)
    assert any("Departure time:" in s and "8am" in s for s in ss)
    assert "$42" in " ".join(ss)


def test_hard_wrapped_event_and_flight_details_stay_together():
    text = "Hello,\n\nRobotics workshop\nThursday, November 12\n2pm to 4pm\nEngineering Room 205\n\nSee you there!"
    ss = pack.sentences(text)
    assert any(all(fact in s for fact in ("Robotics workshop", "November 12", "2pm", "Room 205")) for s in ss)
    wrapped = "Please bring the\nred notebook to our meeting. We will\nreview the experiments."
    assert "Please bring the red notebook to our meeting." in pack.sentences(wrapped)[0]


def test_tiny_budgets_and_long_headers(eng, monkeypatch):
    original = hybrid.search("coding competition", CODING["positive"], CODING["negative"], engine=eng)
    original["hits"] = [h for h in original["hits"] if h["member"]][:1]
    eid = original["hits"][0]["id"]
    eng.conn.execute("update emails set subject=? where id=?", ("Extremely long subject " * 200, eid))
    monkeypatch.setattr(hybrid, "search", lambda *a, **kw: original)
    for budget in (0, 1, 10, 40, 100):
        p = pack.build_context("coding competition", budget_tokens=budget, engine=eng, facets=CODING)
        assert p["tokens"] == pack.count_tokens(p["context"]) <= budget
        assert not p["answerable"] and p["status"] == "insufficient_budget"
        assert p["region"]["related"]  # can't fit is distinct from no match


def test_absent_pack_also_obeys_tiny_budget(eng):
    for budget in (0, 1, 10):
        p = pack.build_context("yacht maintenance", budget_tokens=budget, engine=eng, facets=YACHT)
        assert not p["answerable"] and p["tokens"] <= budget


def test_expansion_filters_reach_search(eng, monkeypatch):
    original = hybrid.search
    seen = []

    def search(*args, **kwargs):
        seen.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(hybrid, "search", search)
    pack.build_context("coding competition", engine=eng, facets={**CODING, "filters": {
        "after": "2026-09-01", "from": "bigredhacks", "sources": ["obsidian"]}}, sources=("gmail",))
    assert seen[0] == {"after": "2026-09-01", "from": "bigredhacks", "sources": ["gmail"]}


def test_duplicate_excerpts_are_not_repeated(eng, monkeypatch):
    res = hybrid.search("coding competition", CODING["positive"], CODING["negative"], engine=eng)
    res["hits"] = [h for h in res["hits"] if h["member"]][:2]
    body = "The programming contest begins tomorrow at noon in Gates Hall."
    for h in res["hits"]:
        eng.conn.execute("update emails set body=?, subject='Reminder', from_name='Organizer', date=1000 where id=?", (body, h["id"]))
    monkeypatch.setattr(hybrid, "search", lambda *a, **kw: res)
    p = pack.build_context("coding competition", engine=eng, facets=CODING)
    assert len(p["items"]) == 1
    assert p["context"].count(body) == 1
    row = eng.get_email(res["hits"][0]["id"])
    assert p["naive_tokens"] == pack.count_tokens(pack.doc_text(row))
    eng.conn.execute("update emails set subject='Different contest' where id=?", (res["hits"][1]["id"],))
    p = pack.build_context("coding competition", engine=eng, facets=CODING)
    assert len(p["items"]) == 2


def test_invalid_context_limits_and_compact_http(eng):
    import server
    c = TestClient(server.app)
    for args in ({"budget_tokens": -1}, {"k": 0}):
        assert c.post("/api/context", json={"question": "hello", **args}).status_code == 422
    with pytest.raises(ValueError):
        pack.build_context("hello", budget_tokens=-1, engine=eng)
    response = c.post("/api/context", json={"question": "hello", "compact": True})
    assert response.status_code == 200
    assert "items" not in response.json()
