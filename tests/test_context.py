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
    assert "Stop searching" in p["context"] and p["tokens"] < 40


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
    assert {t.name for t in tools} == {"atlas_context", "atlas_search", "atlas_related", "atlas_get"}
    out = mod.atlas_get("projects/Japan trip.md#flight-details", 200)
    assert "K7QW2P" in out["text"]
