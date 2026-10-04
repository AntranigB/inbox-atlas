"""Folder navigation over the nested synthetic vault (offline, hash encoder)."""

import importlib.util

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atlas import config, store
from atlas.agent import grok
from atlas.context import pack, tree
from atlas.ingest import obsidian
from atlas.search import engine as engmod

from test_context import VAULT, YACHT

SERVO = {"positive": ["servo calibration", "pulse width", "arm joint angles"], "negative": []}
RAMEN = {"positive": ["ramen broth", "pork bones simmer", "tare and toppings"], "negative": []}


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


def test_folder_tree_and_vectors(eng):
    t = tree.get_tree(eng)
    assert {"", "projects", "projects/robotics", "projects/robotics/arm", "kitchen/recipes"} <= set(t.folders)
    assert not any(p.startswith(("templates", "archive")) for p in t.folders)
    for f in t.folders.values():
        assert abs(np.linalg.norm(f.vec) - 1) < 1e-4 and abs(np.linalg.norm(f.local) - 1) < 1e-4
    arm, rob = t.folders["projects/robotics/arm"], t.folders["projects/robotics"]
    assert arm.n_notes == 2 and arm.n_direct_notes == 2 and rob.n_notes == 3 and rob.n_direct_notes == 1
    assert rob.n_sections > arm.n_sections and "robotics" in rob.top_tags and rob.last_modified == "2026-02-25"
    # the local vector leans toward the folder's own note (Rover) more than the plain mean does
    rover = np.flatnonzero(np.array(t.sec_folder) == "projects/robotics")
    rv = tree._norm(eng.index.E[t.sec_rows[rover]].mean(0))
    assert rv @ rob.local > rv @ rob.vec
    assert tree.get_tree(eng) is t  # cached


def test_points_of_interest_ranks_right_folder(eng):
    out = tree.points_of_interest("servo calibration pulse width", engine=eng, facets=SERVO)
    assert out["related"] and out["folders"]
    top = out["folders"][0]
    assert top["path"] == "projects/robotics/arm/"
    assert top["notes"][0]["uri"].startswith("projects/robotics/arm/Servo calibration.md#")
    assert top["excerpt"] and len(top["notes"]) <= 3
    # an ancestor that only repeats the same notes is not listed again
    assert "projects/robotics/" not in [f["path"] for f in out["folders"]]
    assert out["tokens"] == pack.count_tokens(out["context"]) and out["tokens"] < 200

    out = tree.points_of_interest("ramen broth", engine=eng, facets=RAMEN)
    top = out["folders"][0]
    assert out["related"] and top["path"].startswith("kitchen/")
    assert top["notes"][0]["uri"].startswith("kitchen/recipes/Ramen broth.md#")


def test_within_restricts(eng):
    out = tree.points_of_interest("servo calibration pulse width", engine=eng, facets=SERVO, within="projects/")
    assert out["within"] == "projects/" and out["folders"]
    assert all(f["path"].startswith("projects/") for f in out["folders"])
    assert all(n["uri"].startswith("projects/") for f in out["folders"] for n in f["notes"])
    out = tree.points_of_interest("servo calibration pulse width", engine=eng, facets=SERVO, within="kitchen")
    assert out["related"] is False and out["folders"] == []
    out = tree.points_of_interest("anything", engine=eng, facets=SERVO, within="nope/")
    assert out["related"] is False and out["folders"] == []


def test_related_folders_skip_lineage(eng):
    out = tree.related_folders("projects/robotics/arm", k=10, engine=eng)
    paths = [r["path"] for r in out["related"]]
    assert paths and not {"projects/", "projects/robotics/", "/", "projects/robotics/arm/"} & set(paths)
    assert all(-1 <= r["cosine"] <= 1 for r in out["related"])
    assert [r["cosine"] for r in out["related"]] == sorted((r["cosine"] for r in out["related"]), reverse=True)
    out = tree.related_folders("projects", k=10, engine=eng)
    assert not any(r["path"].startswith("projects/") for r in out["related"])
    # kitchen links to [[Sourdough]], which lives in notes/
    k = {r["path"]: r for r in tree.related_folders("kitchen", k=10, engine=eng)["related"]}
    assert "sourdough" in k["notes/"]["shared_links"]
    assert tree.related_folders("nope", engine=eng)["error"] == "not found"


def test_absent_topic(eng):
    out = tree.points_of_interest("yacht maintenance", engine=eng, facets=YACHT)
    assert out["related"] is False and out["folders"] == []
    assert "Stop searching" in out["context"] and out["tokens"] < 40


def test_api_and_mcp(eng):
    import server

    c = TestClient(server.app)
    r = c.post("/api/context/folders", json={"question": "yacht maintenance"})
    assert r.status_code == 200 and "folders" in r.json()
    r = c.get("/api/context/related_folders", params={"path": "projects/robotics/arm"})
    assert r.status_code == 200 and r.json()["related"]
    spec = importlib.util.spec_from_file_location("atlas_mcp_server_tree", config.ROOT / "mcp" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.atlas_related_folders("kitchen", 3)
    assert len(out["related"]) == 3
    out = mod.atlas_points_of_interest("yacht maintenance")
    assert set(out) >= {"related", "folders", "context", "tokens"}
