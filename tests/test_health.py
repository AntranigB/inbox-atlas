from fastapi.testclient import TestClient


def test_health(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_DATA", str(tmp_path))
    import server
    r = TestClient(server.app).get("/api/health")
    assert r.status_code == 200 and r.json()["ok"]
