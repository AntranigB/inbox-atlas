import pytest


@pytest.fixture(autouse=True)
def _sqlite_unless_pg(request, monkeypatch):
    """Tests run on SQLite even when .env sets DATABASE_URL, except the ones marked pg."""
    if request.node.get_closest_marker("pg") is None:
        monkeypatch.setenv("ATLAS_DB", "sqlite")
