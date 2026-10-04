import pytest

# Secrets and per-machine settings from the developer's .env must never change test behavior.
LOCAL_ENV = ("ATLAS_TOKEN", "SPECTRUM_PROJECT_ID", "SPECTRUM_PROJECT_SECRET", "OWNER_PHONE", "OWNER_HANDLES",
             "GMAIL_USER", "GMAIL_APP_PASSWORD", "GCAL_ICS_URL")


@pytest.fixture(autouse=True)
def _isolated_engine_cache(monkeypatch):
    """Query logging can initialize an engine; don't reuse another test's database."""
    from atlas.search import engine

    monkeypatch.setattr(engine, "_engines", {})


@pytest.fixture(autouse=True)
def _sqlite_unless_pg(request, monkeypatch):
    """Tests run on SQLite even when .env sets DATABASE_URL, except the ones marked pg."""
    if request.node.get_closest_marker("pg") is None:
        monkeypatch.setenv("ATLAS_DB", "sqlite")


@pytest.fixture(autouse=True)
def _no_local_secrets(monkeypatch):
    from atlas import config

    for name in LOCAL_ENV:
        monkeypatch.delenv(name, raising=False)
        if hasattr(config, name):
            monkeypatch.setattr(config, name, "")
