"""Inbox Atlas API server. Each feature branch adds a router in atlas/api/."""

import importlib
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from atlas import config

log = logging.getLogger("atlas")
app = FastAPI(title="Inbox Atlas")

for name in ("search", "voice", "messaging"):
    try:
        mod = importlib.import_module(f"atlas.api.{name}")
        app.include_router(mod.router)
        if hasattr(mod, "on_startup"):
            app.router.on_startup.append(mod.on_startup)
    except ModuleNotFoundError as e:
        if e.name != f"atlas.api.{name}":
            raise
        log.warning("router %s not present yet", name)


@app.get("/api/health")
def health():
    n = 0
    try:
        from atlas import store
        conn = store.connect()
        n = conn.execute("select count(*) from emails").fetchone()[0]
    except Exception:
        pass
    return {"ok": True, "encoder": config.ENCODER, "n_emails": n}


app.mount("/", StaticFiles(directory=config.ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
