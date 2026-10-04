"""Inbox Atlas API server. Each feature branch adds a router in atlas/api/."""

import importlib
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from atlas import auth, config

log = logging.getLogger("atlas")
app = FastAPI(title="Inbox Atlas")
app.add_middleware(auth.TokenMiddleware)

for name in ("search", "voice", "messaging", "chat", "context"):
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
def health(request: Request):
    if not auth.check(request.scope):
        return {"ok": True, "auth_required": True}
    n = 0
    try:
        from atlas import store
        conn = store.connect()
        n = conn.execute("select count(*) from emails").fetchone()[0]
    except Exception:
        pass
    return {"ok": True, "encoder": config.ENCODER, "n_emails": n, "auth_required": bool(auth.token())}


app.mount("/", StaticFiles(directory=config.ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
