"""Optional shared token for the API (ATLAS_TOKEN).

When ATLAS_TOKEN is set, every /api and /ws route needs it, as `Authorization: Bearer <t>`,
`X-Atlas-Token: <t>` or `?token=<t>`. Static files (the web UI, manifest) stay public so the
page can load and ask for the token. /api/health answers without a token but only says ok.
"""

from __future__ import annotations

import hmac

from starlette.responses import JSONResponse

from atlas import config

OPEN_PATHS = {"/api/health"}


def token() -> str:
    return config.env("ATLAS_TOKEN") or config.ATLAS_TOKEN


def _given(scope) -> str:
    headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers") or []}
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if headers.get("x-atlas-token"):
        return headers["x-atlas-token"].strip()
    from urllib.parse import parse_qs
    q = parse_qs((scope.get("query_string") or b"").decode())
    return (q.get("token") or [""])[0]


def check(scope) -> bool:
    t = token()
    if not t:
        return True
    return hmac.compare_digest(_given(scope).encode(), t.encode())


def protected(path: str) -> bool:
    return (path.startswith("/api/") or path.startswith("/ws/")) and path not in OPEN_PATHS


class TokenMiddleware:
    """Pure ASGI so it also covers websockets."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and protected(scope.get("path", "")) and not check(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            await JSONResponse({"detail": "token required"}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)
