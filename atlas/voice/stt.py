"""Grok speech to text. Accepts wav, webm/opus, ogg, mp3 and more as-is."""

import asyncio

import httpx

from atlas import config

STT_URL = f"{config.XAI_BASE}/stt"
STT_MODEL = config.env("GROK_STT_MODEL", "grok-voice-transcribe-2.0")

_clients: dict[int, httpx.AsyncClient] = {}


def client() -> httpx.AsyncClient:
    """One pooled client per event loop, so keep-alive saves the TLS handshake."""
    key = id(asyncio.get_running_loop())
    c = _clients.get(key)
    if c is None or c.is_closed:
        c = _clients[key] = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    return c


class STTError(RuntimeError):
    pass


async def transcribe(audio_bytes: bytes, filename: str = "clip.webm", mime: str = "audio/webm") -> dict:
    """Returns Grok's response: {text, language, duration, words:[{text,start,end}]}."""
    if not config.XAI_API_KEY:
        raise STTError("XAI_API_KEY is not set")
    if not audio_bytes:
        return {"text": "", "language": None, "duration": 0.0, "words": []}
    data = {"model": STT_MODEL}
    r = await client().post(
        STT_URL,
        headers={"Authorization": f"Bearer {config.XAI_API_KEY}"},
        data=data,
        files={"file": (filename or "clip.webm", audio_bytes, mime or "application/octet-stream")},
    )
    if r.status_code != 200:
        raise STTError(f"stt failed {r.status_code}: {r.text[:200]}")
    out = r.json()
    out.setdefault("text", "")
    out.setdefault("words", [])
    return out
