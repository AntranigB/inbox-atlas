"""Voice routes: raw STT, Whisperflow-style dictation, realtime voice agent proxy."""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket

from atlas.voice import realtime
from atlas.voice.cleanup import cleanup, warmup
from atlas.voice.stt import STTError, transcribe

_bg: set = set()


@asynccontextmanager
async def lifespan(app):
    t = asyncio.create_task(warmup())  # first dictation skips Grok's cold path
    _bg.add(t)
    t.add_done_callback(_bg.discard)
    yield


router = APIRouter(lifespan=lifespan)


async def _stt(audio: UploadFile) -> dict:
    data = await audio.read()
    try:
        return await transcribe(data, audio.filename or "clip.webm", audio.content_type or "audio/webm")
    except STTError as e:
        raise HTTPException(502, str(e))


@router.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    t0 = time.perf_counter()
    out = await _stt(audio)
    return {"text": out.get("text", ""), "language": out.get("language"), "duration": out.get("duration"),
            "words": out.get("words", []), "ms": round((time.perf_counter() - t0) * 1000)}


@router.post("/api/dictate")
async def dictate(audio: UploadFile = File(...), style: str = Form("plain")):
    t0 = time.perf_counter()
    out = await _stt(audio)
    t1 = time.perf_counter()
    raw = out.get("text", "")
    text = await cleanup(raw, style)
    t2 = time.perf_counter()
    return {"raw": raw, "text": text, "ms": round((t2 - t0) * 1000),
            "stt_ms": round((t1 - t0) * 1000), "cleanup_ms": round((t2 - t1) * 1000)}


@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await realtime.serve(ws)
