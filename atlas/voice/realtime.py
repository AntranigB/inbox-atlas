"""Browser <-> server <-> Grok realtime voice proxy. The API key never leaves the server.

Browser protocol on /ws/voice:
- binary frames: raw PCM16 mono little endian at RATE Hz (mic audio)
- JSON frames: any OpenAI Realtime client event (input_audio_buffer.*, response.create,
  response.cancel, conversation.item.create), or {"type":"atlas.text","text":...} to type a turn.
- Server sends every Grok event as JSON, plus {"type":"atlas.hits","hits","region","tool","args"}
  when a search tool returns, and {"type":"atlas.error","message"} on failures.
"""

import asyncio
import base64
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from atlas import config
from atlas.voice import tools as vtools

log = logging.getLogger("atlas.voice")

REALTIME_URL = f"wss://api.x.ai/v1/realtime?model={config.env('GROK_VOICE_MODEL', 'grok-voice-latest')}"
RATE = 24000
VOICE = config.env("GROK_VOICE", "eve")
ALLOWED_CLIENT = {"input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.clear",
                  "response.create", "response.cancel", "conversation.item.create"}


def instructions() -> str:
    now = datetime.now(ZoneInfo(config.TIMEZONE))
    return (
        "You are Inbox Atlas, a voice assistant over the user's own email. "
        f"Today is {now:%A %B %d, %Y}, local time {now:%I:%M %p}. "
        "Always use the tools to look things up; never invent emails. "
        "Answer in one to three short spoken sentences. When you mention an email, say who it is "
        "from and the date in a natural way, like 'BigRed Hacks wrote on October 2nd'. "
        "No lists, no markdown, no ids or URLs read aloud. If nothing matches, say so briefly."
    )


def session_update(schemas: list[dict]) -> dict:
    fmt = {"type": "audio/pcm", "rate": RATE}
    return {"type": "session.update", "session": {
        "voice": VOICE,
        "instructions": instructions(),
        "turn_detection": {"type": "server_vad"},
        "audio": {"input": {"format": fmt}, "output": {"format": fmt}},
        "tools": schemas,
    }}


def connect_upstream():
    return websockets.connect(REALTIME_URL, additional_headers={"Authorization": f"Bearer {config.XAI_API_KEY}"},
                              max_size=None, ping_interval=20)


class Bridge:
    def __init__(self, ws: WebSocket, upstream, schemas, run_tool):
        self.ws, self.up, self.schemas, self.run_tool = ws, upstream, schemas, run_tool
        self.pending: dict[str, asyncio.Task] = {}

    async def to_browser(self, ev: dict):
        await self.ws.send_text(json.dumps(ev))

    async def browser_loop(self):
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                await self.up.send(json.dumps({"type": "input_audio_buffer.append",
                                               "audio": base64.b64encode(msg["bytes"]).decode()}))
                continue
            try:
                ev = json.loads(msg.get("text") or "{}")
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "atlas.text":
                await self.up.send(json.dumps({"type": "conversation.item.create", "item": {
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": ev.get("text", "")}]}}))
                await self.up.send(json.dumps({"type": "response.create"}))
            elif t in ALLOWED_CLIENT:
                await self.up.send(json.dumps(ev))

    async def run_call(self, name: str, call_id: str, arguments):
        result = await vtools.call_tool(self.run_tool, name, arguments)
        if isinstance(result.get("hits"), list):
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                args = {}
            await self.to_browser({"type": "atlas.hits", "tool": name, "args": args,
                                   "hits": result["hits"], "region": result.get("region"),
                                   "full": result.get("_full")})
        return call_id, result

    async def flush_calls(self):
        """After a response that requested tools: submit every output, then ask for the answer."""
        tasks, self.pending = list(self.pending.values()), {}
        for call_id, result in await asyncio.gather(*tasks):
            await self.up.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": call_id, "output": vtools.compact_output(result)}}))
        await self.up.send(json.dumps({"type": "response.create"}))

    async def upstream_loop(self):
        async for raw in self.up:
            if isinstance(raw, bytes):  # binary transport is off, but pass audio through if it shows up
                await self.ws.send_bytes(raw)
                continue
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t == "response.function_call_arguments.done":
                cid = ev.get("call_id") or ev.get("item_id") or str(len(self.pending))
                if cid not in self.pending:
                    self.pending[cid] = asyncio.create_task(self.run_call(ev.get("name", ""), cid, ev.get("arguments", "{}")))
            elif t == "response.output_item.done" and (ev.get("item") or {}).get("type") == "function_call":
                it = ev["item"]
                cid = it.get("call_id")
                if cid and cid not in self.pending:
                    self.pending[cid] = asyncio.create_task(self.run_call(it.get("name", ""), cid, it.get("arguments", "{}")))
            await self.ws.send_text(raw if isinstance(raw, str) else raw.decode())
            if t == "response.done" and self.pending:
                await self.flush_calls()


async def serve(ws: WebSocket):
    await ws.accept()
    if not config.XAI_API_KEY:
        await ws.send_text(json.dumps({"type": "atlas.error", "message": "XAI_API_KEY is not set"}))
        await ws.close()
        return
    schemas, run_tool = vtools.load_tools()
    try:
        async with connect_upstream() as up:
            await up.send(json.dumps(session_update(schemas)))
            await ws.send_text(json.dumps({"type": "atlas.ready", "rate": RATE,
                                           "tools": [s.get("name") for s in schemas]}))
            b = Bridge(ws, up, schemas, run_tool)
            tasks = [asyncio.create_task(b.browser_loop()), asyncio.create_task(b.upstream_loop())]
            done, rest = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in rest:
                t.cancel()
            for t in done:
                if t.exception() and not isinstance(t.exception(), (WebSocketDisconnect, websockets.ConnectionClosed)):
                    raise t.exception()
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as e:
        log.exception("voice bridge failed")
        try:
            await ws.send_text(json.dumps({"type": "atlas.error", "message": str(e)[:300]}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
