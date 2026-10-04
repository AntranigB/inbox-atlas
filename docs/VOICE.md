# Voice

Three pieces, all backed by Grok. The API key stays on the server (`XAI_API_KEY` in `.env`).

## 1. Dictation API (Whisperflow style)

`uv run python server.py`, then:

```bash
curl -F audio=@clip.webm -F style=query http://localhost:8765/api/dictate
# {"raw": "So like find me a emails about, no wait, the coding competition.",
#  "text": "coding competition emails", "ms": 916, "stt_ms": 300, "cleanup_ms": 616}
curl -F audio=@clip.wav http://localhost:8765/api/stt     # raw Grok STT with word timings
```

- STT is `POST https://api.x.ai/v1/stt` (`grok-voice-transcribe-2.0`). It accepts wav, webm/opus,
  ogg/opus and mp3 as-is, so the browser's MediaRecorder output goes straight through.
- Cleanup (`atlas/voice/cleanup.py`) removes fillers and false starts, applies self-corrections
  ("by five, no, six" -> "by six"), fixes punctuation, and adds nothing. `style` is `plain`,
  `email`, `message` or `query`. Short utterances that are already clean skip the LLM.
- Env knobs: `GROK_CLEANUP_MODEL` (default `GROK_MODEL`; `grok-4-fast-non-reasoning` is about as
  fast), `CLEANUP_TIMEOUT` seconds (default 6, falls back to the raw transcript), `CLEANUP_HEDGE`
  seconds (default 1.5). Grok chat latency is usually 0.6 s but sometimes spikes to 5 s, so if the
  first request is slower than the hedge a second identical one is raced against it. The server
  also makes one warmup call at startup.

## 2. Browser (`web/voice.js`)

Loaded by `index.html`. It mounts into `<div id="voice-slot">` if present, else floats bottom right.

- **Dictate**: hold Space (when no text field has focus) or hold the mic button. A quick tap on
  the mic toggles recording instead. The cleaned text goes into `#q` (or the field you were typing
  in), then `input` and `atlas:query` (`detail: {text, raw}`) fire.
- **Talk to inbox**: opens `/ws/voice`, streams mic PCM16 at 24 kHz, plays Grok's audio gaplessly
  and shows captions. When the agent searches, `window` gets an `atlas:hits` CustomEvent
  (`detail: {hits, region, args, tool}`) so the map can highlight results while Grok talks.
  Captions also go out as `atlas:voice` (`{role, text, final}`).
  Mic upload pauses while the agent speaks (so laptop speakers do not trigger it). Click the
  button once while it talks to interrupt, again to stop.
- `window.AtlasVoice.ask("what's on my agenda today")` sends a typed turn into the live session.

Mic access needs `localhost` or https.

## 3. Realtime proxy (`WS /ws/voice`)

`atlas/voice/realtime.py` connects to `wss://api.x.ai/v1/realtime?model=grok-voice-latest` and
sends `session.update` with voice `eve`, server VAD, `audio/pcm` at 24 kHz in and out, short
spoken-answer instructions (cite sender and date), and the tools from `atlas.agent.tools`
(`TOOL_SCHEMAS`, `run_tool`). If that module is missing it falls back to a keyword search stub
(`search_region`, `get_email`) over `data/mail.sqlite`.

Browser to server: binary frames are raw PCM16 mic audio; JSON frames may be Realtime client
events (`input_audio_buffer.*`, `response.create`, `response.cancel`,
`conversation.item.create`) or `{"type":"atlas.text","text":...}`.
Server to browser: every Grok event, plus `atlas.ready`, `atlas.hits` and `atlas.error`.
Tool calls are run server side on `response.function_call_arguments.done`; after
`response.done` all outputs go back as `function_call_output` items, then `response.create`.

Env: `GROK_VOICE` (default `eve`), `GROK_VOICE_MODEL` (default `grok-voice-latest`).

## 4. System-wide dictation (`dictate/dictate.py`, macOS)

```bash
uv sync --extra dictate
uv run python dictate/dictate.py                 # hold Right Option, speak, release
uv run python dictate/dictate.py --key cmd_r --style email -v
uv run python dictate/dictate.py --no-paste      # print only
```

Hold the key to record (Tink sound), release to send (Pop sound). It posts to the running server's
`/api/dictate` and, if the server is down, calls Grok directly using `.env`. The cleaned text is
pasted with Cmd+V into whatever app has focus and your previous clipboard is restored.

macOS permissions (the first run will prompt, or add them by hand in System Settings, Privacy
and Security, for the terminal app you run it from, e.g. Terminal, iTerm or Ghostty):

- **Microphone**: needed to record.
- **Accessibility**: needed to send Cmd+V.
- **Input Monitoring**: needed for the global hotkey listener.

Restart the terminal after granting them. If the key does nothing, Input Monitoring is missing;
if text is transcribed but not pasted, Accessibility is missing.
