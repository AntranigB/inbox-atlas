# Inbox Atlas: interface contract

Every branch builds against this file. If you need to change an interface, change it here in
your PR and say so in the PR description. Do not silently diverge.

## Branches and ownership

| Branch | Owns (only edit these, plus tests for them) |
|---|---|
| `training` | `train/`, `atlas/model/`, `TRAINING.md`, `configs/` |
| `ingest` | `atlas/ingest/`, `atlas/store.py`, `atlas/index/`, `atlas/calendar.py` |
| `search-agent` | `atlas/search/`, `atlas/agent/`, `atlas/api/search.py`, `web/index.html`, `web/app.js`, `web/style.css`, `eval/` |
| `voice` | `atlas/voice/`, `atlas/api/voice.py`, `web/voice.js`, `dictate/` |
| `imessage` | `imessage/` (Node sidecar), `atlas/api/messaging.py`, `atlas/notify.py` |

Shared files (`server.py`, `pyproject.toml`, `CONTRACT.md`, `README.md`) may get small additive
edits. Keep them additive so merges stay clean.

## Runtime layout

- Python 3.12, managed by `uv`. `uv sync` installs the app. `uv sync --extra train` adds torch/peft.
- API server: `uv run python server.py` on **port 8765**.
- iMessage sidecar (Node): `imessage/`, **port 8766**.
- All data lives under `data/` (gitignored). Config comes from `.env` (see `.env.example`).

## Storage: `data/mail.sqlite` (owned by `ingest`, schema in `atlas/store.py`)

```sql
emails(
  id TEXT PRIMARY KEY,        -- Gmail X-GM-MSGID, or sha1 of Message-ID
  thread_id TEXT,             -- Gmail X-GM-THRID, or normalized subject hash
  from_addr TEXT, from_name TEXT, to_addrs TEXT,
  date INTEGER,               -- unix seconds UTC
  subject TEXT, body TEXT,    -- body is cleaned plain text (no HTML, quotes, signatures)
  snippet TEXT,               -- first ~200 chars of body
  labels TEXT,                -- JSON list
  source TEXT                 -- 'gmail' | 'fixture' | 'enron'
)
emails_fts USING fts5(subject, body, from_name, content='emails', content_rowid='rowid')
events(id TEXT PRIMARY KEY, title TEXT, start INTEGER, end INTEGER, location TEXT, source TEXT)
watches(id TEXT PRIMARY KEY, name TEXT, positive TEXT, negative TEXT, created INTEGER, last_checked INTEGER)
```

`atlas/store.py` exposes: `connect(path=None)`, `init_db(conn)`, `upsert_emails(conn, rows)`,
`get_email(conn, id)`, `all_ids(conn)`, `texts_for_embedding(conn) -> list[(id, text)]`,
`fts_search(conn, query, k) -> list[(id, bm25)]`, `upsert_events`, `events_between(conn, t0, t1)`.

The embedding text for an email is `f"{subject}\n{from_name}\n{body[:2000]}"`.

## Encoders (owned by `training`, file `atlas/model/encoder.py`)

```python
class Encoder:
    name: str; dim: int
    def encode_docs(self, texts: list[str]) -> np.ndarray      # (n, dim) float32, L2-normalized
    def encode_queries(self, texts: list[str]) -> np.ndarray   # adds the query instruction prefix
def load_encoder(spec: str | None = None) -> Encoder
```

`spec` values: `"hash"` (deterministic hashing encoder for tests, no download),
`"base"` (frozen `BAAI/bge-base-en-v1.5`), or a path to a trained checkpoint dir
(`models/atlas-embed/`, which holds the LoRA adapter, `atlas.json` and optional `region.pt`).
The default comes from env `ATLAS_ENCODER`, else `"base"`. Optional env `ATLAS_DIM` truncates
Matryoshka dims (768/256/64).

## Index (owned by `ingest`, `atlas/index/`)

`data/index/<encoder_name>/`: `emb.npy` (float16, n x dim), `ids.json`, `meta.json`
(`{encoder, dim, built_at}`), `hub.npz` (`mu`, `sigma` per email from 300 probe topics),
`map.json` (`{points: [{id, x, y, cluster}], clusters: [{id, label, size}]}`).
`atlas/index/__init__.py` exposes `load_index(encoder_name) -> Index` with `.ids`, `.E`
(float32), `.mu`, `.sigma`, `.map`, and `build_index(encoder)`.

## Regions (interface owned by `search-agent`, learned impl owned by `training`)

```python
class Region:                                   # atlas/search/region.py
    def score(self, E: np.ndarray) -> np.ndarray          # raw membership score per row
    def prob(self, E: np.ndarray) -> np.ndarray | None    # calibrated P(member), if available
    def describe(self) -> dict                            # anchors, facet info, for UI/Grok
def build_region(pos_vecs, neg_vecs, kind="heuristic"|"learned") -> Region
```

`atlas/model/region_encoder.py` provides `LearnedRegionModel.load(path)` with
`.build(pos_vecs, neg_vecs) -> Region` (same interface). `search` falls back to heuristic
when no `region.pt` exists.

## Grok tools (owned by `search-agent`, `atlas/agent/tools.py`)

Every caller (web chat, voice agent, iMessage) uses the same tool list:

| Tool | Args | Returns |
|---|---|---|
| `search_region` | `positive[] negative[] after? before? from? k=10` | `{region:{size, facet_hits, nearest_clusters}, hits:[{id, from, date, subject, snippet, z, prob}]}` |
| `is_related` | `topic positive[]? ` | `{related: bool, confidence, count, max_z, examples[]}` |
| `get_email` | `id` | `{id, from, date, subject, body(<=2000 chars)}` |
| `list_clusters` | | `[{id, label, size}]` |
| `todays_agenda` | `date? (YYYY-MM-DD, default today local)` | `{events:[...], emails:[...]}` (calendar events + emails mentioning that date) |
| `add_watch` | `name positive[] negative[]` | `{id}`. Standing region; new mail inside it triggers a text |
| `list_watches` | | `[{id, name}]` |

Python entry point: `atlas.agent.grok.ask(text, channel, history=None) -> {reply, hits, region}`.
`channel` is `web`, `voice` or `imessage` (iMessage replies are short plain text, no markdown).
`atlas.agent.tools` also exports `TOOL_SCHEMAS` (OpenAI function format) and `run_tool(name, args)`
for the voice agent, plus `delete_watch(id_or_name)`. `atlas.agent.grok.check_watches(new_ids)`
returns a list of dicts `{watch_name, watch, id, from, subject, date, snippet, z}`.
The web UI exposes `window.atlas = {search(q), ask(text), applyResult(res), setQuery(q)}` and fires a
`atlas:result` DOM event, so voice.js can drive the search box and map.

## HTTP API (FastAPI app in `server.py`, routers in `atlas/api/`)

| Route | Owner | Body / Response |
|---|---|---|
| `POST /api/ask` | search-agent | `{text, channel, history?}` -> `{reply, hits, region}` |
| `POST /api/search` | search-agent | `{query, mode: keyword/embed/region, k}` -> `{hits, region, facets}` |
| `GET /api/related?topic=` | search-agent | `is_related` output |
| `GET /api/map` | search-agent | `map.json` |
| `GET /api/agenda?date=` | search-agent | `todays_agenda` output |
| `GET/POST /api/watches` | search-agent | `DELETE /api/watches/{id_or_name}` removes one |
| `GET /api/email/{id}` | search-agent | full email row (UI expands a result) |
| `GET /api/encoders` | search-agent | `{default, available[]}` index dirs, drives the Base / Tuned toggle |
| `POST /api/stt` | voice | multipart `audio` -> `{text}` (raw Grok STT) |
| `POST /api/dictate` | voice | multipart `audio` -> `{raw, text}` (STT + Grok cleanup) |
| `WS /ws/voice` | voice | proxy to Grok realtime, with tools executed server side |
| `POST /api/notify` | imessage | `{text}` -> sends an iMessage to the owner via the sidecar |
| `GET /api/health` | shared | `{ok, encoder, n_emails}` |

The sidecar exposes `POST http://localhost:8766/send {text, to?}`. Inbound iMessages go to
`POST /api/ask` with `channel="imessage"`.

## Grok

- Chat: `POST https://api.x.ai/v1/chat/completions`, model from env `GROK_MODEL`
  (default `grok-4.20-0309-non-reasoning`), OpenAI-style `tools`.
- STT: `POST https://api.x.ai/v1/stt`, multipart `model=grok-voice-transcribe-2.0`, `file`.
  Verified working on our key.
- Realtime: `wss://api.x.ai/v1/realtime?model=grok-voice-latest`, OpenAI Realtime protocol.
- There is NO xAI embeddings access on our key. Embeddings are always local.

## Tests

`uv run pytest -q`. Tests must pass offline with `ATLAS_ENCODER=hash` and no API keys (mock
HTTP calls). Shared fixture: `tests/fixtures/mailbox.jsonl` (rows match the `emails` table).
Tests that need a real model or network are marked `@pytest.mark.slow` or `@pytest.mark.live`.

The fixture rows carry an extra `topic` field (hackathon, contest, money, housing, travel, class,
jobs, shopping, newsletter, family, fitness, research) used only as ground truth in tests.
Note the hard case: "coding competition" should match hackathon and contest rows but rank the
HackerRank job assessment lower.

## Gmail export rules (ingest branch)

- IMAP is opened **read-only** (`select(mailbox, readonly=True)`). Never STORE, COPY, MOVE,
  EXPUNGE or delete. Nothing in the user's mailbox may change.
- Export **all** mail (`[Gmail]/All Mail`), resumable, saving each raw message to
  `data/raw/gmail/<X-GM-MSGID>.eml` before parsing. Raw files are never deleted.
- The parsed rows go into `data/mail.sqlite`. The personal training dataset is built from it
  by `train/build_personal.py` (training branch) into `data/datasets/personal/`.
