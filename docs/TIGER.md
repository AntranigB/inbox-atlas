# Inbox Atlas on Tiger Data

Inbox Atlas can keep its embeddings, emails and Grok chat history in Postgres with pgvector, the
Tiger Data stack (Tiger Cloud / TimescaleDB). The same code runs against a local container and
against Tiger Cloud; only `DATABASE_URL` changes. Without `DATABASE_URL` everything stays on
SQLite + numpy, exactly as before.

## What is stored

| table | contents |
|---|---|
| `emails` | the SQLite columns plus `raw_path` (`data/raw/gmail/<id>.eml`, the original message on disk) and a generated, GIN-indexed `tsv` for keyword search |
| `email_vectors` | `(email_id -> emails, encoder, dim, embedding vector(768))`, one row per email per encoder, HNSW index with `vector_cosine_ops` |
| `chunks`, `chunk_vectors` | long documents split into spans. A chunk points at an email (`doc_id` = email id) or a note (`source='obsidian'`, `doc_id` = note path, `uri` = `path#heading`), with `span_start` / `span_end` offsets |
| `chat_sessions`, `chat_messages` | every Grok turn from web, voice and iMessage: user text, each tool call with its args (`jsonb`) and the email ids it returned (`hits`), the assistant reply |
| `watches`, `events` | mirrors of the SQLite tables |
| `query_log` | per-question metrics (channel, query, facets, region size, tokens returned, latency). A TimescaleDB hypertable when the extension is present |

Matryoshka-truncated encoders (`ATLAS_DIM=256` or `64`) and the 256-d hash test encoder share the
`vector(768)` column: vectors are zero-padded. Both sides are L2-normalized and the padding is
zero, so cosine distance in Postgres equals cosine in the truncated space. `dim` records the
real size and `vectors_matrix` slices it back off.

## Local run

Needs Docker (Docker Desktop or colima). The image `timescale/timescaledb-ha:pg17` ships
pgvector, pgvectorscale and pgai.

```bash
docker-compose up -d                       # or: docker compose up -d. Port 127.0.0.1:5433, volume inbox-atlas_atlas-tiger
echo 'DATABASE_URL=postgresql://postgres:atlas@localhost:5433/atlas' >> .env
uv sync --extra dev
uv run python -m atlas.db.migrate          # data/mail.sqlite + data/index/*/emb.npy -> Postgres
uv run python server.py                    # ATLAS_DB defaults to pg when DATABASE_URL is set
```

Password comes from `ATLAS_PG_PASSWORD` in `.env` (default `atlas`), set it before the first
`up`. `ATLAS_DB=sqlite` forces the old path even when `DATABASE_URL` is set; `ATLAS_DB=pg` with an
unreachable database logs a warning and falls back to SQLite so the app still starts.

After the migration, `atlas.index.build` and incremental sync (`Index.add`) also write new
emails and vectors to Postgres whenever the pg backend is active.

Inspect:

```bash
docker exec -it atlas-tiger psql -U postgres -d atlas
select role, left(content, 60), tool_name, tool_args, hits from chat_messages order by id desc limit 20;
select encoder, dim, count(*) from email_vectors group by 1, 2;
```

## Tiger Cloud

1. Sign up at [console.cloud.tigerdata.com](https://console.cloud.tigerdata.com) (free tier).
2. Create a service (Postgres with TimescaleDB; the vector extension is available on every service).
3. Copy the connection string from the service overview. It looks like
   `postgres://tsdbadmin:<password>@<id>.<project>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require`.
4. Put it in `.env` as `DATABASE_URL=...` and run `uv run python -m atlas.db.migrate`.

Nothing else changes. `schema.sql` is idempotent and runs on every connect, creating the
`vector` extension and (when allowed) turning `query_log` into a hypertable.

## Python API

```python
from atlas.db.backend import get_backend
db = get_backend()                        # PgStore or SqliteBackend
db.knn(qvec, k=10, encoder="base", filters={"after": "2026-09-01", "from": "devpost"})
db.fts("hackathon", k=20)                 # [(id, rank)]
db.vectors_matrix("base")                 # (ids, E float32), cached; the region scorer uses it
db.append_message(session_id, "user", "any hackathons?", channel="web")
db.history(session_id, n=12)

from atlas.db import chat                 # thin helpers for other callers
chat.add_message(sid, "user", text, channel="imessage"); chat.get_history(sid); chat.list_sessions()
```

`atlas.agent.grok.ask(text, channel, history=None, session_id=None, user_handle=None, persist=True)`
saves each turn and returns `session_id`. `POST /api/ask` accepts `session_id` and `user_handle`;
when `history` is omitted and `session_id` is set, history is loaded from the database.

## Differences from the SQLite path

- Region and embed search give the same results: region scoring loads the full vector matrix
  from `email_vectors` once (cached) and runs the same numpy code; embed mode uses
  `ORDER BY embedding <=> q` through HNSW (approximate at scale, exact on small tables where the
  planner does a sequential scan).
- Keyword mode ranks with `ts_rank_cd` over an English `tsvector` (stemmed, subject weighted
  above sender above body) instead of SQLite FTS5 BM25. Same matched set, different order and
  score scale.
- The 2D map and hub statistics still come from `data/index/<encoder>/` files.

## Tests

`ATLAS_ENCODER=hash uv run pytest -q` stays offline: the autouse fixture in `tests/conftest.py`
forces SQLite. Tests marked `@pytest.mark.pg` run against `DATABASE_URL` inside a throwaway
`atlas_test` schema and are skipped when it is unset.
