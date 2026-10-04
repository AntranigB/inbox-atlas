# Ingest, index, calendar

All data lands in `data/` (gitignored). Set `ATLAS_DATA=/some/dir` to put it elsewhere.

## Demo mode (no credentials)

```
uv run python -m atlas.ingest.load_fixture          # 24 fixture emails -> data/mail.sqlite
uv run python -m atlas.index.build --encoder base    # embeddings, hub stats, map
```

## Real Gmail (read-only)

1. Turn on 2-Step Verification, create an App Password at https://myaccount.google.com/apppasswords,
   and make sure IMAP is on in Gmail settings.
2. In `.env`: `GMAIL_USER=you@gmail.com` and `GMAIL_APP_PASSWORD=...` (spaces are fine).
3. Export and index:

```
uv run python -m atlas.ingest.gmail_export             # newest GMAIL_MAX_MESSAGES (0 = all)
uv run python -m atlas.ingest.gmail_export --max 0     # all mail
uv run python -m atlas.ingest.gmail_export --parse-only --reparse   # re-clean raw files, no IMAP
uv run python -m atlas.index.build --encoder base      # or --encoder models/atlas-embed
```

Safety: the exporter opens `[Gmail]/All Mail` with `readonly=True`, fetches with `BODY.PEEK[]` (the
read flag never changes) and only ever sends LOGIN, LIST, SELECT (read-only), UID SEARCH and UID FETCH.
`SafeIMAP` in `atlas/ingest/imap.py` raises on STORE, COPY, MOVE, EXPUNGE, DELETE or APPEND, and
`tests/test_ingest.py` checks this against a fake server. Nothing is ever deleted, in Gmail or on disk.

Every message is saved raw to `data/raw/gmail/<X-GM-MSGID>.eml` before parsing. Rerunning skips files
already on disk, so an interrupted export just resumes. `_meta.jsonl` holds uid, thread id and labels;
`_state.json` holds the last UID seen (used by sync).

## Incremental sync

```
uv run python -m atlas.ingest.sync               # once
uv run python -m atlas.ingest.sync --loop 120    # poll every 2 minutes
```

In code: `from atlas.ingest.sync import sync_new; ids = sync_new()` returns the new email ids
(`[]` when there are no credentials or nothing new). It saves raw files, upserts rows, embeds the new
mail with `ATLAS_ENCODER` and appends it to that encoder's index (hub stats and map position included).
Safe to call from a background thread. If the full export never ran it starts watching from now.

## Calendar

Put the Google Calendar secret iCal URL in `.env` as `GCAL_ICS_URL`, then:

```
uv run python -m atlas.calendar     # next 14 days, recurring events expanded, into the events table
```

`atlas.calendar.agenda(conn, "2026-10-04")` returns `{date, events, emails}`, where `emails` are mails
from the last 30 days that mention that day ("Oct 4", "10/4", "Sunday", "tomorrow", ...).

## Index layout

`data/index/<encoder>/`: `emb.npy` (float16), `ids.json`, `meta.json`, `hub.npz` (`mu`, `sigma` over
300 probe topics from `atlas/index/probes.txt`, plus the probe vectors `P`), `map.json`.
Cluster labels are TF-IDF terms of the subjects, or Grok labels when `XAI_API_KEY` is set
(cached in `data/index/grok_labels.json`, skip with `--no-grok`). Under 200 emails the map uses
PCA plus agglomerative clustering instead of UMAP plus HDBSCAN.

```python
from atlas.index import load_index
idx = load_index("base")      # idx.ids, idx.E (float32), idx.mu, idx.sigma, idx.map, idx.add(ids, vecs)
```

## Tests

```
ATLAS_ENCODER=hash uv run pytest -q     # offline
uv run pytest -q -m slow                # bge-base download, UMAP path
```
