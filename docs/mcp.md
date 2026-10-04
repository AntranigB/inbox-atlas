# Inbox Atlas for agents (MCP + HTTP)

Agents usually burn context two ways: they paste whole documents, or they grep and open file after file.
Inbox Atlas hands them the minimal context instead: the region a question maps to, the few emails or
notes inside it, and only the sentences that matter, under a token budget. When the region is empty
it says so (`answerable: false`, about 30 tokens) so the agent stops searching.

## Setup

```bash
uv sync
uv run python -m atlas.ingest.load_fixture                    # or your real mail (see README)
OBSIDIAN_VAULT=~/Brain uv run python -m atlas.ingest.obsidian  # notes chunked by heading, stored locally
uv run python -m atlas.index.build --encoder base --no-grok    # local embeddings, both sources, one map
```

Vault text is embedded locally and never sent to Grok at ingest. At query time only the question goes
to Grok (for facets); the packed excerpts go wherever your agent sends its context.
`.obsidian/`, `archive/`, `backup/` and `templates/` are skipped.

## Tools

| tool | what it returns |
|---|---|
| `atlas_context(question, budget_tokens=800, sources=["gmail","obsidian"])` | `answerable`, `confidence`, a ready-to-paste `context` string, `items` (source, uri, title, date, excerpt, z), `tokens`, `tokens_saved_vs_naive` |
| `atlas_search(query, k=10, sources)` | ranked hits with uri, title, date, a 160 char snippet and z |
| `atlas_related(topic, sources)` | calibrated yes/no with up to 3 example uris |
| `atlas_get(uri, max_tokens=1500)` | one email (`gmail:<id>`), one note section (`path.md#anchor`) or a whole note (`path.md`), capped |

Same thing over HTTP from the main server (`uv run python server.py`):

```bash
curl -s localhost:8765/api/context -H 'content-type: application/json' \
  -d '{"question": "when is my flight?", "budget_tokens": 300}'
curl -s localhost:8765/api/context/get -H 'content-type: application/json' -d '{"uri": "projects/Japan trip.md"}'
```

## Run the MCP server

```bash
uv run --directory /path/to/inbox-atlas python mcp/server.py                  # stdio
uv run --directory /path/to/inbox-atlas python mcp/server.py --http --port 8767  # streamable HTTP at /mcp
```

Set `ATLAS_DATA` to point at another data directory and `ATLAS_ENCODER` to pick the index (default `base`).

### Claude Code

```bash
claude mcp add inbox-atlas -- uv run --directory /path/to/inbox-atlas python mcp/server.py
# or, with the HTTP server running:
claude mcp add --transport http inbox-atlas http://127.0.0.1:8767/mcp
```

A line for the vault's `CLAUDE.md` that makes the planning system use it:
"Before opening notes, call `atlas_context` with the question. If it says answerable=false, do not grep."

### Hermes Agent

`~/.hermes/config.yaml` (Hermes reads MCP servers from `mcp_servers`):

```yaml
mcp_servers:
  inbox-atlas:
    command: uv
    args: ["run", "--directory", "/path/to/inbox-atlas", "python", "mcp/server.py"]
    env:
      OBSIDIAN_VAULT: /Users/you/Brain
    timeout: 60
```

or, with the HTTP server: `inbox-atlas: {url: "http://127.0.0.1:8767/mcp"}`.

### OpenClaw

`~/.openclaw/openclaw.json` (servers live under `mcp.servers`):

```json
{
  "mcp": {
    "servers": {
      "inbox-atlas": {
        "command": "uv",
        "args": ["run", "--directory", "/path/to/inbox-atlas", "python", "mcp/server.py"]
      }
    }
  }
}
```

Remote form: `{"transport": "streamable-http", "url": "http://127.0.0.1:8767/mcp"}`. Check it with
`openclaw mcp probe inbox-atlas`.

The Hermes and OpenClaw snippets follow their published docs (Hermes `mcp_servers` in config.yaml,
OpenClaw `mcp.servers` in openclaw.json) but were not run against a live install here. Any other MCP
client works with the generic stdio command above.

## How the pack is built

1. Grok turns the question into facets (cached in `data/cache/expand.json`).
2. The facets become a region; every email and note section gets a hub-corrected z score.
3. Calibrated verdict: top z >= 3.0 and raw >= floor. If not, return `answerable: false` and stop.
4. Members of the region are ranked by z. The budget is the recall knob: <= 400 tokens keeps the
   core (z >= 0.6 x best) and 2 sentences per item, <= 1000 keeps z >= 0.35 x best and 4 sentences,
   above that the whole region plus near members (z >= 2) and 6 sentences.
5. Every sentence of every kept item is scored with the same encoder against the same region; items
   keep their best sentences (in reading order, `...` for gaps). Bodies under 70 tokens stay whole.
6. Greedy packing under the budget, dropping weakest sentences first. Tokens are tiktoken cl100k_base.

Numbers: [eval/token_results.md](../eval/token_results.md).
