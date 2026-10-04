# Demo runbook

Everything below runs on the demo inbox (24 fictional student emails) and the synthetic vault
(fictional nested markdown). Nothing personal is on screen.

## 10 minutes before judging

```bash
cd ~/Documents/bigred/inbox-atlas
uv run atlas status        # api up, postgres up; if not: uv run atlas up
SIDECAR=http://127.0.0.1:8766 MCP=1 scripts/demo_check.sh   # PASS/FAIL per demo step, about 30 s
```

`scripts/demo_check.sh [base url]` reads ATLAS_TOKEN from the env or `.env`, hits every endpoint
below and exits with the number of FAILs. It also warms Grok's expansion cache
(`data/cache/expand.json`) for the demo queries, so the live searches are instant and use the
facets the check just verified. If you rebuild data or clear that cache, run it again.

Open four windows:

1. **Browser:** http://localhost:8765 (enter the ATLAS_TOKEN from .env once; the page remembers it).
2. **Terminal A, iMessage:** the mock phone. Start it with a clean thread:
   ```bash
   pkill -f "node src/index.js" ; rm -f data/imessage_history.mock.json
   cd imessage && set -a && source ../.env && set +a && npm run mock
   ```
   With real Photon keys (SPECTRUM_PROJECT_ID, SPECTRUM_PROJECT_SECRET, OWNER_PHONE in .env),
   `uv run atlas up` starts the real iMessage sidecar instead; text the Photon line once first.
3. **Terminal B, agent:** Cursor CLI in the repo, with the project MCP server approved once:
   `cursor-agent mcp list` should show `inbox-atlas: ready` (it is in ~/.cursor/mcp.json), then `cursor-agent`.
   Claude Code works the same way (`claude`; added with `claude mcp add --scope user`).
4. **Phone (optional):** the Cloudflare link from `grep trycloudflare data/logs/cloudflared.log`,
   same password. If cloudflared stopped: `cloudflared tunnel --url http://localhost:8765`.

## Slides

Slides: http://localhost:8765/deck/ (P for presenter view, N for notes, F fullscreen). Arrows, space or click to move; `#n` in the URL jumps to slide n. Offline copy: open `web/deck/index.html` from disk, or `docs/deck.pdf`.

## The demo (about 3 minutes)

**1. Search by meaning (browser, 40 s).** Click the `coding competition` chip in the **try** row
(or type it).
- Toggle **Keyword**: the HackerRank job assessment is the only hit and none of the contest emails appear. "This is Gmail today."
- Toggle **Region**: Codeforces, LeetCode, ICPC, Mom's "after your hackathon", DevPost, Kaggle,
  BigRed (8 in region). The job assessment sits below the "outside the region: near misses"
  line at z 2.9. The facet chips show what Grok wrote, including the `not` chips that push it out.
- Badge: **RELATED: YES**, and next to it **427 tokens for an agent** (the `/api/context` pack).
  Then click `yacht maintenance`: **RELATED: NO**, "Nothing in your inbox about this", **28 tokens:
  agent stops here**. "It knows when nothing is there."
- Point at the map: the region glows where those emails live; stars are the facets.
- Optional: **Base / Tuned** swaps to our trained encoder (same 8, strongest z 8.3).
- While Grok writes facets the results show a skeleton and the map breathes; about 1 s uncached,
  instant once `demo_check.sh` has warmed the cache.

Backup screenshots: `docs/img/demo/` (keyword, region, related NO, tuned, phone, loading).

**2. Text it (Terminal A, 40 s).** Type each once, in a fresh thread:
- `what do I have tomorrow?` (expected: "Lab meeting Mon 10am codec ablation. Priya Natarajan Sep 30: ...")
- `did anyone ask me for money?` (expected: "Yes. Venmo Wed Sep 30: Alex requested $42.50 for pizza + uber friday.")
- `any emails about coding competitions?` (expected: Codeforces, LeetCode, Cornell ACM ICPC; no HackerRank)
- `watch internships` then `watches` (a standing region that texts you when new mail lands in it)

Each answer came back the same 3 out of 3 times from a fresh thread (`docs/img/demo/imessage_transcript.txt`).
Do not promise Jordan's $60 concert tickets email: it is a short casual "send whenever" note that
sits below the region for every phrasing we tried (found 1 of 3 times with `do I owe anyone money?`).
Clean up after the demo with `stop internships`.

**3. Talk or dictate (browser, 30 s).** Click **Talk to inbox** and ask "do I have anything about
the hackathon?" (first audio about 3 s, full answer about 7 s: Mom, BigRed Hacks demo day and
acceptance, the DevPost receipt). Or hold Space on the mic and ramble a query with an "um, no
wait"; the cleaned text lands in the search box in about 1 s
("So like, find me emails about, um, no wait, the coding competition" became
"find emails about the coding competition").

**4. The agent and the tokens (Terminal B, 50 s).** Ask the agent:
- "Use inbox-atlas: what pulse widths did I use for the arm servos?"
  Expected: `projects/robotics/arm/`, `Servo calibration.md#pulse-widths`, "Elbow servo: 610 to
  2380 microseconds", about 210 tokens.
- "Use inbox-atlas: did I book a yacht charter?" Expected: nothing here, about 23 tokens, stop.
- The 210 and 23 are `atlas_points_of_interest`; `atlas_context` returns 347 and 28 for the same
  two questions. Cursor (`cursor-agent -p ... --mode ask --approve-mcps --trust`) calls
  `atlas_context` on its own and answers in about 4 s. To see the raw tool output without an
  agent: `uv run python scripts/mcp_check.py` (saved in `docs/img/demo/mcp_output.txt`).
- Then show `assets/token_efficiency.png`: same accuracy as chunk RAG at a third of the tokens;
  31 tokens on absent topics vs 742 to 4,039.

**5. The model (10 s).** `assets/architecture.png` and the Enron table in the README.

## If something breaks

- **Browser shows "token required":** paste ATLAS_TOKEN from .env into the prompt.
- **Grok slow or down:** the region search and "is it here" still run locally. Use the keyword
  vs region toggle.
- **iMessage answer lists the HackerRank job assessment:** should not happen any more: the agent
  now passes a plain `topic` and the server reuses the same cached Grok expansion as the web search
  (with its `not` facets). If it does, text `reset` and ask once. The job assessment is at z 2.9
  against a 3.0 line, so a fresh uncached expansion could tip it in; run `demo_check.sh` (step 1b
  checks exactly this) so the cache holds a good expansion.
- **API process died mid search (Metal "command encoder" crash in data/logs/api.log):** fixed on
  demo-qa (all model calls run on one thread). Without the fix, two searches at once can abort
  the server; the daemon restarts it in a few seconds.
- **MCP server not loaded in the agent:** call it over HTTP instead:
  `curl -s -X POST localhost:8765/api/context/folders -H "Authorization: Bearer $ATLAS_TOKEN" -H 'content-type: application/json' -d '{"question":"what pulse widths did I use for the arm servos?"}'`
- **Everything down:** `uv run atlas down && uv run atlas up`, then `uv run atlas status`.
