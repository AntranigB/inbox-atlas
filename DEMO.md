# Demo runbook

Everything below runs on the demo inbox (24 fictional student emails) and the synthetic vault
(fictional nested markdown). Nothing personal is on screen.

## 10 minutes before judging

```bash
cd ~/Documents/bigred/inbox-atlas
uv run atlas status        # api up, postgres up; if not: uv run atlas up
```

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
   `cursor-agent mcp list` (approve `inbox-atlas` if it says pending), then `cursor-agent`.
   Claude Code works the same way (`claude`, approve the project MCP server).
4. **Phone (optional):** the Cloudflare link from `grep trycloudflare data/logs/cloudflared.log`,
   same password. If cloudflared stopped: `cloudflared tunnel --url http://localhost:8765`.

## The demo (about 3 minutes)

**1. Search by meaning (browser, 40 s).** Type `coding competition`.
- Toggle **Keyword**: the HackerRank job assessment ranks first and none of the contest emails appear. "This is Gmail today."
- Toggle **Region**: Codeforces, LeetCode, ICPC, DevPost, Kaggle, BigRed. The facet chips show
  what Grok wrote, including the negatives that push the job assessment out.
- Badge: **RELATED: YES**. Then type `yacht maintenance`: **NO**. "It knows when nothing is there."
- Point at the map: the region glows where those emails live.

**2. Text it (Terminal A, 40 s).** Type each once, in a fresh thread:
- `what do I have tomorrow?` (expected: Monday 10am lab meeting with Priya)
- `did anyone ask me for money?` (expected: Venmo from Alex, $42.50; it sometimes also finds Jordan's concert tickets email, sometimes only on a follow-up like `anyone else?`)
- `watch internships` then `watches` (a standing region that texts you when new mail lands in it)

**3. Talk or dictate (browser, 30 s).** Click **Talk to inbox** and ask "do I have anything about
the hackathon?" Or hold Space on the mic and ramble a query with an "um, no wait"; the cleaned text
lands in the search box.

**4. The agent and the tokens (Terminal B, 50 s).** Ask the agent:
- "Use inbox-atlas: what pulse widths did I use for the arm servos?"
  Expected: `projects/robotics/arm/`, `Servo calibration.md#pulse-widths`, "Elbow servo: 610 to
  2380 microseconds", about 210 tokens.
- "Use inbox-atlas: did I book a yacht charter?" Expected: nothing here, about 23 tokens, stop.
- Then show `assets/token_efficiency.png`: same accuracy as chunk RAG at a third of the tokens;
  31 tokens on absent topics vs 742 to 4,039.

**5. The model (10 s).** `assets/architecture.png` and the Enron table in the README.

## If something breaks

- **Browser shows "token required":** paste ATLAS_TOKEN from .env into the prompt.
- **Grok slow or down:** the region search and "is it here" still run locally. Use the keyword
  vs region toggle.
- **iMessage answer lists the HackerRank job assessment:** Grok's facets vary between runs. Clear
  the thread (step 2 of setup) and ask once. This is why the search tool separates "borderline"
  near misses from answers.
- **MCP server not loaded in the agent:** call it over HTTP instead:
  `curl -s -X POST localhost:8765/api/context/folders -H "Authorization: Bearer $ATLAS_TOKEN" -H 'content-type: application/json' -d '{"question":"what pulse widths did I use for the arm servos?"}'`
- **Everything down:** `uv run atlas down && uv run atlas up`, then `uv run atlas status`.
