# Inbox Atlas, semantic navigation for your Gmail

> Say or type "coding competition" and get the BigRed registration email, the Codeforces
> round reminder and the DevPost receipt, even though none of them contain the words
> "coding competition". You also get a calibrated **yes/no: does this topic exist in my inbox?**

Theme fit (Navigation): this is *information* navigation. Kleinberg's own research area is
navigation in networks and information spaces. Your inbox becomes a map. A query becomes a
**region** on that map, not a string, and Grok steers that region the way you would steer
on a map: widen, narrow, exclude.

Deadline: **DevPost Sun 8:30 AM**. Everything below is cut to fit about 9.5 hours of build time.

---

## 0. Verified facts (checked live tonight, not assumed)

| Thing | Status |
|---|---|
| Grok chat `grok-4.20-0309-non-reasoning` via `/v1/chat/completions` | works (used in nomad) |
| Grok STT `POST https://api.x.ai/v1/stt`, model `grok-voice-transcribe-2.0` | **works on our key**: returned text + word timestamps |
| Grok Voice Agent realtime `wss://api.x.ai/v1/realtime?model=grok-voice-latest` | documented, OpenAI-Realtime-compatible, supports `tools`. Not tested yet. |
| Grok TTS `POST /v1/tts` | documented, untested |
| xAI embeddings `/v1/embeddings` | **NOT available on our team**, so embed locally |
| Grok Imagine (image/video) | in our model list (optional garnish) |

---

## 1. Honest prior art (say this before a judge does)

Semantic and AI email search already exists: **Gmail's Gemini side panel, Shortwave AI search,
Superhuman AI search**, plus many "RAG over email" repos. Overall this is *useful execution*,
not a new mechanism. These parts are actually ours:

1. **Query = region, not point.** Grok expands the query into a topic list (positive facets +
   explicit negatives). We compress that list into a region (centroid + facet set + boundary)
   and score emails against the region.
2. **Calibrated "is this topic related?"** Instead of a raw cosine number, the answer is a
   per-inbox, hub-corrected z-score with a yes/no and a confidence. Related idea: CSLS hubness
   correction (Lample et al. 2018). Say so.
3. **Agentic navigation loop.** Grok gets region statistics back (size, sample subjects,
   nearest clusters), not just hits, and refines the region the way you re-center a map.
4. **Voice everywhere.** Whisperflow-style push-to-talk dictation (Grok STT) plus a
   talk-to-your-inbox Grok Voice agent that calls the same tools.
5. **Local-first.** Email bodies are embedded on your laptop. Only top-hit snippets go to Grok.

---

## 2. Architecture

```
            ┌──────────── voice ─────────────┐
 mic ──► push-to-talk ──► Grok STT ──► cleanup (Grok fast) ──► text box / any app (Whisperflow mode)
 mic ──► Grok Voice Agent (realtime WS) ──► function calls ─┐
                                                            │
 text query ──► Grok orchestrator (chat + tools) ───────────┤
                  │ 1. expand_topics (Grok itself)          │
                  ▼                                         ▼
          ┌──────────────── FastAPI (Python) ────────────────────────┐
          │ /tools/search_region  /tools/is_related  /tools/get_email│
          │ /api/map  /api/stt  /api/realtime-token                  │
          └───────┬──────────────────────────────────────────────────┘
                  ▼
   Region engine (numpy):  E (n×d, L2-normed)  @  topic vectors
     + hubness stats (μ_i, σ_i per email)  + SQLite FTS5 (BM25) for hybrid/baseline
                  ▲
   Ingest: Gmail IMAP ──► clean (strip HTML, quotes, signatures) ──► chunk ──► local embedder
                                                                   (bge-small/base, MPS)
   Storage: SQLite (emails, chunks, FTS5) + emb.npy + umap2d.npy
```

### Stack choices (decided)

| Layer | Pick | Why / fallback |
|---|---|---|
| Ingest | **Gmail IMAP + App Password** (`imaplib`, `X-GM-THRID`, `X-GM-LABELS`) | 5 min setup, no Google Cloud OAuth consent screen. Stretch: Gmail API OAuth for a "Sign in with Google" button. |
| Parse | `email` stdlib + `selectolax`/BeautifulSoup + `talon`-style quote stripping (regex is fine) | Remove `On … wrote:` blocks and signatures, or every reply looks the same |
| Embedder | **Custom: `inbox-atlas-embed`**, fine-tuned from `BAAI/bge-small-en-v1.5` (384d) on your own inbox (see §3.5). Base bge is the baseline and fallback. | sentence-transformers is already in `nomad/ml/.venv`. Matryoshka-trained, so it serves 384/128/64 dims. |
| Vector store | **numpy brute force** (`E @ q`) | 5k×768 matmul under 5 ms, exact. pgvector buys nothing at this scale, and Tiger isn't a prize track. |
| Keyword | SQLite **FTS5** BM25 | Hybrid fusion + the "Gmail-style baseline" in the eval |
| Map | **UMAP** → 2D (`umap-learn`), precomputed; HDBSCAN clusters labeled by Grok | The "navigation" visual |
| API | **FastAPI** + uvicorn | Embeddings are Python, so one process |
| LLM | `grok-4.20-0309-non-reasoning`, tool calling | Fast, cheap, already proven |
| Voice | Grok STT (dictation) + Grok Voice Agent realtime (conversation) | Satisfies SpaceX "Grok Voice API" |
| Frontend | Single page, vanilla JS + `<canvas>` scatter (or `regl-scatterplot` from CDN) | No build step |
| Dictation client | Tiny Python menubar/hotkey script (`sounddevice` + `pynput` + paste) | Whisperflow-style "hold key, talk, text appears in any app" |
| Built in | **Cursor**, all of it | SpaceX track requirement. Screenshot your Cursor sessions for DevPost. |

---

## 3.5 Custom embedding model (the trainable core)

**Decision: fine-tune, don't train from scratch.** A from-scratch encoder can't learn English
in 9 hours on a laptop. Contrastive fine-tuning of a small pretrained encoder on *your* mail
takes about 15-40 min on MPS and measurably beats the base model on domain retrieval.
Prior art, stated honestly: GPL (Wang et al. 2022), InPars, and Promptagator all do
"LLM-generated queries → fine-tune a retriever". Our twist is the **topic-region objective**
and the **compression** (Matryoshka dims), which echoes NeuroZip.

### Training data (all self-supervised from your inbox plus Grok)
| Pair type | How | ~Count |
|---|---|---|
| **Topic → email** (the important one) | Grok reads each email, emits 2-3 *abstract* topic labels that avoid the email's own words ("coding competition", "money owed to a friend") | 6-9k |
| **Search query → email** | Grok writes 2 queries a person would type to find it later (one vague, one specific) | 6k |
| Subject ↔ body | free | 3k |
| Reply ↔ parent (same `X-GM-THRID`) | free | ~1-2k |
| **Hard negatives** | for each pair, base-bge top-20 minus same-thread emails minus anything Grok labeled with the same topic | 1 per pair |

Batch Grok at 10 emails/call, 8 calls in parallel: about 300 calls, 15-20 min. Send only
subject + first 800 chars. **Privacy note:** this step does send snippets to xAI. Say so
in the README.

### Objective
- `MultipleNegativesRankingLoss` (InfoNCE, in-batch + 1 mined hard negative), wrapped in
  **`MatryoshkaLoss` dims=[384,128,64]**. Use `CachedMultipleNegativesRankingLoss` to get an
  effective batch of 256 on MPS.
- Query side keeps the bge query prefix. Topic labels and queries are both "query" side.
- 1-2 epochs, lr 2e-5, warmup 10%. Save the best checkpoint by held-out nDCG@10.
- Optional adapter-only variant (freeze encoder, train linear `W` 384×384 initialized to
  identity) as a 2-minute ablation.

### Splits (so the eval isn't baked into the training)
- **80/20 split by thread** (not by email), so no reply chain leaks across.
- **Held-out topics:** cluster all Grok topic labels and drop ~15% of topic *clusters*
  entirely from training. Novelty probe: does the model improve on **topics it never saw as
  labels**, on **emails it never saw**? If yes, it learned your inbox's geometry, not a lookup
  table. That's the claim the training can't bake in.
- The 12 handwritten queries in §6 are never shown to Grok's generator.

### Why it helps the region search specifically
Base bge clusters by *surface form* (all newsletters together, all receipts together).
Topic→email training pulls emails together by *what they're about*, so a Grok topic list
compresses into a tighter region with a cleaner boundary. Measure it: average within-region
cosine and the "related?" yes/no accuracy, base vs tuned.

### Files
```
train/gen_pairs.py     # Grok topic labels + queries → data/pairs.jsonl (resumable, cached per email id)
train/mine_negs.py     # base-bge hard negatives, false-negative filter
train/finetune.py      # sentence-transformers trainer, Matryoshka + CachedMNRL, eval each 200 steps
train/README.md        # data card + loss + splits + results table
models/inbox-atlas-embed/   # output (gitignored)
```

---

## 3. The core: query → topic list → region

### 3a. Grok expands the query (structured output)

Prompt Grok to return JSON:
```json
{
  "positive": ["hackathon", "ICPC / programming contest", "Codeforces round", "LeetCode weekly contest",
               "DevPost submission", "Kaggle competition", "competitive programming team"],
  "negative": ["job application coding assessment", "online course homework"],
  "filters":  {"after": null, "before": null, "from": null},
  "intent":   "find | ask_related | summarize"
}
```
Rule in the prompt: facets are **short phrases in the vocabulary emails actually use**
(event names, platforms, sender-style phrases), not synonyms of the query.

### 3b. Compress the topic list to a region

```python
P = embed_queries([query] + positive)     # (k, d), L2-normed
N = embed_queries(negative)               # (m, d)
c = normalize(P.mean(0))                  # region center
# region = { center c, facets P, anti-facets N, boundary tau }

s_facet = (E @ P.T).max(1)                # matches ANY facet (breadth)
s_core  =  E @ c                          # near the center (coherence)
s_neg   = (E @ N.T).max(1) if m else 0
raw     = 0.6*s_facet + 0.4*s_core - 0.5*np.maximum(0, s_neg - s_facet)
```

### 3c. Hub correction + calibrated "is it related?"

Newsletters and long promos sit close to *everything* (hubness). Fix it once at index time:
- Embed about 300 random "probe" topics (have Grok generate 300 diverse unrelated topic
  phrases once, cache them). `R = E @ probes.T` gives per-email `mu_i, sigma_i`.
- `z_i = (raw_i - mu_i) / sigma_i`. Rank by `z` (tie-break with `raw`).
- **Related?** = `z_top ≥ 3.0 and raw_top ≥ floor`. Report `"YES, 14 emails in region,
  strongest z=5.2"` or `"NO, nothing beyond noise (max z=1.4)"`. This works per inbox
  with no labels.
- Hybrid: Reciprocal Rank Fusion of region rank + BM25 rank (k=60). Expose a toggle
  so the demo can show keyword-only vs region side by side.

### 3d. What the tool returns to Grok (so it can navigate)

```json
{"region": {"size": 23, "tau_z": 3.0, "facet_hits": {"hackathon": 11, "Codeforces round": 6, "Kaggle": 0},
            "nearest_clusters": ["Hackathons & DevPost", "Cornell CS clubs"]},
 "hits": [{"id": "...", "from": "...", "date": "...", "subject": "...", "snippet": "...", "z": 5.2}]}
```
Grok sees that `Kaggle` found 0, so it drops it. If size is 400 it narrows; if 0 it widens.
**Cap at 2 refinement rounds.** That loop is the "navigation" story.

### 3e. Grok tool schema (shared by chat, voice agent, and the iMessage stretch)

```
search_region(positive[], negative[], after?, before?, from?, k=10)
is_related(topic, positive[]?, scope_email_id?)  -> {related, z, count, examples[]}
get_email(id)                                    -> full cleaned body (truncated 2k chars)
list_clusters()                                  -> map clusters with labels and sizes
```

---

## 4. Voice

**A. Whisperflow-style dictation (do first; cheap and verified)**
- Browser: hold `Space` on the mic button → `MediaRecorder` (webm/opus) → `POST /api/stt`
  → server forwards to `https://api.x.ai/v1/stt` (`model=grok-voice-transcribe-2.0`).
- Cleanup pass: Grok fast model, prompt "remove fillers/false starts, keep meaning, no
  additions". This is what makes it Whisperflow and not raw STT.
- System-wide stretch: `dictate.py` holds a global hotkey (`pynput`), records with
  `sounddevice`, transcribes plus cleans up, then pastes into whatever app has focus. It
  works in Gmail compose, Slack, etc. That's a good live demo moment.

**B. Talk to your inbox (Grok Voice Agent, realtime)**
- Server mints the session (keep the API key server-side; proxy the WS through FastAPI if
  ephemeral tokens aren't available).
- `session.update`: `voice: "eve"`, `turn_detection: {type: "server_vad"}`, `instructions`
  (short spoken answers, cite sender + date), `tools`: the 4 functions in §3e.
- On `response.function_call_arguments.done` → run the tool locally → send
  `conversation.item.create` with the output → `response.create`.
- The UI highlights the region on the map live while it talks.

---

## 5. UI (one page)

- Search bar + mic. Under it: **Related: YES/NO** badge with z and count.
- Left: result list (sender, date, subject, snippet, a z bar, and which facet matched).
- Right: **inbox map** (UMAP dots colored by cluster). The query region glows and facet
  points are drawn as small stars, so you watch the region move as Grok refines it.
- Toggle: `Keyword | Embedding | Region` to show why the region wins.

---

## 6. Eval (the rigor slide; about 45 min, worth it)

- 12 queries where wording differs from the emails ("coding competition", "money I owe
  someone", "anything about housing next year", "travel plans", …).
- Pool the top 20 from BM25, plain cosine, and region. Grok-as-judge labels relevance
  (spot-check 20 labels by hand and report the agreement).
- Report **Recall@10 and nDCG@10**: BM25 vs single-vector cosine vs region vs region+hub-z.
- "Related?" check: 10 topics you *know* are in the inbox and 10 you know aren't (e.g.
  "yacht maintenance"). Report accuracy of the yes/no.
- Ablation: region without negatives and without hub correction. One table.
- **Model table (the headline):** base bge-small vs **inbox-atlas-embed** (384 / 128 / 64
  dims) vs linear-adapter-only, on seen vs **held-out topics** × held-out emails. Recall@10,
  nDCG@10, related? accuracy. Compressing to 64 dims with little loss is a second headline.

---

## 7. Repo layout

```
inbox-atlas/
  ingest/imap_fetch.py      # IMAP → data/mail.sqlite (id, thread, from, to, date, subject, body, labels)
  ingest/clean.py           # html→text, strip quotes/signatures, chunk
  index/embed.py            # inbox-atlas-embed (or base bge) → data/emb.npy (+ chunk→email map), max-pool chunks per email
  index/probes.py           # 300 probe topics → mu/sigma per email
  index/map.py              # UMAP + HDBSCAN + Grok cluster labels → data/map.json
  core/region.py            # §3b-3c (pure functions, unit-tested)
  core/hybrid.py            # FTS5 BM25 + RRF
  agent/grok.py             # expansion prompt + tool loop (max 2 refinements)
  agent/tools.py            # §3e tool schemas, shared by chat and voice
  server.py                 # FastAPI: /tools/*, /api/search, /api/map, /api/stt, /ws/voice
  web/index.html app.js style.css
  dictate.py                # system-wide push-to-talk (stretch)
  eval/queries.json run_eval.py
  .env  (XAI_API_KEY, GMAIL_USER, GMAIL_APP_PASSWORD)
```

---

## 8. Build order (time-boxed; it's Sat ~10:45 PM)

| When | Do | Done when |
|---|---|---|
| 10:45-11:30 | Gmail App Password (needs 2FA) → `imap_fetch.py` pulls the last ~3-5k messages | `select count(*)` > 3000 |
| 11:30-12:15 | `clean.py` + `embed.py` (base bge-small) + FTS5. **Kick off `gen_pairs.py` in the background.** | base-model search works from CLI; pairs file growing |
| 12:15-1:30 | `region.py` (expansion, facets, negatives, probes, z, related?) + tests, while pairs finish | "yacht maintenance" → NO, "coding competition" → YES |
| 1:30-1:45 | `mine_negs.py`, then **start `finetune.py` in the background** (bge-small, ~20-30 min) | training loss falling, eval nDCG logged |
| 1:45-3:00 | `server.py` + Grok tool loop with region-stat feedback | chat answer cites the right emails |
| 3:00 | Re-embed the inbox with the tuned model; swap via `EMBED_MODEL` env var | tuned beats base on held-out nDCG, or keep base and report honestly |
| 3:00-4:30 | Web UI + UMAP map + region glow + mode toggle (+ `Base / Tuned` toggle) | demoable without voice |
| 4:30-5:30 | Voice A (STT + cleanup), then Voice B (realtime agent with tools) | spoken question → spoken answer + map moves |
| 5:30-6:30 | Eval script → model table + ablations | numbers in README |
| 6:30-7:45 | DevPost write-up, 2-min video, Cursor screenshots, prior-art paragraph | submitted draft |
| 7:45-8:30 | Buffer. **Submit by 8:15.** |, |

**Cut list if behind (in order):** Photon iMessage → `dictate.py` system-wide → HDBSCAN
labels → Matryoshka dims (plain MNRL) → realtime agent (keep STT, which still counts as
Grok Voice) → map (keep the list).
**Never cut:** custom model + its held-out eval, region search, related? yes/no, Grok
expansion, STT. If fine-tuning doesn't beat base by 3 AM, ship base and report the
negative result honestly. Don't fake the table.

**Stretch if ahead:** (a) Photon/Spectrum: text the agent "did I get anything about
internships?" (same tools, separate prize). (b) Fine-tune `bge-base` (768d) too and add a
column. (c) Continual learning: clicks on results become new positive pairs.

---

## 9. Demo script (2 min)

1. "Gmail search is string matching." Type `coding competition` in Keyword mode: junk.
2. Switch to Region mode. Grok's facets appear, the region glows on the map, and the
   BigRed / Codeforces / DevPost emails show up. Badge: **YES, z=5.2**.
3. "Yacht maintenance": **NO**, nothing beyond noise. That's the calibrated part.
4. Hold the mic: "anything about money I owe someone?" Grok Voice answers out loud,
   citing a Venmo request, and the map moves.
5. Whisperflow moment: in Gmail compose, hold the hotkey, ramble a reply, and clean text appears.
6. Model table: BM25 X → base bge Y → **inbox-atlas-embed Z on held-out topics**, plus
   "64 dims keeps N% of it". This is the model *we trained on my inbox overnight*.

## 10. Track targeting

- **SpaceX "Make it Legendary":** Cursor + Grok Voice (STT + realtime) + optional Grok
  Imagine. Risk: the blurb wants "real space data". Mitigation: one slide noting the region
  engine is corpus-agnostic, and optionally a second index over NASA APOD or arXiv astro-ph
  (`mission-compass/PLAN.md` already has the APOD fetch plan). Only if time allows.
- **Software track, People's Choice:** everyone has an inbox they can't search.
- **Photon:** only if the stretch lands.
- **Big Red:** "navigating information space" framing plus the map visual.

## 11. Privacy and safety notes

- App Password lives only in `.env` (gitignored). Read-only IMAP (`readonly=True` on select).
- Never send full bodies to Grok in bulk. Send only top-k snippets per tool call.
- For the public demo, consider a demo inbox, or blur senders on the projector.
