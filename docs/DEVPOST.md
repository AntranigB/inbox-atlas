# Inbox Atlas: DevPost draft

Paste each section into the matching DevPost field. Every number comes from a measured run in this
repo (eval/token_results.md, eval/results.md, TRAINING.md).

## Tagline

Your agents read 4,000 tokens to find one email. Inbox Atlas hands them the 250 that matter, or tells them nothing is there.

## Inspiration

This year's theme is Navigation, and the information space we navigate worst is our own: email
and notes. We run LLM agents (Cursor, Claude Code, Hermes, OpenClaw) on top of a personal second
brain in Obsidian, and watched them waste context the same way every time: paste whole
documents, or grep and open file after file, and keep searching when the answer is not there.
Keyword search also misses meaning. Search "coding competition" and Gmail returns a HackerRank
job assessment because it contains the word "coding", while the Codeforces round, the ICPC tryout
and the DevPost receipt never show up. We wanted retrieval that moves through meaning like a map,
spends as few tokens as possible, and knows when to stop.

## What it does

- **Context for agents.** One MCP tool, `atlas_context(question, budget_tokens)`, for Cursor,
  Claude Code, Hermes, OpenClaw or any MCP client. It returns the few emails or note sections that
  answer the question, cut down to the sentences that matter, under a token budget. If the topic
  is not there it returns a calibrated "nothing here" so the agent stops looking.
- **Folders first, then notes.** For deep markdown trees (Obsidian vaults, agent memory folders),
  `atlas_points_of_interest` embeds the folder tree, finds where a question lives, and drills
  down: folder, note, exact line in about 210 tokens.
- **Search by meaning.** Grok writes facets for a question in the words documents actually use
  ("Codeforces round", "ICPC regional") plus exclusions ("job coding interview"), and every email
  and note is scored against that region.
- **"Is this here at all?"** A YES or NO with a confidence. "Coding competition": YES. "Yacht
  maintenance": NO.
- **For people.** A web app with an inbox map, a realtime Grok voice agent, Whisperflow-style
  dictation anywhere on a Mac, and a two-way iMessage agent on Photon: text it "what do I have
  tomorrow?", and it texts you a morning brief and alerts when new mail lands in a topic you are
  watching.
- **One command.** `atlas up` runs the API, Tiger Cloud storage, Grok, Photon messaging and Gmail
  sync, reachable from a phone behind a password.

## How we built it

- **atlas-embed:** a frozen bge-base-en-v1.5 backbone with LoRA adapters (1.18M trainable out of
  110M parameters), Matryoshka heads at 768, 256 and 64 dims, InfoNCE on mined hard negatives, and
  listwise distillation from a frozen bge-reranker-v2-m3 cross-encoder that also filters false
  negatives. Trained on 224,867 deduplicated Enron emails with 426 folder topics, 19,980 emails
  labeled by Grok and 491k teacher scores, overnight on one RTX 4070 SUPER. The adapter ships in
  the repo (17 MB).
- **Region search with calibrated abstention:** facet max plus centroid minus an exclusion
  penalty, then each document's score is compared against random topic sets with the same number
  of facets, so a match has to stand out from chance. That gives the yes or no and the "stop
  searching" signal.
- **atlas-region:** a Set Transformer that turns an LLM's facet list into a calibrated region.
- **Context packer:** sentence-level extraction inside the region, greedy packing under a token
  budget, abstention when nothing is there. Served as an MCP server and `POST /api/context`.
- **Folder tree:** folder vectors over the markdown tree, ranked by the region scores of the notes
  inside, with a `within` drill-down and related folders.
- **Tiger Cloud:** emails, embeddings (pgvector HNSW), chat history and a TimescaleDB hypertable
  that logs every query with its latency and tokens. The search engine reads from it live.
- **Grok:** facet expansion, the tool-calling chat agent with session memory, speech to text
  (grok-voice-transcribe-2.0) with a cleanup pass for dictation, and the realtime voice API
  through a server-side proxy that runs the same tools.
- **Photon:** a Spectrum sidecar for two-way iMessage, plus a scheduler for briefs and alerts.
- **Cursor:** in our own testing outside this repo, the Cursor CLI was the most token-efficient
  harness at routing compared with Claude Code, Codex and Grok, so it is the agent we demo the MCP
  tool in.
- **Engineering:** branches built in parallel against a written interface contract, each merged
  through an integration branch only after the full suite passed (92 Python tests, 19 Node tests).

## Challenges we ran into

- **Hubness.** Newsletters sit close to every topic, so our first relatedness check said 13 of 24
  demo emails were about everything. Comparing each email against random probe sets with the same
  facet count fixed it.
- **Calibration at a real base rate.** A topic covers about 0.1 percent of an inbox while training
  sets are near balanced; fitting a temperature and an offset at the real base rate fixed the
  probabilities.
- **Keeping a mailbox safe.** "Take all my mail, delete nothing" became a rule enforced in code
  and tests: read-only IMAP, BODY.PEEK so nothing is marked read, raw messages kept.
- **Making Grok consistent over iMessage.** We separated in-region answers from borderline near
  misses in the tool output so the agent never presents a job test as a coding competition.

## Accomplishments that we're proud of

- **Keyword search found nothing for 7 of 12** meaning-based questions; Atlas found the right
  emails for all 12 (Recall@10 0.867 vs 0.175, about 5x).
- **18x fewer tokens than grep** (249 vs 4,445 per question) and **a third of chunk RAG** at the
  same accuracy (249 vs 740 tokens).
- **31 tokens when the topic is not there**, abstaining on 13 of 14, where chunk RAG hands over 742
  and grep 1,858.
- **A real 419-note second brain:** the right note 12 of 12 times in 625 tokens, against 10,907 for
  grep.
- **Folder navigation:** folder, note and exact line in 210 tokens; absent topics in 23.
- **A trained encoder** that beats the frozen base model on vague queries (nDCG@10 0.692 vs 0.635)
  and lifts region membership AUROC from 0.819 to 0.843.
- **It's real:** live on Tiger Cloud, texting over Photon iMessage, and running over a real
  29,322-message Gmail account, read-only.

## What we learned

- A question works better as a set with a boundary than as a single point.
- Calibration is what turns a ranking into an answer, and an answer is what saves tokens.
- Agents spend most of their retrieval budget on searches that fail; making "nothing here" cheap
  matters as much as making "here it is" accurate.

## What's next

- Fine-tune per user on their own mail (the pipeline is built).
- More sources behind the same tool: calendar, docs, chat logs.
- Agent feedback becoming training pairs.

## Built with

python, pytorch, peft, sentence-transformers, bge, fastapi, postgresql, pgvector, timescaledb,
tiger-data, sqlite, mcp, numpy, umap, hdbscan, grok, xai, photon, spectrum, imessage, node.js,
javascript, html, css, gmail, imap, obsidian, cursor, cloudflare, enron

## Team

Avinash Senthil, Antranig Baghdassarian, Chelsea Lin, Emily Han.

## Prize tracks to select

Big Red (Navigation), Software, People's Choice, Photon (Agents in iMessage). SpaceX (Grok Voice
plus Cursor) if the team built with Cursor.
