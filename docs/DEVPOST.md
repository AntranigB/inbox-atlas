# Inbox Atlas: DevPost draft

Paste each section into the matching DevPost field. Every number below comes from a real run in
this repo (eval/token_results.md, eval/results.md, TRAINING.md).

## Tagline

Your agents read 4,000 tokens to find one email. Inbox Atlas hands them the 250 that matter, or tells them nothing is there.

## Inspiration

This year's theme is Navigation. Jon Kleinberg's research is about navigating networks and
information spaces, and the information space we navigate worst is our own: email and notes. We
use LLM agents (Claude Code, Hermes, OpenClaw) on top of a personal second brain in Obsidian, and
watched them burn context the same way every time: paste whole documents, or grep and open file
after file, and keep searching when the answer is not there. Keyword search also misses meaning:
search "coding competition" and Gmail returns a HackerRank job assessment because it contains the
word "coding", while the Codeforces round, the ICPC tryout and the DevPost receipt never show up.
We wanted retrieval that moves through meaning like a map, spends as few tokens as possible, and
knows when to stop.

## What it does

- **Context for agents.** One MCP tool, `atlas_context(question, budget_tokens)`, for Claude Code,
  Hermes, OpenClaw or any MCP client. It returns the few emails or note sections that answer the
  question, cut down to the sentences that matter, under a token budget. If the topic is not
  there it returns a calibrated "nothing here" so the agent stops looking.
- **Search by region, not by string.** Grok writes facets for a question in the words documents
  actually use ("Codeforces round", "ICPC regional", "Devpost submission") plus things to exclude
  ("job coding interview"), and the search scores every email and note against that region.
- **"Is this topic here at all?"** A YES or NO with a confidence. "Coding competition": YES,
  8 emails. "Yacht maintenance": NO.
- **Gmail and Obsidian.** Gmail is exported read-only; an Obsidian vault is chunked by heading and
  embedded locally.
- **For people too.** A web app with an inbox map, a realtime Grok voice agent, Whisperflow-style
  dictation anywhere on a Mac, and an iMessage agent on Photon: text it "what do I have today?",
  and it texts you a morning brief and alerts when new mail lands in a topic you are watching.
- **Runs locally.** One command (`atlas up`) runs the API, Postgres with pgvector on the Tiger
  Data stack, Grok, Photon messaging and Gmail sync, reachable from a phone behind a password.

## How we built it

- **atlas-embed:** a frozen bge-base-en-v1.5 backbone with LoRA adapters (1.18M trainable out of
  110M parameters), Matryoshka heads at 768, 256 and 64 dims, InfoNCE on mined hard negatives,
  and listwise distillation from a frozen bge-reranker-v2-m3 cross-encoder that also filters
  false negatives.
- **Training data:** 224,867 deduplicated Enron emails with 426 folder topics (63 held out
  entirely), 19,980 emails labeled by Grok with abstract topics and search queries, and 491k
  teacher scores. Trained overnight on one RTX 4070 SUPER.
- **Region search and calibration:** facet max plus centroid minus a negative penalty, then a hub
  correction: each document's score is compared against random sets of probe topics with the
  same facet count, so newsletters that sit near everything stop counting.
- **atlas-region:** a Set Transformer that learns the region from the facet set, with temperature
  and offset calibration. We kept it as a research component because it did not beat the
  heuristic on held-out topics (below).
- **Context packer:** sentence-level extraction inside region hits, greedy packing under a
  token budget, abstention when the yes/no check says no. Shipped as an MCP server and as
  `POST /api/context`.
- **Storage:** Postgres with pgvector (HNSW) and a TimescaleDB hypertable for query logs, in the
  Tiger Data container; switching to Tiger Cloud is one connection string. Raw .eml files are
  kept and every vector points back to its email.
- **Grok everywhere:** facet expansion, the tool-calling chat agent with session history, speech
  to text (grok-voice-transcribe-2.0) with a cleanup pass for dictation, and the realtime voice
  API through a server-side proxy that runs the same tools.
- **Photon:** a Spectrum sidecar for two-way iMessage, plus a scheduler for briefs and alerts.
- **Engineering:** ten branches built in parallel against a written interface contract, each
  merged through an integration branch only after the full suite passed (86 Python tests, a
  Postgres test, 19 Node tests).

## Challenges we ran into

- **Hubness.** Our first relatedness check said 13 of 24 demo emails were "about" every topic.
  Matching the probe sets to the query's facet count fixed it.
- **Calibration.** A topic covers about 0.1 percent of an inbox, but training sets are near
  balanced. A temperature alone left an ECE of 0.42; adding an offset fit at the real base rate
  fixed it.
- **A learned region that did not learn.** The Set Transformer region never beat its own
  starting point on held-out topics in five variants. We report it instead of hiding it.
- **A late regression.** Once the learned region checkpoint existed, search silently switched to
  it while the hub correction still assumed the heuristic scale, and the yes/no check fell from
  20 of 20 to 10 of 20. Comparing both encoders side by side caught it before submission.
- **Keeping a mailbox safe.** "Take all my mail, delete nothing" became a rule enforced in code
  and in tests: read-only IMAP, BODY.PEEK so nothing is marked read, raw messages kept.

## Accomplishments that we're proud of

- **Tokens:** the 800-token pack averages 249 tokens at 75% accuracy: the accuracy of chunk RAG at a
  third of the tokens, and above grep-style retrieval (67% at 4,445 tokens) at an eighteenth.
- **Knowing when to stop:** on 14 absent topics the pack spends 31 tokens and abstains on 13;
  the baselines hand over 742 to 4,039 tokens of unrelated text.
- **A real second brain:** on a real 419-note Obsidian vault (embedded locally, nothing sent to
  Grok), the pack contains the target note for 12 of 12 questions at 625 tokens, against 7,400
  tokens for top-10 whole notes.
- **Search:** region search puts 7 of the top 8 on topic for "coding competition" (cosine 6,
  keyword 0), and the yes/no check is right on 20 of 20 demo topics.
- **The trained encoder:** on unseen Enron emails it beats the frozen base model on vague queries
  (nDCG@10 0.692 vs 0.635) and subject to body (0.408 vs 0.342), and it lifts region membership
  AUROC from 0.819 to 0.843.

## What we learned

- Whole-document RAG is still the most accurate (92%), and we say so. The win is accuracy per
  token and abstention, not raw accuracy.
- A query works better as a set with a boundary than as a single point, and calibration is what
  turns a ranking into an answer.
- Our encoder did not generalize to unseen folder-name queries, and our learned region did not
  beat a heuristic. Measuring honestly showed us where the real gains were.

## What's next

- Run the zero-shot test that is already built: a model trained only on 2001 corporate Enron
  mail, scored on a real 2026 Gmail inbox.
- Fine-tune per user (stage C is built), and let agent feedback become training pairs.
- More sources behind the same tool: calendar, docs, chats.

## Built with

python, pytorch, peft, sentence-transformers, bge, fastapi, postgresql, pgvector, timescaledb,
tiger-data, sqlite, mcp, numpy, umap, hdbscan, grok, xai, photon, spectrum, imessage, node.js,
javascript, html, css, gmail, imap, obsidian, cloudflare, enron

## Prize tracks to select

Big Red (Navigation), Software, People's Choice, Photon (Agents in iMessage). SpaceX requires the
project to be built with Cursor; select it only if that is true for your team.
