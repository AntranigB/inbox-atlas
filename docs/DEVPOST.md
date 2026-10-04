# Inbox Atlas: DevPost draft

Paste each section into the matching DevPost field. Numbers marked PENDING come from the main
training run; fill them from `runs/eval2/metrics.md` before submitting or delete the line.

## Tagline

Navigate your inbox by meaning. Ask by typing, talking or texting, and get a yes or no on whether a topic is even in there.

## Inspiration

This year's theme is Navigation, and Jon Kleinberg's research is about navigating networks and
information spaces. The space we navigate most often and most badly is our own inbox. Gmail
search is string matching: search "coding competition" and you get a HackerRank job assessment
because it contains the word "coding", while the Codeforces round, the ICPC tryout and the
DevPost receipt never show up because none of them say "coding competition". We wanted search
that moves through meaning the way you move through a map, and that can tell you when there is
nothing there.

## What it does

- **Search by region, not by string.** Type, say or text a question. Grok writes a set of facets
  in the words emails actually use ("Codeforces round", "ICPC regional", "Devpost submission")
  plus things to exclude ("job coding interview"). Our model turns that set into a region of
  embedding space and ranks every email by how far inside it is.
- **"Is this topic in my inbox at all?"** A calibrated YES or NO with a confidence, instead of a
  list of weak matches. "Coding competition": YES, 8 emails, z 10. "Yacht maintenance": NO.
- **An inbox map.** Every email is a point; the region for your question lights up on it.
- **Talk to it.** A realtime Grok voice agent searches your mail while it answers out loud.
- **Dictate anywhere.** Whisperflow-style hold-to-talk: Grok speech to text plus a cleanup pass
  that drops the "um"s and applies corrections ("by five, no, six" becomes "by six"), in the
  search box or any app on your Mac.
- **Text it, and it texts you.** An iMessage agent on Photon Spectrum: "what do I have today?"
  answers from your calendar and mail. "watch internships" creates a standing region, and it
  texts you when a new email lands inside it. It also sends a morning brief.

## How we built it

- **Ingest:** Gmail over IMAP, strictly read-only (mailbox opened read-only, bodies fetched with
  BODY.PEEK so nothing is even marked read, and a guard that raises on any mutating command).
  Every raw message is saved before parsing. Cleaning strips HTML, quoted replies and signatures.
  SQLite with FTS5 for keyword search.
- **atlas-embed:** a frozen bge-base-en-v1.5 backbone with LoRA adapters (1.18M trainable out of
  110M parameters), trained with InfoNCE on mined hard negatives, Matryoshka heads at 768, 256 and
  64 dims, and listwise KL distillation from a frozen bge-reranker-v2-m3 cross-encoder teacher,
  which also filters false negatives.
- **atlas-region:** a Set Transformer (2 ISAB layers, PMA with 4 seeds) that reads the facet set
  Grok writes and outputs a region: 4 anchors, temperatures and a boundary, with temperature and
  offset calibration so its output is a real probability.
- **Training data:** 224,867 deduplicated Enron emails with 426 folder topics, 63 held out
  entirely, plus 19,980 emails labeled by Grok with abstract topics and search queries, plus 491k
  teacher scores. Trained on one RTX 4070 SUPER over the night.
- **Hub correction:** newsletters sit close to everything, so each email's score is compared
  against random sets of probe topics of the same size. That is what makes the YES or NO
  trustworthy.
- **Agent:** Grok with tools (search_region, is_related, get_email, todays_agenda, add_watch). The
  search tool returns region geometry (size, hits per facet, nearest clusters), so Grok steers
  the region: drop dead facets, narrow, widen.
- **Voice:** Grok STT (grok-voice-transcribe-2.0) and the Grok realtime voice API through a
  server-side proxy that runs the same tools.
- **iMessage:** a Node sidecar on Photon Spectrum, plus a scheduler for the morning brief and
  watch alerts.
- **Engineering:** five parallel branches against a written interface contract, each merged
  through an integration branch only after the full test suite passed (53 Python tests offline,
  3 slow model tests, 15 Node tests).

## Challenges we ran into

- **Hubness.** Our first relatedness check said 13 of 24 demo emails were "about" every topic,
  because taking the max over 8 facets inflates scores. Comparing each email against random probe
  sets with the same facet count fixed it: absent topics now sit near z 1 and present ones at 5 to 10.
- **Calibration.** Training sets are sampled near balanced, but a real topic covers about 0.1
  percent of an inbox. A temperature alone left an ECE of 0.42; adding an offset fit at the
  natural base rate fixed it.
- **Noisy labels.** Enron folder names include codes and people's names, so topic retrieval is hard
  for every method, BM25 included. We report it anyway rather than hide it.
- **Keeping a mailbox safe.** "Take all my mail, delete nothing" became a hard rule enforced in
  code and in tests, not a promise.

## Accomplishments that we're proud of

- On the demo inbox, region search puts 7 of the top 8 results on topic for "coding competition",
  against 6 for plain embeddings (which rank the job assessment first) and 0 for keyword search.
- The YES or NO check got 20 of 20 present and absent topics right on the demo inbox.
- A trained model with frozen backbone, frozen teacher and a learned set-to-region head, trained on
  a quarter million real emails in one night on a single consumer GPU.
- One set of tools behind four interfaces: web, voice, dictation and iMessage.
- PENDING: atlas-embed vs base on held-out Enron topics, and the zero-shot transfer number on a
  real 2026 Gmail inbox.

## What we learned

- A query is better treated as a set with a boundary than as a single point.
- Calibration is the difference between a ranking and an answer.
- Being honest about prior art made the design better: semantic email search, query expansion,
  LoRA, Matryoshka and distillation all exist. What we added is the calibrated set-to-region
  model, the yes or no, and an agent that steers regions like a map.

## What's next

- Fine-tune per user on their own mail (stage C is built) and let clicks become new training pairs.
- Regions that span mail, calendar, docs and chats.
- On-device embedding, so nothing leaves the laptop except the snippets Grok needs.

## Built with

python, pytorch, peft, sentence-transformers, bge, fastapi, sqlite, umap, hdbscan, numpy, grok,
xai, photon, spectrum, imessage, node.js, javascript, html, css, gmail, imap, enron

## Prize tracks to select

Big Red (Navigation), Software, People's Choice, Photon (Agents in iMessage), SpaceX (Grok Voice
API is used; check the track's Cursor requirement before selecting it).
