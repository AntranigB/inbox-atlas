# Inbox Atlas: pitch

## One line

Your agents read 4,000 tokens to find one email. Inbox Atlas hands them the 250 that matter, or tells them nothing is there.

## The 2 minute pitch (say this)

**Problem (20 s).** Everyone has an inbox and a pile of notes they cannot search. Gmail search
is string matching: type "coding competition" and you get a HackerRank job assessment because it
says "coding", and you miss the Codeforces round, the ICPC tryout and the DevPost receipt. And
now our AI agents search the same way, only more expensively: Claude Code, Hermes, OpenClaw and
every second brain setup either paste whole documents into context or grep and open file after
file, and when the answer is not there they keep looking and keep burning tokens.

**What we built (30 s).** Inbox Atlas is one retrieval layer for both people and LLMs. A
question becomes a region of meaning, not a keyword. Grok writes the facets ("Codeforces round",
"ICPC regional") and the things to exclude ("job coding interview"). We score every email and
note against that region with our own encoder, trained overnight on a quarter million real
emails on one gaming GPU, with a frozen backbone, LoRA adapters and a frozen teacher model. Then
we hand back only the sentences that answer the question, under a token budget, or a calibrated
"nothing here" so the agent stops searching.

**For people (20 s).** Search on the web, talk to your inbox with Grok voice, dictate
Whisperflow style, or just text it over iMessage with Photon: "what do I have tomorrow?" It also
texts you first, with a morning brief and alerts when new mail lands in a topic you are watching.

**For agents (30 s).** The same engine is an MCP server, so Claude Code, Cursor, Hermes and
OpenClaw can call it directly. It generalizes past email to any second brain: a deep tree of
markdown gets embedded at the folder level too, so an agent first finds the points of interest,
which folders and notes a question lives in, then drills down, instead of querying every file.
Ask "what pulse widths did I use for the arm servos?" and it returns the
`projects/robotics/arm/` folder, the exact note, and the one line with the numbers, in 210
tokens.

**Numbers (20 s).** Our context pack gets the accuracy of chunk RAG at a third of the tokens,
and beats grep-style retrieval with an eighteenth of the tokens. When the topic is not there it
spends 31 tokens and stops, where the usual approaches hand the agent 700 to 4,000 tokens of
noise. On a real 419-note personal vault it found the right note 12 of 12 times in 625 tokens,
against 7,400 for reading whole notes.

**Close (10 s).** Navigation used to mean maps. For the next hundred years it also means
navigating what we and our agents know. Inbox Atlas is a map of your information that people and
LLMs can both use, at a fraction of the tokens.

## Numbers you can quote (all measured in this repo)

| claim | number | source |
|---|---|---|
| Context pack vs chunk RAG | 75% accuracy at 249 tokens vs 75% at 740 | eval/token_results.md |
| Context pack vs grep-style top 10 | 75% at 249 tokens vs 67% at 4,445 | eval/token_results.md |
| Absent topics | 31 tokens, abstains on 13 of 14; baselines 742 to 4,039 tokens | eval/token_results.md |
| Real vault (419 notes, local only) | target note in context 12 of 12 at 625 tokens vs 7,400 | eval/token_results.md |
| Folder navigation | right folder, note and line in 210 tokens; absent topic 23 tokens | README, Navigating nested markdown |
| Search, demo inbox | region 7 of top 8 on topic, cosine 6, keyword 0 | assets/search_modes.png |
| "Is this here?" | 20 of 20 demo topics correct | assets/related_calibration.png |
| Trained encoder, unseen Enron mail | vague queries nDCG@10 0.692 vs 0.635 base; subject to body 0.408 vs 0.342 | TRAINING.md |
| Training scale | 224,867 emails, 19,980 Grok labels, 491k teacher scores, one RTX 4070 SUPER overnight | TRAINING.md |

## Agent harnesses and routing

We use these agents ourselves, so we tested how they route retrieval. In our own testing,
done outside this repo, the Cursor CLI was the most token-efficient at routing compared with
Claude Code, Codex and Grok, so Cursor is the harness we demo with. Inbox Atlas is harness
agnostic: it is an MCP server, so the same tool works in Cursor, Claude Code, Hermes and
OpenClaw. If a judge asks for numbers on that comparison, say it was informal testing, not a
benchmark in this repo.

## How it generalizes

- **Email** (built): Gmail read-only, region search, "is this here", iMessage, voice.
- **Second brain** (built): Obsidian vault chunked by heading, embedded locally, never sent to
  Grok at ingest; only the packed excerpts go out at question time.
- **Nested markdown and agent memory** (built): folder vectors over the tree, `points_of_interest`
  to find where a question lives, `within` to drill into a subtree, `related_folders` for
  neighbors. Hermes and OpenClaw memory folders are the same shape.
- **Next**: calendar, docs and chat logs behind the same tool; per-user fine-tuning (built, needs
  the real Gmail export); agent feedback becoming training pairs.

## Likely judge questions

**Isn't this just RAG?** RAG retrieves top-k chunks and always returns something. We return a
region with a calibrated yes or no, cut to sentences, under a budget. The headline is tokens per
correct answer and abstention, and we measured both against RAG baselines.

**Is it more accurate than RAG?** No, and we say so. Whole-document RAG is 92% accurate at
4,270 tokens; our pack is 75 to 79% at 156 to 465 tokens. We trade some accuracy on broad,
multi-item questions for 10x to 20x fewer tokens, and we stop early when nothing is there.

**What did you train?** atlas-embed: a frozen bge-base backbone with 1.18M LoRA parameters,
Matryoshka heads, and distillation from a frozen cross-encoder teacher, on 224,867 Enron emails.
It beats the base model on vague queries and subject to body. It is worse on folder names it
never saw, and we report that.

**What did not work?** Our learned Set Transformer region did not beat the simple region on
held-out topics. It calibrates well, so it is kept as an option, and the live system uses the
simple region. We also caught and fixed a late bug where that model was silently switched on.

**Privacy?** Gmail is read-only, nothing is deleted or even marked read, raw messages are kept.
Embeddings are computed locally. Only the few excerpts needed for an answer go to Grok. The
phone link is behind a password.

**Why Grok?** Facet writing, the tool-calling agent, speech to text, dictation cleanup and the
realtime voice agent all run on Grok.

**Where does Tiger Data fit?** Emails, vectors (pgvector HNSW), chat history and a TimescaleDB
query log live in Postgres on the Tiger Data stack, locally in Docker; Tiger Cloud is one
connection string.

**Is it novel?** The parts have prior art (semantic email search, query expansion, LoRA,
distillation, queries as regions in Query2Box). Our contribution is the system: a calibrated,
token-budgeted context tool with abstention and folder navigation for agents, with measured
results including the ones that failed.

## Tracks

Big Red (Navigation), Software, People's Choice, Photon (Agents in iMessage). SpaceX if the
team built with Cursor and you can show it.
