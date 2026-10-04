# Inbox Atlas: pitch kit

Everything a presenter needs: hooks, three timed versions, the demo talk track, the numbers to
memorize, the "isn't this just RAG" answer, track angles and judge Q&A. Every number here was
measured in this repo; the source is in the cheat sheet.

---

## 1. The one line

> **Your agents read 4,000 tokens to find one email. Inbox Atlas hands them the 250 that matter, or tells them nothing is there.**

Alternate hooks (pick one, open with it, never explain it first):

- "Search your inbox for *coding competition*. Gmail finds a job interview. Your AI agent finds the same thing, and pays for it."
- "Keyword search found nothing for 7 of our 12 questions. Not wrong answers. Nothing."
- "The most expensive thing an AI agent does is look for something that isn't there."

---

## 2. The 30 second version (hallway, People's Choice, elevator)

Your AI agents search your email and notes the way Gmail does: by matching words. When the
email says "Codeforces Round 1043" and you ask about "coding competitions", they find nothing,
then grep again, and open file after file. Inbox Atlas searches by meaning. It turns a question
into a region, hands the agent only the sentences that answer it, and says "nothing here" when
there is nothing. That's 18 times fewer tokens than grep, and 31 tokens instead of thousands when
the topic is missing. People can use it too: search it, talk to it, or text it on iMessage.

---

## 3. The 2 minute version (judging table)

**Hook (10 s).** Your agents read 4,000 tokens to find one email. We hand them the 250 that
matter, or tell them nothing is there.

**Problem (25 s).** Everyone has an inbox and a pile of notes they can't search. Gmail matches
strings: "coding competition" returns a HackerRank job assessment because it says "coding", and
misses the Codeforces round, the ICPC tryout and the DevPost receipt. Now our AI agents, Claude
Code, Cursor, Hermes, OpenClaw, every second brain setup, search the same way, only more
expensively. They paste whole documents or grep and open file after file, and when the answer
isn't there they keep looking.

**What we built (30 s).** Inbox Atlas is one retrieval layer for people and LLMs. A question
becomes a region of meaning instead of a keyword: Grok writes the facets, "Codeforces round",
"ICPC regional", plus what to exclude, "job coding interview". We score every email and note
against that region with our own encoder, trained overnight on 224,867 real emails on one gaming
GPU: a frozen backbone, LoRA adapters and a frozen teacher model. Then we hand back only the
sentences that answer, under a token budget, or a calibrated "nothing here" so the agent stops.

**Proof (25 s).** On questions phrased differently from the email, keyword search found nothing
for 7 of 12. We found the right emails for all 12, five times the recall. We use 18 times fewer
tokens than grep and a third of standard chunk RAG at the same accuracy. When the topic isn't
there we spend 31 tokens, not thousands. On a real 419-note second brain: the right note 12 of
12 times in 625 tokens instead of 10,907.

**For people too (15 s).** Search it on the web, talk to it with Grok voice, dictate anywhere
Whisperflow style, or text it on iMessage through Photon: "what do I have tomorrow?" It texts you
first too, with a morning brief and alerts when mail lands in a topic you're watching.

**If a judge spots "Dinner Sunday?" in the coding results:** "That one says *after your hackathon*. It's about the hackathon even though it never says coding, which is the point."

**Close (15 s).** Navigation used to mean maps. For the next hundred years it also means
navigating what we and our agents know. Inbox Atlas is a map of your information that people and
LLMs can both use, at a fraction of the tokens.

---

## 4. The 4 minute finalist version (with live demo)

Run the windows from DEMO.md. Say the **bold lines**; the rest is what is on screen.

| Time | Screen | Say |
|---|---|---|
| 0:00 | Title slide or README hero | **"Your agents read 4,000 tokens to find one email. We hand them the 250 that matter."** |
| 0:15 | Browser, Keyword mode, `coding competition` | **"This is how Gmail and most agents search today: by words."** The job assessment comes first; no contests. |
| 0:35 | Switch to Region mode | **"Same question, searched by meaning."** Codeforces, LeetCode, ICPC, DevPost, Kaggle, BigRed. Point at the facet chips: **"Grok wrote these, including what to exclude. That's why the job test is gone."** |
| 1:00 | RELATED: YES badge, then type `yacht maintenance` | **"And it knows when nothing is there."** NO. **"That one answer is where most of the token savings come from."** |
| 1:15 | Map | **"Your inbox as a map. The question becomes a region."** |
| 1:30 | Terminal A, mock iMessage: `what do I have tomorrow?` | **"For people, it's just a text."** Monday 10am lab meeting. Then `did anyone ask me for money?` |
| 2:00 | Terminal B, Cursor: arm servo question | **"For agents, it's one MCP tool."** Folder, note and the exact line, about 210 tokens. **"It found the folder first, then the note, in a nested markdown vault. That's how it generalizes to any second brain."** |
| 2:30 | Same terminal: `did I book a yacht charter?` | About 23 tokens, stop. **"No grep loop. It just stops."** |
| 2:45 | token_savings.png | **"Seven of twelve questions: keyword found nothing. We found all twelve. 18x fewer tokens than grep, 60x fewer when the topic isn't there."** |
| 3:15 | more_than_rag.png | **"And it's more than RAG: RAG always returns k chunks. We return a decision, sentences instead of chunks, and folders before notes."** |
| 3:40 | architecture.png | **"Under it: our own encoder, frozen backbone, LoRA, a frozen teacher, trained overnight on a quarter million emails on one GPU."** |
| 3:50 | Close | **"Navigation for the next hundred years means navigating what we and our agents know."** |

If the live demo fails at any step, switch to the README graphics and keep talking; never
debug in front of judges.

---

## 5. Numbers to memorize (five is enough)

| Say | Exact | Source |
|---|---|---|
| "Keyword found nothing for 7 of 12" | 7 of 12 meaning-based queries with zero relevant hits; Atlas 0 of 12 | eval/results.md |
| "Five times the recall" | Recall@10 0.867 vs 0.175 | eval/results.md |
| "18 times fewer tokens than grep" | 249 vs 4,445 tokens per question | eval/token_results.md |
| "A third of chunk RAG at the same accuracy" | 249 vs 740 tokens, both 75% | eval/token_results.md |
| "31 tokens when it isn't there" | vs 742 chunk RAG, 1,858 grep; abstains on 13 of 14 | eval/token_results.md |

Backup numbers:

| Claim | Exact | Source |
|---|---|---|
| Real second brain | right note 12 of 12 at 625 tokens; grep 10,907; whole notes 7,400 | eval/token_results.md |
| Folder navigation | folder, note and exact line in 210 tokens; absent topic 23 tokens | README |
| "Is this here?" | 20 of 20 demo topics | assets/related_calibration.png |
| Search quality | region 7 of top 8 on topic, cosine 6, keyword 0 | assets/search_modes.png |
| Trained encoder | vague queries nDCG@10 0.692 vs 0.635 base; subject to body 0.408 vs 0.342 | TRAINING.md |
| Region score | membership AUROC 0.819 to 0.843 with the trained encoder | TRAINING.md |
| Training scale | 224,867 emails, 19,980 Grok labels, 491k teacher scores, one RTX 4070 SUPER overnight | TRAINING.md |
| Engineering | 11 branches merged through an integration branch, 92 Python tests, 19 Node tests | repo |

---

## 6. "Isn't this just RAG?" (the answer to rehearse)

> "RAG is the starting point. Standard RAG embeds your question, grabs the top k chunks and pastes them in. It always returns something. We changed three things, and each one is measured against RAG on the same data. One: it knows when nothing is there, so an absent topic costs 31 tokens instead of 742. Two: it hands over the sentences that answer, not whole chunks, so it's a third of the tokens at the same accuracy. Three: it navigates folders before notes, so in a deep vault it finds the folder, the note and the line in about 200 tokens. Plus the encoder underneath is trained on email."

Short version: **"RAG always returns k chunks. We return a decision."**

---

## 7. Track angles

- **Big Red (Navigation).** Navigation of information. A question becomes a region on a map of
  your inbox; Grok steers it like moving a map (widen, narrow, exclude); folders before notes is
  wayfinding through a knowledge tree. Kleinberg's own work is navigation in networks and
  information spaces.
- **Software.** Five subsystems working end to end: a trained encoder, region search with
  calibrated abstention, an MCP server, Postgres with pgvector and TimescaleDB on Tiger Cloud, and voice plus iMessage frontends. One command (`atlas up`) runs it all; 92 tests.
- **People's Choice.** Everyone has an inbox they can't search. Demo: text it "what do I have
  tomorrow?"
- **Photon (Agents in iMessage).** Two-way: you text it questions, and it texts you first with
  a morning brief and alerts when new mail lands in a topic you asked it to watch ("watch
  internships"). Built on Spectrum; the same agent can run on WhatsApp or Telegram.
- **SpaceX (Make it Legendary).** Grok everywhere: facet writing, the tool-calling agent, speech
  to text, dictation cleanup and the realtime voice agent. In our own testing outside this repo,
  the Cursor CLI was the most token-efficient harness at routing compared with Claude Code, Codex
  and Grok, so Cursor is what we demo the MCP tool in. Only enter this track if the team built
  with Cursor.

---

## 8. Judge Q&A

**Is it more accurate than RAG?**
"Whole-document RAG reads everything, so it's the accuracy ceiling at about 4,300 tokens a
question. We match chunk RAG's accuracy at a third of its tokens, and we stop early when the topic
isn't there. The win is accuracy per token."

**What did you actually train?**
"atlas-embed: a frozen bge-base backbone with 1.18 million trainable LoRA parameters, Matryoshka
heads so it also works at 256 dimensions, and listwise distillation from a frozen cross-encoder
teacher. 224,867 Enron emails, 19,980 labeled by Grok, 491 thousand teacher scores, one RTX 4070
SUPER overnight. It beats the base model on vague queries, 0.692 versus 0.635."

**Why Enron?**
"It's the largest public real email corpus, with user folders we could use as free topic labels,
and we didn't want to train on anyone's private mail. Your own Gmail can be added as a
fine-tuning stage; that path is built."

**How does "nothing here" work?**
"Each document's score is compared against what it scores on random topic sets with the same
number of facets. A real match has to stand out from chance. That gives a calibrated yes or no
with no extra training, and it's why absent topics cost 31 tokens."

**Why Grok?**
"Grok writes the facets in the vocabulary emails actually use, runs the tool-calling agent, does
speech to text and dictation cleanup, and powers the realtime voice agent."

**Where does Tiger Data fit?**
"It runs on Tiger Cloud: emails, embeddings with a pgvector HNSW index, chat history, and a
TimescaleDB hypertable that logs every query with its latency and tokens. The search engine
reads from it live."

**Is it private?**
"Gmail is read-only: nothing is deleted or even marked read, and raw messages are kept.
Embeddings are computed on your machine. Only the few sentences needed for an answer go to Grok.
The phone link is password protected."

**How does an agent use it?**
"It's an MCP server, so Cursor, Claude Code, Hermes and OpenClaw call it like any tool:
`atlas_context` for an answer pack, `atlas_points_of_interest` for folders in a vault."

**What's novel?**
"The parts have prior art: semantic search, query expansion, LoRA, distillation. What's new is
the system: calibrated abstention, sentence-level packing under a budget, and folder-first
navigation in one tool, measured end to end in tokens."

**What's next?**
"Fine-tune on each user's own mail, add calendar, docs and chat logs behind the same tool, and
turn agent feedback into training pairs."

---

## 9. Say this, not that

| Say | Don't say | Why |
|---|---|---|
| "18x fewer tokens than grep" | "fewer searches" with a number | we measured tokens, not search counts |
| "a third of chunk RAG at the same accuracy" | "more accurate than RAG" | whole-document RAG is the accuracy ceiling |
| "in our own testing, Cursor CLI routed most efficiently" | Cursor benchmark numbers | the comparison was informal, outside this repo |
| "on the demo inbox" | "on my real inbox" | the live demo uses the fictional demo inbox |

---

## 10. Who says what (suggested split for four)

| Part | Presenter |
|---|---|
| Hook, problem, close | the strongest speaker |
| Live search and map | whoever drives the browser |
| iMessage and voice | a second laptop or phone |
| Agent, tokens, more than RAG, model | whoever can answer technical follow-ups |

Practice the 2 minute version until it fits in 2 minutes with the five numbers said from memory.
