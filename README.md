<a id="readme-top"></a>

# Inbox Atlas

Navigate your inbox by meaning. Say or text "coding competition" and get the hackathon
acceptance, the Codeforces reminder and the DevPost receipt, even though none of them use
those words. Ask "what do I have today?" over iMessage and get an answer.

## About The Project

Inbox Atlas embeds your whole Gmail with a custom encoder trained on email, then turns every
question into a region of embedding space instead of a keyword string. Grok writes a topic list
for the question, a learned set encoder compresses that list into a region, and a calibrated
membership score answers both "which emails?" and "is this topic in my inbox at all?".

See `PLAN.md` for the full design and `CONTRACT.md` for interfaces.

### Built With

Python, FastAPI, PyTorch, sentence-transformers, PEFT (LoRA), SQLite FTS5, UMAP, Grok
(chat, STT, realtime voice), Photon Spectrum (iMessage), built in Cursor.

## Getting Started

```sh
cp .env.example .env    # fill in keys
uv sync --extra dev
uv run pytest -q
uv run python server.py # http://localhost:8765
```

Training instructions live on the `training` branch in `TRAINING.md`.

## Usage

Web: search box, mic button, inbox map. iMessage: text the Atlas number. Dictation:
`uv run --extra dictate python dictate/dictate.py`, hold Right Option to talk.

## Roadmap

- [ ] Gmail export and index
- [ ] Custom encoder and learned region model
- [ ] Region search with Grok topic expansion
- [ ] Voice dictation and voice agent
- [x] iMessage agent with proactive texts (see docs/IMESSAGE.md)

## Acknowledgments

BigRed//Hacks 2026, xAI Grok, Photon, BAAI bge, Enron email corpus (CMU).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
