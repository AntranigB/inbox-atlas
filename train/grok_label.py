"""Label emails with Grok: 2-3 abstract topic phrases (avoiding the email's own words) and 2
search queries (one vague, one specific). Async, concurrency 8, 10 emails per call, resumable.

Only the subject and the first 800 chars of the body are sent to xAI.
Output: data/datasets/<ds>/grok_labels.jsonl rows {id, topics: [...], queries: [...]}.

usage: uv run python -m train.grok_label --ds enron --cap 20000
"""

import argparse
import asyncio
import json
import os
import random
import re
import time

import httpx

from train.common import DATASETS, ROOT, read_jsonl

URL = "https://api.x.ai/v1/chat/completions"

SYSTEM = """You label emails for training a semantic search model.
For each email return:
- "topics": 2 to 3 short abstract topic phrases (2 to 5 words) describing what the email is about,
  the way a person would describe the category later. Do NOT reuse distinctive words from the
  subject or body; paraphrase at a higher level (e.g. "coding competition", "money owed to a friend",
  "power plant financing").
- "queries": exactly 2 search queries a person might type to find this email later: first a vague
  one (what they half remember), then a specific one.
Return JSON: {"items": [{"i": <index>, "topics": [...], "queries": [...]}, ...]} with one item per email."""


def load_key():
    k = os.getenv("XAI_API_KEY")
    if k:
        return k
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("XAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("XAI_API_KEY not set")


def render(i, e):
    body = re.sub(r"\s+", " ", e.get("body") or "")[:800]
    return f"[{i}] Subject: {e.get('subject') or ''}\n{body}"


async def call(client, key, model, batch, sem, retries=4):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(render(i, e) for i, e in enumerate(batch))}]
    async with sem:
        for t in range(retries):
            try:
                r = await client.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=90,
                                      json={"model": model, "messages": msgs, "temperature": 0.3,
                                            "response_format": {"type": "json_object"}})
                if r.status_code == 429 or r.status_code >= 500:
                    await asyncio.sleep(2 ** t + random.random())
                    continue
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"]
                items = json.loads(txt)["items"]
                out = []
                for it in items:
                    i = int(it["i"])
                    if 0 <= i < len(batch):
                        tp = [str(x).strip() for x in it.get("topics", []) if str(x).strip()][:3]
                        qs = [str(x).strip() for x in it.get("queries", []) if str(x).strip()][:2]
                        if tp:
                            out.append({"id": batch[i]["id"], "topics": tp, "queries": qs})
                return out
            except Exception as ex:  # noqa: BLE001
                if t == retries - 1:
                    print("batch failed:", type(ex).__name__, str(ex)[:120], flush=True)
                await asyncio.sleep(2 ** t)
    return []


def select(emails, cap, seed=0):
    """Labeled-folder emails first (they also have folder ground truth), then the rest, all splits."""
    rng = random.Random(seed)
    lab = [e for e in emails if e.get("topics")]
    unl = [e for e in emails if not e.get("topics")]
    rng.shuffle(lab)
    rng.shuffle(unl)
    n_lab = min(len(lab), int(cap * 0.6))
    return lab[:n_lab] + unl[: cap - n_lab]


async def run(a):
    d = DATASETS / a.ds
    out = d / "grok_labels.jsonl"
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    emails = read_jsonl(d / "emails.jsonl")
    todo = [e for e in select(emails, a.cap) if e["id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to label", flush=True)
    key = load_key()
    sem = asyncio.Semaphore(a.concurrency)
    batches = [todo[i : i + a.batch] for i in range(0, len(todo), a.batch)]
    t0, n = time.time(), 0
    async with httpx.AsyncClient() as client:
        with open(out, "a") as f:
            for i in range(0, len(batches), a.concurrency * 4):
                chunk = batches[i : i + a.concurrency * 4]
                res = await asyncio.gather(*(call(client, key, a.model, b, sem) for b in chunk))
                for rows in res:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                        n += 1
                f.flush()
                el = time.time() - t0
                print(f"labeled {n}/{len(todo)} in {el / 60:.1f}m, eta {(len(todo) - n) / max(n / el, 1e-6) / 60:.0f}m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="enron")
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default=os.getenv("GROK_MODEL", "grok-4.20-0309-non-reasoning"))
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
