"""Score (query, candidate) pairs with a FROZEN cross-encoder teacher for listwise distillation.

For every unique query text of the topic / subject / grok anchors: the top mined candidates
plus all of that query's positives. Writes data/datasets/<ds>/teacher.json {query: {id: logit}}.
Resumable: an existing teacher.json is extended, not recomputed.

usage: uv run python -m train.teacher [--ds enron] [--model BAAI/bge-reranker-v2-m3] [--ncand 31]
"""

import argparse
import json
import random
import time
from collections import defaultdict

from train.common import DATASETS, doc_text, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="enron")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--ncand", type=int, default=31)
    ap.add_argument("--max-subject", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = DATASETS / a.ds
    emails = {e["id"]: e for e in read_jsonl(d / "emails.jsonl")}
    anchors = read_jsonl(d / "anchors.jsonl")
    mined = json.loads((d / "mined.json").read_text())
    path = d / "teacher.json"
    done = json.loads(path.read_text()) if path.exists() else {}

    pos = defaultdict(set)
    mode = {}
    kinds = {}
    for x in anchors:
        if x["kind"] == "reply":
            continue
        pos[x["q"]].add(x["pos"])
        mode[x["q"]] = x["mode"]
        kinds[x["q"]] = x["kind"]
    qs = list(pos)
    subj = [q for q in qs if kinds[q] == "subject"]
    rest = [q for q in qs if kinds[q] != "subject"]
    random.Random(0).shuffle(subj)
    qs = rest + subj[: a.max_subject]
    pairs = []
    for q in qs:
        have = done.get(q, {})
        for c in list(pos[q]) + mined.get(q, [])[: a.ncand]:
            if c in emails and c not in have:
                pairs.append((q, c))
    pairs = list(dict.fromkeys(pairs))
    print(f"teacher {a.model}: {len(qs)} queries, {len(pairs)} new pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    texts = [(q, doc_text(emails[c], mode[q])[:2000]) for q, c in pairs]
    order = sorted(range(len(pairs)), key=lambda i: -len(texts[i][1]))
    t0 = time.time()
    with torch.inference_mode():
        for bi, i in enumerate(range(0, len(order), a.bs)):
            idx = order[i : i + a.bs]
            enc = tok([texts[j][0] for j in idx], [texts[j][1] for j in idx], padding=True, truncation="only_second",
                      max_length=a.max_len, return_tensors="pt").to("cuda")
            s = model(**enc).logits.view(-1).float().cpu().tolist()
            for j, v in zip(idx, s):
                q, c = pairs[j]
                done.setdefault(q, {})[c] = round(v, 4)
            if bi % 200 == 0:
                el = time.time() - t0
                rate = (i + len(idx)) / max(el, 1e-6)
                print(f"{i + len(idx)}/{len(pairs)} pairs, {rate:.0f}/s, eta {(len(pairs) - i) / max(rate, 1e-6) / 60:.0f}m", flush=True)
            if bi % 3000 == 2999:
                path.write_text(json.dumps(done))
    path.write_text(json.dumps(done))
    print(f"wrote {path} in {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
