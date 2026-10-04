"""Build training anchors and mine hard-negative candidates with the frozen base bge.

Anchor kinds (all from the train split only):
  topic       humanized folder label -> email in that folder          (query side)
  subject     subject line -> the email rendered without its subject  (query side)
  reply       reply body -> parent body                               (doc side, symmetric)
  grok_topic  Grok topic phrase -> email   (if data/datasets/<ds>/grok_labels.jsonl exists)
  grok_query  Grok search query -> email

Writes data/datasets/<ds>/anchors.jsonl and mined.json ({query text: [top candidate ids]}).

usage: uv run python -m train.mine [--ds enron] [--topk 48]
"""

import argparse
import json
import random
import time
from collections import defaultdict

import numpy as np

from train.common import DATASETS, doc_text, norm_subject, read_jsonl

GENERIC_SUBJ = {"hello", "hi", "meeting", "update", "question", "thanks", "thank you", "fyi", "test", "info", "lunch", "call"}


def build_anchors(emails, topics, rng, cap_topic=60, cap_subject=60000, cap_reply=30000, grok=None, split="train"):
    tr = [e for e in emails if e["split"] == split]
    byid = {e["id"]: e for e in emails}
    anchors = []
    by_label = defaultdict(list)
    for e in tr:
        for l in e["topics"]:
            if topics.get(l, {}).get("split") == "train":
                by_label[l].append(e["id"])
    for l, ids in by_label.items():
        rng.shuffle(ids)
        for i in ids[:cap_topic]:
            anchors.append({"kind": "topic", "q": l, "qq": True, "pos": i, "mode": "full", "label": l})
    subj = [e for e in tr if not e["is_reply"] and len(norm_subject(e["subject"])) >= 8
            and norm_subject(e["subject"]) not in GENERIC_SUBJ]
    rng.shuffle(subj)
    for e in subj[:cap_subject]:
        anchors.append({"kind": "subject", "q": e["subject"], "qq": True, "pos": e["id"], "mode": "nosubj"})
    rep = [e for e in tr if e.get("parent_id") and e["parent_id"] in byid and byid[e["parent_id"]]["split"] == split]
    rng.shuffle(rep)
    for e in rep[:cap_reply]:
        anchors.append({"kind": "reply", "q": doc_text(e, "nosubj"), "qq": False, "pos": e["parent_id"], "mode": "nosubj"})
    if grok:
        trids = {e["id"] for e in tr}
        for g in grok:
            if g["id"] not in trids:
                continue
            for p in g.get("topics", []):
                anchors.append({"kind": "grok_topic", "q": p, "qq": True, "pos": g["id"], "mode": "full"})
            for p in g.get("queries", []):
                anchors.append({"kind": "grok_query", "q": p, "qq": True, "pos": g["id"], "mode": "full"})
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="enron")
    ap.add_argument("--topk", type=int, default=48)
    ap.add_argument("--cap-topic", type=int, default=60)
    ap.add_argument("--cap-subject", type=int, default=60000)
    ap.add_argument("--cap-reply", type=int, default=30000)
    ap.add_argument("--doc-len", type=int, default=256)
    a = ap.parse_args()

    import torch

    from atlas.model.trained import load_base

    d = DATASETS / a.ds
    emails = read_jsonl(d / "emails.jsonl")
    topics = json.loads((d / "topics.json").read_text())
    grok = read_jsonl(d / "grok_labels.jsonl") if (d / "grok_labels.jsonl").exists() else None
    rng = random.Random(0)
    anchors = build_anchors(emails, topics, rng, a.cap_topic, a.cap_subject, a.cap_reply, grok)
    kinds = defaultdict(int)
    for x in anchors:
        kinds[x["kind"]] += 1
    print("anchors", dict(kinds), flush=True)

    tr = [e for e in emails if e["split"] == "train"]
    enc = load_base(max_len=a.doc_len, batch_size=128)
    t0 = time.time()
    D = enc.encode_docs([doc_text(e) for e in tr])
    print(f"embedded {len(tr)} train docs in {time.time() - t0:.0f}s", flush=True)
    np.save(d / "base_train_emb.npy", D.astype(np.float16))
    (d / "base_train_ids.json").write_text(json.dumps([e["id"] for e in tr]))

    uq = {}
    for x in anchors:
        uq.setdefault(x["q"], x["qq"])
    qs = list(uq)
    qq = [q for q in qs if uq[q]]
    qd = [q for q in qs if not uq[q]]
    enc.max_len = 64
    Q1 = enc.encode_queries(qq)
    enc.max_len = a.doc_len
    Q2 = enc.encode_docs(qd)
    Q = np.concatenate([Q1, Q2]) if len(qd) else Q1
    qs = qq + qd
    Dt = torch.from_numpy(D).to(enc.device, torch.float16)
    ids = [e["id"] for e in tr]
    mined = {}
    for i in range(0, len(qs), 1024):
        q = torch.from_numpy(Q[i : i + 1024]).to(enc.device, torch.float16)
        top = (q @ Dt.T).topk(a.topk, dim=1).indices.cpu().numpy()
        for j, row in enumerate(top):
            mined[qs[i + j]] = [ids[k] for k in row]
    print(f"mined {len(mined)} queries in {time.time() - t0:.0f}s", flush=True)
    from train.common import write_jsonl

    write_jsonl(d / "anchors.jsonl", anchors)
    (d / "mined.json").write_text(json.dumps(mined))


if __name__ == "__main__":
    main()
