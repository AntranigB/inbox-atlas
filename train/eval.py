"""Evaluate encoders and region models. Writes runs/<run>/metrics.json and prints markdown tables.

Retrieval (Recall@10 = hits in top 10 / min(|relevant|, 10), nDCG@10, binary relevance):
  topic_seen        train folder labels -> train emails (20k sample)          memorization check
  topic_heldthread  train folder labels -> unseen emails (test_thread + test_topic corpus)
  topic_heldfolder  held-out folder labels -> unseen emails (same corpus)    never-seen topics
  subject_heldthread  subject -> its own email rendered without subject (test_thread)
  grok_query_test   Grok search query -> its email (test corpus), if Grok labels exist
Systems: BM25, frozen base bge (768), each --models checkpoint at 768/256/64.

Region membership on held-out folders (test corpus): positives = folder name + Grok phrases of
3 seed member emails (the seeds are removed from scoring), negatives = 2 nearest train folder
names. Heuristic region (PLAN.md 3b: 0.6 max-facet + 0.4 centroid - 0.5 relu(neg - facet),
Platt-calibrated on val) vs learned region (its own fitted T). AUROC (mean over topics), pooled ECE,
nDCG@10 of the region ranking.

usage: uv run python -m train.eval --run eval1 --models models/atlas-embed[,models/a1,...] [--ds enron]
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from train.common import DATASETS, RUNS, doc_text, read_jsonl
from train.metrics import auroc, ece, fit_platt, retrieval
from train.topic_sets import load_grok


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer

        self.cv = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", lowercase=True, min_df=1)
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        n = tf.shape[0]
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        tf.data = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.W = tf.tocsr()

    def scores(self, queries):
        q = self.cv.transform(queries)
        q.data[:] = 1
        return (q @ self.W.T).toarray()


def heuristic_score(E, P, N):
    c = P.mean(0)
    c = c / np.linalg.norm(c)
    sf = (E @ P.T).max(1)
    sc = E @ c
    sn = (E @ N.T).max(1) if len(N) else 0
    return 0.6 * sf + 0.4 * sc - 0.5 * np.maximum(0, sn - sf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ds", default="enron")
    ap.add_argument("--models", default="models/atlas-embed")
    ap.add_argument("--dims", default="768,256,64")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--seen-sample", type=int, default=20000)
    a = ap.parse_args()

    from atlas.model.trained import load, load_base

    t0 = time.time()
    d = DATASETS / a.ds
    emails = read_jsonl(d / "emails.jsonl")
    topics = json.loads((d / "topics.json").read_text()) if (d / "topics.json").exists() else {}
    grok = load_grok(d / "grok_labels.jsonl")
    rng = random.Random(0)
    test = [e for e in emails if e["split"] in ("test_thread", "test_topic")]
    train = [e for e in emails if e["split"] == "train"]
    rng.shuffle(train)
    train = train[: a.seen_sample]
    val = [e for e in emails if e["split"] == "val"]
    tl = sorted(l for l, v in topics.items() if v["split"] == "train")
    hl = sorted(l for l, v in topics.items() if v["split"] == "heldout")

    def rel_for(labels, docs):
        rel = [{j for j, e in enumerate(docs) if l in e["topics"]} for l in labels]
        keep = [i for i, r in enumerate(rel) if len(r) >= 3]
        return [labels[i] for i in keep], [rel[i] for i in keep]

    tasks = {}
    q, r = rel_for(tl, train)
    tasks["topic_seen"] = ("train", q, r, "full")
    q, r = rel_for(tl, test)
    tasks["topic_heldthread"] = ("test", q, r, "full")
    q, r = rel_for(hl, test)
    tasks["topic_heldfolder"] = ("test", q, r, "full")
    st = [j for j, e in enumerate(test) if e["split"] == "test_thread" and not e["is_reply"] and len(e["subject"]) >= 8]
    rng.shuffle(st)
    st = st[:3000]
    tasks["subject_heldthread"] = ("test", [test[j]["subject"] for j in st], [{j} for j in st], "nosubj")
    gq = [(j, grok[e["id"]]["queries"][-1]) for j, e in enumerate(test) if e["id"] in grok and grok[e["id"]].get("queries")]
    if gq:
        tasks["grok_query_test"] = ("test", [x[1] for x in gq], [{x[0]} for x in gq], "full")
        gv = [(j, grok[e["id"]]["queries"][0]) for j, e in enumerate(test) if e["id"] in grok and len(grok[e["id"]].get("queries", [])) > 1]
        tasks["grok_vague_test"] = ("test", [x[1] for x in gv], [{x[0]} for x in gv], "full")
    corpora = {"train": train, "test": test}
    print({k: len(v[1]) for k, v in tasks.items()}, f"corpus test {len(test)} train {len(train)}", flush=True)

    results = {}
    # BM25
    for name, (corp, qs, rel, mode) in tasks.items():
        bm = BM25([doc_text(e, mode) for e in corpora[corp]])
        results.setdefault(name, {})["bm25"] = retrieval(bm.scores(qs), rel)
    print("bm25 done", f"{time.time() - t0:.0f}s", flush=True)

    encs = [] if a.no_base else [("base-bge", None)]
    encs += [(Path(m).name, m) for m in a.models.split(",") if m and (Path(m) / "atlas.json").exists()]
    dims = [int(x) for x in a.dims.split(",")]
    region_data = {}
    for ename, path in encs:
        enc = load_base() if path is None else load(path)
        cache = {}
        for corp, docs in corpora.items():
            for mode in ("full", "nosubj"):
                if any(t[0] == corp and t[3] == mode for t in tasks.values()):
                    cache[(corp, mode)] = enc.encode_docs([doc_text(e, mode) for e in docs])
        for name, (corp, qs, rel, mode) in tasks.items():
            Q = enc.encode_queries(qs)
            D = cache[(corp, mode)]
            for dim in (dims if path else [768]):
                from atlas.model.trained import truncate

                key = ename if dim == 768 else f"{ename}@{dim}"
                results[name][key] = retrieval(truncate(Q, dim) @ truncate(D, dim).T, rel)
        region_data[ename] = (enc, path, cache[("test", "full")])
        print(f"{ename} done {time.time() - t0:.0f}s", flush=True)

    # region eval
    reg = {}
    rq, rrel = rel_for(hl, test)
    tq = [l for l in tl]
    for ename, (enc, path, D) in region_data.items():
        TL = enc.encode_queries(tq)
        Dv = enc.encode_docs([doc_text(e) for e in val])
        sets = []
        for l, rset in zip(rq, rrel):
            mem = sorted(rset)
            seeds = random.Random(l).sample(mem, min(3, len(mem) - 1))
            phr = [l] + [p for s in seeds for p in grok.get(test[s]["id"], {}).get("topics", [])]
            lv = enc.encode_queries([l])[0]
            negs = [tq[i] for i in np.argsort(-(TL @ lv))[:2]]
            sets.append((l, phr, negs, set(seeds), rset))
        # heuristic platt on val using train labels
        vy, vs = [], []
        for l in random.Random(1).sample(tl, min(60, len(tl))):
            y = np.array([1.0 if l in e["topics"] else 0.0 for e in val])
            if y.sum() < 3:
                continue
            P = enc.encode_queries([l])
            vs.append(heuristic_score(Dv, P, np.zeros((0, P.shape[1]))))
            vy.append(y)
        pa, pc = fit_platt(np.concatenate(vy), np.concatenate(vs)) if vy else (1.0, 0.0)
        learned = None
        if path and (Path(path) / "region.pt").exists():
            from atlas.model.region_encoder import LearnedRegionModel

            learned = LearnedRegionModel.load(Path(path) / "region.pt")
        out = {"heuristic": {"auroc": [], "ndcg": [], "y": [], "p": []}}
        variants = {"heuristic": None}
        if learned:
            variants["learned"] = learned
            out["learned"] = {"auroc": [], "ndcg": [], "y": [], "p": []}
        for l, phr, negs, seeds, rset in sets:
            P = enc.encode_queries(phr)
            N = enc.encode_queries(negs)
            keep = np.array([j not in seeds for j in range(len(test))])
            y = np.array([1.0 if j in rset else 0.0 for j in range(len(test))])[keep]
            Dk = D[keep]
            for vname, model in variants.items():
                if model is None:
                    s = heuristic_score(Dk, P, N)
                    p = 1 / (1 + np.exp(-(pa * s + pc)))
                else:
                    rg = model.build(P, N)
                    s, p = rg.score(Dk), rg.prob(Dk)
                o = out[vname]
                o["auroc"].append(auroc(y, s))
                o["ndcg"].append(retrieval(s[None], [set(np.where(y > 0)[0])])["ndcg@10"])
                o["y"].append(y)
                o["p"].append(p)
        for vname, o in out.items():
            Y = np.concatenate(o["y"]) if o["y"] else np.zeros(0)
            Pp = np.concatenate(o["p"]) if o["p"] else np.zeros(0)
            reg[f"{ename}/{vname}"] = {"auroc": float(np.nanmean(o["auroc"])) if o["auroc"] else None,
                                       "ndcg@10": float(np.mean(o["ndcg"])) if o["ndcg"] else None,
                                       "ece": ece(Y, Pp) if len(Y) else None, "n_topics": len(o["auroc"])}
    results["region_heldfolder"] = reg

    run = RUNS / a.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.json").write_text(json.dumps(results, indent=1))
    lines = []
    for name, res in results.items():
        if name == "region_heldfolder":
            continue
        lines.append(f"\n### {name} (n={next(iter(res.values()))['n']})\n\n| system | Recall@10 | nDCG@10 |\n|---|---|---|")
        for k, v in res.items():
            lines.append(f"| {k} | {v['recall@10']:.3f} | {v['ndcg@10']:.3f} |")
    lines.append("\n### region membership, held-out folders\n\n| encoder/region | AUROC | nDCG@10 | ECE |\n|---|---|---|---|")
    for k, v in reg.items():
        lines.append(f"| {k} | {v['auroc']:.3f} | {v['ndcg@10']:.3f} | {v['ece']:.4f} |")
    md = "\n".join(lines)
    (run / "metrics.md").write_text(md)
    print(md, flush=True)
    print(f"eval done in {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
