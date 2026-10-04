"""Retrieval eval: keyword vs embedding vs region (+ ablations), Grok as judge.

Usage: uv run python -m eval.run_eval [--encoder base] [--queries eval/queries.json]
Writes eval/results.md. Judge labels are cached in data/cache/judge.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from atlas import config, store
from atlas.agent import grok, prompts
from atlas.search import hybrid
from atlas.search import region as R
from atlas.search.engine import get_engine

HERE = Path(__file__).parent
POOL = 20


def _judge_cache():
    p = config.DATA / "cache" / "judge.json"
    return p, (json.loads(p.read_text()) if p.exists() else {})


def judge(query, ids, conn):
    """Grok labels each pooled email 0/1 for the query. Cached per (query, id)."""
    path, cache = _judge_cache()
    todo = [i for i in ids if f"{query}||{i}" not in cache]
    for s in range(0, len(todo), 15):
        batch = todo[s:s + 15]
        lines = []
        for eid in batch:
            r = store.get_email(conn, eid) or {}
            body = " ".join((r.get("body") or "").split())[:400]
            lines.append(f"<{eid}> from: {r.get('from_name') or r.get('from_addr')} | subject: {r.get('subject')} | {body}")
        msg = grok.chat([{"role": "user", "content": prompts.JUDGE.format(query=query, emails="\n".join(lines))}],
                        json_mode=True, temperature=0)
        labels = grok._parse_json(msg.get("content")).get("labels", {})
        for eid in batch:
            cache[f"{query}||{eid}"] = int(labels.get(eid, 0) or 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=0))
    return {i: cache[f"{query}||{i}"] for i in ids}


def rankings(eng, q, exp):
    ix = eng.index
    pos, neg = exp["positive"], exp["negative"]
    out = {}
    out["keyword (BM25)"] = [e for e, _ in store.fts_search(eng.conn, q, POOL)]
    qv = eng.enc.encode_queries([q])[0]
    out["embedding (cosine)"] = [ix.ids[i] for i in np.argsort(-(ix.E @ qv))[:POOL]]
    _, raw, z = hybrid.region_scores(eng, q, [], [])
    out["region, query only"] = [ix.ids[i] for i in np.argsort(-z)[:POOL]]
    _, raw, z = hybrid.region_scores(eng, q, pos, [])
    out["region, no negatives"] = [ix.ids[i] for i in np.argsort(-z)[:POOL]]
    _, raw, z = hybrid.region_scores(eng, q, pos, neg)
    out["region, no hub z (raw)"] = [ix.ids[i] for i in np.argsort(-raw)[:POOL]]
    out["region (full)"] = [ix.ids[i] for i in np.argsort(-z)[:POOL]]
    out["hybrid (region + BM25 RRF)"] = [h["id"] for h in hybrid.search(q, pos, neg, k=POOL, mode="hybrid", engine=eng)["hits"]]
    return out


def ndcg(rank, rel, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, e in enumerate(rank[:k]) if rel.get(e))
    n = sum(rel.values())
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n, k)))
    return dcg / idcg if idcg else 0.0


def recall(rank, rel, k=10):
    n = sum(rel.values())
    return sum(1 for e in rank[:k] if rel.get(e)) / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--queries", default=str(HERE / "queries.json"))
    ap.add_argument("--out", default=str(HERE / "results.md"))
    a = ap.parse_args()
    spec = json.loads(Path(a.queries).read_text())
    eng = get_engine(a.encoder)
    n_emails = len(eng.index.ids)
    per = {}
    for q in spec["queries"]:
        exp = grok.expand(q)
        ranks = rankings(eng, q, exp)
        pool = list(dict.fromkeys(ranks["keyword (BM25)"] + ranks["embedding (cosine)"] + ranks["region (full)"]))
        rel = judge(q, pool, eng.conn)
        per[q] = {m: (recall(r, rel), ndcg(r, rel)) for m, r in ranks.items()}
        per[q]["_nrel"] = sum(rel.values())
        print(q, per[q]["_nrel"], {m: round(v[1], 2) for m, v in per[q].items() if m != "_nrel"})

    methods = [m for m in next(iter(per.values())) if m != "_nrel"]
    lines = [f"# Inbox Atlas retrieval eval", "",
             f"Encoder `{eng.name}`, {n_emails} emails, {len(per)} handwritten queries, "
             f"pool = top {POOL} of keyword + embedding + region, Grok judge (`{config.GROK_MODEL}`). "
             f"Run {time.strftime('%Y-%m-%d %H:%M')}.", "",
             "| Method | Recall@10 | nDCG@10 |", "|---|---|---|"]
    for m in methods:
        rs = [per[q][m][0] for q in per if per[q]["_nrel"]]
        ns = [per[q][m][1] for q in per if per[q]["_nrel"]]
        lines.append(f"| {m} | {np.mean(rs):.3f} | {np.mean(ns):.3f} |")
    lines += ["", "## Per query nDCG@10", "", "| Query | relevant | " + " | ".join(methods) + " |",
              "|---|---|" + "---|" * len(methods)]
    for q in per:
        lines.append(f"| {q} | {per[q]['_nrel']} | " + " | ".join(f"{per[q][m][1]:.2f}" for m in methods) + " |")

    # related? accuracy
    rows, correct = [], 0
    for truth, topics in (("present", spec["related"]["present"]), ("absent", spec["related"]["absent"])):
        for t in topics:
            exp = grok.expand(t)
            out = hybrid.is_related(t, exp["positive"], exp["negative"], engine=eng)
            ok = out["related"] == (truth == "present")
            correct += ok
            rows.append(f"| {t} | {truth} | {'YES' if out['related'] else 'NO'} | {out['max_z']} | {out['count']} | {'ok' if ok else 'MISS'} |")
    total = len(rows)
    lines += ["", f"## Related? yes/no accuracy: {correct}/{total} = {correct / total:.0%}", "",
              f"Rule: top hub corrected z >= {R.Z_MIN} and raw >= {R.floor_for(eng.name)}.", "",
              "| Topic | truth | answer | max z | count | |", "|---|---|---|---|---|---|"] + rows
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(methods) + 4]))
    print(f"related accuracy {correct}/{total}")


if __name__ == "__main__":
    main()
