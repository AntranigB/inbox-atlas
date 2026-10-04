"""Topic sets for region training and eval.

Two sources:
  folder  Enron folder labels (or Gmail labels): members = emails with the label,
          phrase pool = label + Grok topic phrases of member emails + member subjects
  grok    clusters of Grok topic phrases (embedding + agglomerative over k-means micro
          clusters): members = emails carrying any phrase in the cluster

Each set: {"name", "src", "split": "train"|"heldout", "phrases": [...], "members": [email ids]}
"""

import json
import random
from collections import defaultdict

import numpy as np

from train.common import norm_subject, sha


def folder_sets(emails, topics, grok, max_subj=20):
    by = defaultdict(list)
    for e in emails:
        for l in e.get("topics", []):
            by[l].append(e)
    out = []
    for l, es in by.items():
        ph = [l]
        for e in es:
            ph += grok.get(e["id"], {}).get("topics", [])
        subs = list(dict.fromkeys(norm_subject(e["subject"]) for e in es if len(norm_subject(e["subject"])) >= 6))
        random.Random(l).shuffle(subs)
        out.append({"name": l, "src": "folder", "split": topics.get(l, {}).get("split", "train"),
                    "phrases": list(dict.fromkeys(ph)), "subjects": subs[:max_subj],
                    "members": [e["id"] for e in es]})
    return out


def grok_sets(emails, grok, embed_fn, n_micro=1500, dist=0.25, min_members=8, heldout_frac=0.15, max_phrases=60000):
    """embed_fn(list[str]) -> (n, d) unit vectors (query side)."""
    from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

    ph2ids = defaultdict(set)
    for e in emails:
        for p in grok.get(e["id"], {}).get("topics", []):
            ph2ids[p.lower().strip()].add(e["id"])
    phrases = sorted(ph2ids)[:max_phrases]
    if len(phrases) < 50:
        return []
    V = embed_fn(phrases)
    km = MiniBatchKMeans(n_clusters=min(n_micro, len(phrases) // 4), random_state=0, n_init=1, batch_size=4096).fit(V)
    C = km.cluster_centers_
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    agg = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=dist).fit(C)
    lab = agg.labels_[km.labels_]
    groups = defaultdict(list)
    for p, g in zip(phrases, lab):
        groups[g].append(p)
    out = []
    for g, ps in groups.items():
        mem = set().union(*(ph2ids[p] for p in ps))
        if len(mem) < min_members or len(mem) > 3000:
            continue
        name = max(ps, key=lambda p: len(ph2ids[p]))
        split = "heldout" if int(sha("g:" + name), 16) % 100 < heldout_frac * 100 else "train"
        out.append({"name": name, "src": "grok", "split": split, "phrases": ps, "subjects": [], "members": sorted(mem)})
    return out


def load_grok(path):
    if not path.exists():
        return {}
    out = {}
    for l in open(path):
        r = json.loads(l)
        out[r["id"]] = r
    return out
