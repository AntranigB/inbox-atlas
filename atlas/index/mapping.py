"""2D map + clusters: UMAP (cosine) + HDBSCAN, labeled by TF-IDF of subjects (optionally Grok).

Tiny corpora (under SMALL_N emails, e.g. the 24 email fixture) use PCA + agglomerative
clustering instead, so tests stay fast and nothing crashes.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from atlas import config

SMALL_N = 200
EXTRA_STOP = {"re", "fwd", "fw", "your", "you", "new", "re:", "the", "for", "and", "is", "are", "has", "have",
              "now", "just", "get", "out", "up", "here", "this", "that", "with", "from", "our", "we", "will", "can",
              "email", "update", "reminder", "notification", "please", "via", "dear", "hi"}


def _pca2(E):
    X = E - E.mean(0)
    if len(X) < 2:
        return np.zeros((len(X), 2))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:2].T
    if Y.shape[1] < 2:
        Y = np.hstack([Y, np.zeros((len(Y), 2 - Y.shape[1]))])
    return Y


def layout(E, verbose=False):
    """Return (xy (n,2), labels (n,)) cluster ids, -1 = noise."""
    n = len(E)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, int)
    if n < SMALL_N:
        xy = _pca2(E)
        if n < 4:
            return xy, np.zeros(n, int)
        from sklearn.cluster import AgglomerativeClustering

        k = max(2, min(12, n // 4))
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(E)
        return xy, lab
    import hdbscan
    import umap

    if verbose:
        print(f"  umap on {n} points")
    xy = umap.UMAP(n_components=2, n_neighbors=15, metric="cosine", min_dist=0.05,
                   random_state=42, low_memory=True).fit_transform(E)
    if verbose:
        print("  hdbscan")
    mcs = max(8, n // 150)
    lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=max(3, mcs // 3)).fit_predict(xy)
    return xy, lab


def tfidf_labels(subjects_by_cluster, top=3):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    keys = list(subjects_by_cluster)
    docs = [" ".join(subjects_by_cluster[k]) for k in keys]
    stop = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
    try:
        vec = TfidfVectorizer(stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,}\b",
                              sublinear_tf=True, max_df=0.6 if len(docs) > 4 else 1.0)
        X = vec.fit_transform(docs)
    except ValueError:
        return {k: "misc" for k in keys}
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for r, k in enumerate(keys):
        row = X[r].toarray().ravel()
        idx = [i for i in np.argsort(-row)[:top] if row[i] > 0]
        out[k] = " / ".join(vocab[idx]) if idx else "misc"
    return out


def _grok_label(subjects, cache):
    import httpx

    key = hashlib.sha1("\n".join(subjects).encode()).hexdigest()
    if key in cache:
        return cache[key]
    prompt = ("These are email subjects from one cluster of a personal inbox. Give a short topic label "
              "(2 to 4 words, Title Case, no quotes, no punctuation at the end) describing what they share.\n\n"
              + "\n".join(f"- {s}" for s in subjects))
    r = httpx.post(f"{config.XAI_BASE}/chat/completions", timeout=30,
                   headers={"Authorization": f"Bearer {config.XAI_API_KEY}"},
                   json={"model": config.GROK_MODEL, "temperature": 0.2, "max_tokens": 20,
                         "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    label = r.json()["choices"][0]["message"]["content"].strip().strip('"').strip()
    label = re.sub(r"[–—]", "-", label)[:40]
    cache[key] = label
    return label


def grok_labels(subjects_by_cluster, fallback):
    path = config.INDEX_DIR / "grok_labels.json"
    try:
        cache = json.loads(path.read_text())
    except Exception:
        cache = {}
    out = dict(fallback)

    def one(k):
        try:
            return k, _grok_label(subjects_by_cluster[k][:15], cache)
        except Exception:
            return k, None

    with ThreadPoolExecutor(8) as ex:
        for k, lab in ex.map(one, list(subjects_by_cluster)):
            if lab:
                out[k] = lab
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))
    return out


def build_map(conn, ids, E, grok=False, verbose=False):
    xy, lab = layout(E, verbose=verbose)
    if len(xy):
        lo, hi = xy.min(0), xy.max(0)
        xy = (xy - lo) / np.maximum(hi - lo, 1e-9)  # normalize to [0, 1]
    subj = {r[0]: (r[1] or "") for r in conn.execute("select id, subject from emails")}
    by_cluster = {}
    for i, c in zip(ids, lab):
        if c >= 0:
            by_cluster.setdefault(int(c), []).append(subj.get(i, ""))
    labels = tfidf_labels(by_cluster) if by_cluster else {}
    if grok and config.XAI_API_KEY and by_cluster:
        if verbose:
            print(f"  labeling {len(by_cluster)} clusters with Grok")
        labels = grok_labels(by_cluster, labels)
    points = [{"id": i, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "cluster": int(c)}
              for i, p, c in zip(ids, xy, lab)]
    clusters = sorted(({"id": k, "label": labels.get(k, "misc"), "size": len(v)} for k, v in by_cluster.items()),
                      key=lambda c: -c["size"])
    return {"points": points, "clusters": clusters}
