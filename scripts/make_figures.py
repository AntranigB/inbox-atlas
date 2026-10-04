"""Render README figures from real artifacts only: training logs, eval metrics, and the live API.

Usage: uv run --extra figures python scripts/make_figures.py [--runs data/runs] [--api http://localhost:8765]
Figures that lack their inputs are skipped, never faked.
"""

import argparse
import json
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
RED, BLUE, GRAY, INK, MUTED = "#b31b1b", "#4c72b0", "#a0a4aa", "#1f2328", "#6e7781"
MODEL_COLORS = {"bm25": GRAY, "base-bge": BLUE}
MODEL_NAMES = {"bm25": "BM25", "base-bge": "base bge (frozen)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip().startswith("{")]


def fig_training(runs):
    logs = {r: read_jsonl(runs / r / "log.jsonl") for r in ("a1", "a2") if (runs / r / "log.jsonl").exists()}
    if not logs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for run, rows in logs.items():
        steps = [r for r in rows if "ema" in r]
        c = BLUE if run == "a1" else RED
        ax1.plot([r["step"] for r in steps], [r["ema"] for r in steps], color=c, lw=1.8,
                 label=f"{run}: total loss (EMA)")
        if any(r.get("kd") for r in steps):
            ax1.plot([r["step"] for r in steps], [r["kd"] for r in steps], color=c, lw=1, ls="--", alpha=0.7,
                     label=f"{run}: teacher KL term")
        vals = [r for r in rows if "val" in r]
        for key, ls in (("subject", "-"), ("topic", ":")):
            ax2.plot([r["step"] for r in vals], [r["val"][key]["ndcg@10"] for r in vals], color=c, ls=ls,
                     marker="o", ms=3, lw=1.5, label=f"{run}: {key}")
    ax1.set(xlabel="step", ylabel="loss", title="Stage A training loss")
    ax2.set(xlabel="step", ylabel="val nDCG@10", title="Validation retrieval")
    ax1.legend(frameon=False, fontsize=8)
    ax2.legend(frameon=False, fontsize=8)
    save(fig, "training_curves.png")


def fig_eval(runs):
    metrics = {}
    for p in sorted(runs.glob("eval*/metrics.json")):
        for task, by_model in json.load(open(p)).items():
            if task.startswith("region"):
                continue
            for model, m in by_model.items():
                metrics.setdefault(task, {})[model] = m["ndcg@10"]
    if not metrics:
        return
    tasks = list(metrics)
    models = list(dict.fromkeys(m for t in tasks for m in metrics[t]))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, model in enumerate(models):
        xs = [t + i * w for t in range(len(tasks))]
        ys = [metrics[t].get(model, 0) for t in tasks]
        ax.bar(xs, ys, w * 0.92, color=MODEL_COLORS.get(model, RED), label=MODEL_NAMES.get(model, model))
    labels = {"topic_seen": "topic\nseen", "topic_heldthread": "topic\nheld-out threads",
              "topic_heldfolder": "topic\nheld-out folders", "subject_heldthread": "subject to body",
              "grok_query_test": "specific\nquery", "grok_vague_test": "vague\nquery"}
    ax.set_xticks([t + 0.4 - w / 2 for t in range(len(tasks))], [labels.get(t, t) for t in tasks], fontsize=8)
    ax.set(ylabel="nDCG@10", title="Enron retrieval, unseen emails")
    ax.legend(frameon=False, ncol=len(models))
    save(fig, "eval_enron.png")


def fig_search(api):
    try:
        res = {m: httpx.post(f"{api}/api/search", json={"query": "coding competition", "mode": m, "k": 8},
                             timeout=60).json() for m in ("keyword", "embed", "region")}
    except httpx.HTTPError as e:
        print("skip search figure:", e)
        return
    fixture = {json.loads(l)["id"]: json.loads(l)["topic"] for l in open(ROOT / "tests/fixtures/mailbox.jsonl")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.subplots_adjust(wspace=0.95)
    titles = {"keyword": "Keyword (BM25)", "embed": "Embedding (cosine)", "region": "Region (Grok facets + z)"}
    for ax, (mode, r) in zip(axes, res.items()):
        hits = r.get("hits", [])[:8]
        ys = [7 - i for i in range(len(hits))]  # rank 1 on top, empty slots stay empty
        colors = [RED if fixture.get(h["id"]) in ("hackathon", "contest")
                  else (INK if "HackerRank" in (h["from"] or "") else GRAY) for h in hits]
        ax.barh(ys, [8 - i for i in range(len(hits))], color=colors)
        ax.set_yticks(ys, [h["subject"][:34] for h in hits], fontsize=7.5)
        ax.set_xticks([])
        ax.set_xlim(0, 8.4)
        ax.set_ylim(-0.6, 7.6)
        ax.set_title(f"{titles[mode]}: {sum(c == RED for c in colors)}/8 relevant", fontsize=10)
        if not hits:
            ax.text(0.5, 0.5, "no keyword match", ha="center", va="center", transform=ax.transAxes, color=MUTED)
    fig.suptitle('"coding competition" on the demo inbox (red: contest or hackathon, black: job assessment)',
                 fontsize=10, color=INK)
    save(fig, "search_modes.png")


def fig_related(api):
    topics = json.load(open(ROOT / "eval/queries.json"))["related"]
    rows = []
    try:
        for kind in ("present", "absent"):
            for t in topics[kind]:
                r = httpx.get(f"{api}/api/related", params={"topic": t}, timeout=60).json()
                rows.append((t, kind, r["max_z"], r["related"]))
    except httpx.HTTPError as e:
        print("skip related figure:", e)
        return
    fig, ax = plt.subplots(figsize=(10, 3.4))
    xs = range(len(rows))
    ax.bar(xs, [r[2] for r in rows], color=[RED if r[1] == "present" else GRAY for r in rows])
    ax.axhline(3.0, color=INK, lw=1, ls="--")
    ax.text(9.5, 3.15, "related threshold z = 3", ha="center", fontsize=8, color=INK)
    for x, r in zip(xs, rows):
        ok = r[3] == (r[1] == "present")
        ax.text(x, max(r[2], 0) + 0.25, "ok" if ok else "miss", ha="center", fontsize=7, color=INK if ok else RED)
    ax.set_xticks(list(xs), [r[0] for r in rows], rotation=55, ha="right", fontsize=7.5)
    acc = sum(r[3] == (r[1] == "present") for r in rows)
    ax.set(ylabel="max hub-corrected z", title=f'"Is this topic in my inbox?" {acc}/{len(rows)} correct '
                                                 "(red: present, gray: absent)")
    save(fig, "related_calibration.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "data/runs"))
    ap.add_argument("--api", default="http://localhost:8765")
    a = ap.parse_args()
    fig_training(Path(a.runs))
    fig_eval(Path(a.runs))
    fig_search(a.api)
    fig_related(a.api)
