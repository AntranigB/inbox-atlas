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


def _auth():
    """The live API may require ATLAS_TOKEN; read it from the env or .env without printing it."""
    import os

    tok = os.environ.get("ATLAS_TOKEN")
    if tok is None and (ROOT / ".env").exists():
        for line in open(ROOT / ".env"):
            if line.startswith("ATLAS_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}
OUT = ROOT / "assets"
RED, BLUE, GRAY, INK, MUTED = "#b31b1b", "#4c72b0", "#a0a4aa", "#1f2328", "#6e7781"
MODEL_COLORS = {"bm25": GRAY, "base-bge": BLUE, "atlas-embed": RED, "atlas-embed@256": "#d9706f", "atlas-embed@64": "#efb8b7"}
MODEL_NAMES = {"bm25": "BM25", "base-bge": "base bge (frozen)", "atlas-embed": "atlas-embed 768", "atlas-embed@256": "atlas-embed 256",
               "atlas-embed@64": "atlas-embed 64"}

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
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, model in enumerate(models):
        xs = [t - 0.4 + w * (i + 0.5) for t in range(len(tasks))]
        ys = [metrics[t].get(model, 0) for t in tasks]
        bars = ax.bar(xs, ys, w * 0.9, color=MODEL_COLORS.get(model, RED), label=MODEL_NAMES.get(model, model))
        if len(models) <= 3:
            ax.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED, padding=2)
    labels = {"topic_seen": "topic\nseen folders", "topic_heldthread": "topic\nheld-out threads",
              "topic_heldfolder": "topic\nheld-out folders", "subject_heldthread": "subject\nto body",
              "grok_query_test": "specific\nquery", "grok_vague_test": "vague\nquery"}
    ax.set_xticks(range(len(tasks)), [labels.get(t, t) for t in tasks], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.set_ylim(0, max(max(v.values()) for v in metrics.values()) * 1.15)
    ax.set(ylabel="nDCG@10", title="Enron retrieval on unseen emails")
    ax.legend(frameon=False, ncol=len(models), loc="upper left", fontsize=8.5)
    save(fig, "eval_enron.png")


def fig_search(api):
    """Three ranked lists side by side, drawn as rows so every label sits inside its own row."""
    from matplotlib.patches import FancyBboxPatch

    try:
        res = {m: httpx.post(f"{api}/api/search", json={"query": "coding competition", "mode": m, "k": 8},
                             headers=_auth(), timeout=60).raise_for_status().json() for m in ("keyword", "embed", "region")}
    except httpx.HTTPError as e:
        print("skip search figure:", e)
        return
    fixture = {json.loads(l)["id"]: json.loads(l)["topic"] for l in open(ROOT / "tests/fixtures/mailbox.jsonl")}
    titles = {"keyword": "Keyword (BM25)", "embed": "Embedding (cosine)", "region": "Region (Grok facets + z)"}
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    for c, (mode, r) in enumerate(res.items()):
        hits = r.get("hits", [])[:8]
        good = sum(fixture.get(h["id"]) in ("hackathon", "contest") for h in hits)
        ax.text(c + 0.5, 9.15, titles[mode], ha="center", fontsize=11, weight="bold", color=INK)
        ax.text(c + 0.5, 8.7, f"{good} of 8 on topic", ha="center", fontsize=9, color=RED if good >= 6 else MUTED)
        for i in range(8):
            y = 7.75 - i
            if i >= len(hits):
                ax.add_patch(FancyBboxPatch((c + 0.04, y), 0.92, 0.78, boxstyle="round,pad=0,rounding_size=0.06",
                                            fc="none", ec="#d8dee4", lw=0.8, ls=(0, (3, 3))))
                continue
            h = hits[i]
            topic = fixture.get(h["id"])
            job = "HackerRank" in (h["from"] or "")
            fc = RED if topic in ("hackathon", "contest") else (INK if job else "#eef0f2")
            tc = "white" if topic in ("hackathon", "contest") or job else INK
            ax.add_patch(FancyBboxPatch((c + 0.04, y), 0.92, 0.78, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=fc, ec="none"))
            ax.text(c + 0.08, y + 0.39, f"{i + 1}", va="center", fontsize=8.5, color=tc, weight="bold")
            subj = h["subject"] if len(h["subject"]) <= 36 else h["subject"][:35] + "..."
            ax.text(c + 0.15, y + 0.39, subj, va="center", fontsize=8.5, color=tc)
        if not hits:
            ax.text(c + 0.5, 4.2, "no keyword match", ha="center", color=MUTED)
    ax.text(1.5, 0.05, "red: contest or hackathon     black: HackerRank job assessment (the trap)     gray: other",
            ha="center", fontsize=8.5, color=MUTED)
    fig.suptitle('Top 8 for "coding competition" on the demo inbox', fontsize=12, color=INK, y=0.99)
    save(fig, "search_modes.png")


def fig_related(api):
    topics = json.load(open(ROOT / "eval/queries.json"))["related"]
    rows = []
    try:
        for kind in ("present", "absent"):
            for t in topics[kind]:
                r = httpx.get(f"{api}/api/related", params={"topic": t}, headers=_auth(), timeout=60).raise_for_status().json()
                rows.append((t, kind, r["max_z"], r["related"]))
    except httpx.HTTPError as e:
        print("skip related figure:", e)
        return
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(8, 6))
    ys = range(len(rows))
    ax.barh(ys, [r[2] for r in rows], color=[RED if r[1] == "present" else GRAY for r in rows], height=0.7)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(3.0, color=INK, lw=1.2, ls="--")
    for y, r in zip(ys, rows):
        if r[3] != (r[1] == "present"):
            ax.text(max(r[2], 0) + 0.15, y, "wrong", va="center", fontsize=8, color=RED, weight="bold")
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    acc = sum(r[3] == (r[1] == "present") for r in rows)
    ax.set_xlabel("strongest hub-corrected z in the inbox")
    ax.set_title(f'"Is this topic in my inbox?"  {acc} of {len(rows)} correct', fontsize=11, color=INK, loc="left")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label="topic is in the inbox"), Patch(color=GRAY, label="topic is not"),
                       Line2D([], [], color=INK, ls="--", label="related threshold z = 3")],
              frameon=False, loc="lower right", fontsize=8.5)
    save(fig, "related_calibration.png")


def fig_architecture():
    """Embedding and region architecture. Gray hatched = frozen, red = trained."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 72)
    ax.axis("off")
    FROZEN, TRAIN, DATA = "#e9ecef", "#f6d5d5", "white"

    def box(x, y, w, h, text, kind="data", size=9, weight="normal"):
        fc = {"frozen": FROZEN, "train": TRAIN, "data": DATA, "loss": "white"}[kind]
        ec = {"frozen": "#8c959f", "train": RED, "data": "#8c959f", "loss": INK}[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec,
                                    lw=1.6 if kind == "loss" else 1.2, hatch="////" if kind == "frozen" else None))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight,
                bbox=dict(fc=fc, ec="none", pad=0.5) if kind == "frozen" else None)

    def arrow(x1, y1, x2, y2, color=INK, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color,
                                     lw=1.2, ls=ls, shrinkA=0, shrinkB=0))

    # panel titles
    ax.text(2, 69.5, "Stage A: atlas-embed", fontsize=13, weight="bold", color=INK)
    ax.text(2, 66.6, "frozen bge-base backbone, trainable LoRA, Matryoshka heads, frozen teacher", fontsize=9,
            color=MUTED)
    ax.text(84, 69.5, "Stage B: atlas-region", fontsize=13, weight="bold", color=INK)
    ax.text(84, 66.6, "Set Transformer over Grok facets, outputs a calibrated region", fontsize=9, color=MUTED)
    ax.plot([80, 80], [4, 70], color="#d0d7de", lw=1)

    # inputs
    box(2, 55, 22, 7, "query side\nfacet / Grok query\n+ bge query prefix", "data", 8.5)
    box(27, 55, 22, 7, "doc side\nsubject + sender\n+ cleaned body", "data", 8.5)
    arrow(13, 55, 22, 50.5)
    arrow(38, 55, 31, 50.5)

    # frozen backbone with LoRA
    ax.add_patch(FancyBboxPatch((4, 22), 46, 28, boxstyle="round,pad=0,rounding_size=1.5", fc=FROZEN,
                                ec="#8c959f", lw=1.2, hatch="////"))
    ax.text(27, 47.6, "bge-base-en-v1.5   FROZEN   110M params", ha="center", fontsize=9.5, weight="bold",
            color=INK, bbox=dict(fc=FROZEN, ec="none", pad=0.6))
    for i in range(4):
        y = 41 - i * 4.6
        label = "transformer layer " + ("12" if i == 0 else "11" if i == 1 else "..." if i == 2 else "1")
        ax.add_patch(FancyBboxPatch((7, y), 26, 3.4, boxstyle="round,pad=0,rounding_size=0.6", fc="white",
                                    ec="#8c959f", lw=0.9))
        ax.text(20, y + 1.7, label, ha="center", va="center", fontsize=8, color=INK)
        ax.add_patch(FancyBboxPatch((35, y), 12.5, 3.4, boxstyle="round,pad=0,rounding_size=0.6", fc=TRAIN,
                                    ec=RED, lw=1.1))
        ax.text(41.25, y + 1.7, "LoRA r=16", ha="center", va="center", fontsize=7.5, color=RED, weight="bold")
        ax.plot([33, 35], [y + 1.7, y + 1.7], color=RED, lw=1)
    ax.text(27, 23.6, "LoRA on attention q, k, v, o:  W x + B A x,  1.18M trainable", ha="center", fontsize=8.5,
            color=RED, bbox=dict(fc=FROZEN, ec="none", pad=0.5))

    # pooling and matryoshka
    arrow(27, 22, 27, 19)
    box(17, 14.5, 20, 4.5, "CLS token, L2 normalize", "data", 8.5)
    arrow(27, 14.5, 27, 12)
    for i, (dim, w) in enumerate([(768, 40), (256, 22), (64, 9)]):
        ax.add_patch(FancyBboxPatch((7, 8.4 - i * 2.6), w, 2.0, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=RED, ec="none", alpha=0.35 + 0.25 * i))
        ax.text(7 + w + 1, 9.4 - i * 2.6, f"{dim} d", va="center", fontsize=8, color=INK)
    ax.text(37.5, 5.2, "Matryoshka: every prefix\nis a usable embedding", fontsize=8, color=MUTED, va="center")

    # teacher
    box(55, 49, 22, 13, "bge-reranker-v2-m3\ncross-encoder\nFROZEN teacher", "frozen", 9, "bold")
    arrow(49, 58.5, 55, 57)
    ax.text(66, 46.5, "scores (query, candidate)\nfor 32 mined candidates", fontsize=8, color=MUTED, ha="center",
            va="top")

    # losses
    box(55, 30, 22, 9, "KL( teacher || student )\nlistwise distillation", "loss", 8.5)
    box(55, 13, 22, 9, "InfoNCE per dim\nin-batch + hard negatives", "loss", 8.5)
    arrow(66, 42.5, 66, 39, color=MUTED, ls="--")
    arrow(53, 9.4, 55, 17.5, color=INK)
    arrow(53, 9.4, 55, 34.5, color=INK)
    ax.text(66, 26, "teacher also drops\nfalse negatives", fontsize=7.5, color=MUTED, ha="center", va="center")

    # region side
    chips = [("hackathon", 1), ("Codeforces round", 1), ("ICPC regional", 1), ("not: job interview", 0)]
    for i, (c, pos) in enumerate(chips):
        x = 84 + (i % 2) * 22
        y = 59 - (i // 2) * 4.2
        ax.add_patch(FancyBboxPatch((x, y), 20, 3.2, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc="white", ec=RED if pos else INK, lw=1.1, ls="-" if pos else "--"))
        ax.text(x + 10, y + 1.6, c, ha="center", va="center", fontsize=8, color=RED if pos else INK)
    ax.text(106, 63.5, "Grok facet set (1 to 8 positive, 0 to 3 negative)", ha="center", fontsize=8, color=MUTED)
    arrow(106, 54.5, 106, 51.5)
    box(88, 45.5, 36, 6, "atlas-embed (frozen) + type embedding", "frozen", 8.5)
    arrow(106, 45.5, 106, 42.5)
    box(92, 36.5, 28, 6, "ISAB x2  (16 inducing points)", "train", 9)
    arrow(106, 36.5, 106, 33.5)
    box(92, 27.5, 28, 6, "PMA  (K = 4 seeds)", "train", 9)
    arrow(106, 27.5, 106, 24.5)
    box(84, 17, 44, 7.5, "region: anchors a1..a4, temperatures tau_k, bias b", "data", 8.5)
    arrow(106, 17, 106, 14)
    box(84, 6, 44, 8, "m(e) = logsumexp_k( tau_k cos(e, a_k) ) - b\nP(member) = sigmoid( (m - shift) / T )", "loss",
        8.5)
    ax.text(106, 2.6, "BCE on members vs nearest non-members, negative-phrase term, teacher KL;\n"
            "T and shift fit on validation at the real 0.1% base rate", ha="center", fontsize=7.5, color=MUTED)

    # legend
    for i, (label, kind) in enumerate([("frozen", "frozen"), ("trained", "train"), ("loss / output", "loss")]):
        x = 2 + i * 17
        ax.add_patch(FancyBboxPatch((x, 0.8), 3.5, 2.2, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc={"frozen": FROZEN, "train": TRAIN, "loss": "white"}[kind],
                                    ec={"frozen": "#8c959f", "train": RED, "loss": INK}[kind],
                                    hatch="////" if kind == "frozen" else None))
        ax.text(x + 4.5, 1.9, label, va="center", fontsize=8.5, color=INK)
    save(fig, "architecture.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "data/runs"))
    ap.add_argument("--api", default="http://localhost:8765")
    a = ap.parse_args()
    fig_architecture()
    fig_training(Path(a.runs))
    fig_eval(Path(a.runs))
    fig_search(a.api)
    fig_related(a.api)
