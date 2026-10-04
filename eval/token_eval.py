"""Token efficiency for LLM agents: how much context each retrieval strategy hands the agent, and
whether the agent can still answer from it.

Strategies (all over the same store of emails + Obsidian sections):
  keyword top-10 docs      grep-style: FTS over the question's words, read the top 10 documents whole
  embedding top-10 docs    standard RAG: cosine over whole documents, read the top 10 whole
  embedding top-8 chunks   chunked RAG: ~120 word chunks, top 8 by cosine
  atlas pack @300/800/1500 atlas.context.build_context at three budgets

Quality: Grok (config.GROK_MODEL) answers from ONLY that context; a separate Grok judge call scores the
answer 0/1 against a reference answer built from the documents judged relevant (map: per document facts,
reduce: answer from those documents in full) out of the union every strategy retrieved. Every Grok call is cached in <data>/cache/token_eval.json.

    ATLAS_DATA=data/eval_demo uv run python -m eval.token_eval              # demo inbox + synthetic vault
    uv run python -m eval.token_eval --private data/brain_questions.json  # real vault: tokens only, no Grok

--private mode never sends anything to Grok (no expansion, no answers, no judge) and writes only
aggregate numbers: questions and note titles stay out of the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from atlas import config, store
from atlas.context import pack
from atlas.search.engine import get_engine

HERE = Path(__file__).parent
BUDGETS = (300, 800, 1500)
TOP_DOCS, TOP_CHUNKS, CHUNK_WORDS = 10, 8, 120
STOP = set("a an the of to in on for and or is are was were be been i me my mine you your we our it its this that "
           "what when where who whom which why how do does did can could should would will with about from at by as "
           "any anything there have has had am so up out if not no".split())

TODAY = "2026-10-04"  # fixed so cached answers stay valid; the demo inbox is dated Sep to Oct 2026
ANSWER = ("You answer a request about the user's own email and notes using ONLY the context below. The request may be "
           "a question or just a topic; for a topic, report the specific facts the context has about it (what, who, "
           "when, amounts). Set found=false only if nothing in the context is about the request. Keep the answer "
           "concise (at most 4 sentences). Today is " + TODAY + ".\n\nRequest: {q}\n\nContext:\n{ctx}\n\n"
           "Request again: {q}\n"
           'Return ONLY JSON: {{"found": true or false, "answer": "<answer, empty when found is false>"}}')
EXTRACT = ("Does this one document (an email or a note of the user) contain information about the request? If yes, "
           "list the relevant facts briefly. Today is " + TODAY + ".\n\nRequest: {q}\n\nDocument:\n{doc}\n\n"
           "Request again: {q}\n"
           'Return ONLY JSON: {{"found": true or false, "facts": "<relevant facts, empty when found is false>"}}')
JUDGE = ("You grade an answer against a reference answer for the same question about a person's email and notes.\n"
         "Question: {q}\nReference answer: {ref}\nCandidate answer: {cand}\n\n"
         "Rules: correct=1 if the candidate states the key facts of the reference and does not contradict it. If the "
         "reference lists several items, the candidate must give most of the important ones. Ignore wording and "
         "length; extra details (a year, extra items) are fine unless they contradict the reference.\n"
         'Return ONLY JSON: {{"correct": 0 or 1, "why": "<short reason>"}}')
NOT_FOUND = "NOT FOUND"


# ---------- documents ----------

class Corpus:
    """Documents an agent could open: one per email, one per Obsidian note (all its sections)."""

    def __init__(self, eng):
        self.eng, conn = eng, eng.conn
        rows = [dict(r) for r in conn.execute("select * from emails order by rowid")]
        self.row = {r["id"]: r for r in rows}
        self.doc_of, self.docs = {}, {}
        for r in rows:
            d = r["thread_id"] if r.get("source") == "obsidian" else r["id"]
            self.doc_of[r["id"]] = d
            self.docs.setdefault(d, []).append(r)
        self.ids = list(self.docs)
        self.kind = {d: pack.hybrid.source_kind(rs[0].get("source")) for d, rs in self.docs.items()}
        self.text = {d: (pack.full_note_text(conn, d) if self.kind[d] == "obsidian" else pack.doc_text(rs[0]))
                     for d, rs in self.docs.items()}
        enc = eng.enc
        self.D = np.asarray(enc.encode_docs([self.text[d][:4000] for d in self.ids]), np.float32)
        self.chunks = []
        for d in self.ids:
            for c in chunk_words(self.text[d]):
                self.chunks.append((d, c))
        self.C = np.asarray(enc.encode_docs([c for _, c in self.chunks]), np.float32)

    def mask(self, sources):
        return np.array([self.kind[d] in sources for d in self.ids])


def chunk_words(text, n=CHUNK_WORDS):
    out, cur = [], []
    for para in re.split(r"\n\s*\n", text):
        w = para.split()
        if cur and len(cur) + len(w) > n:
            out.append(" ".join(cur))
            cur = []
        cur += w
        while len(cur) > n:
            out.append(" ".join(cur[:n]))
            cur = cur[n:]
    if cur:
        out.append(" ".join(cur))
    return out


def keyword_terms(q):
    return " ".join(t for t in re.findall(r"[A-Za-z0-9]+", q) if t.lower() not in STOP)


def strat_keyword(cp, q, sources):
    seen = []
    for rid, _ in store.fts_search(cp.eng.conn, keyword_terms(q), 400):
        d = cp.doc_of.get(rid)
        if d and d not in seen and cp.kind[d] in sources:
            seen.append(d)
        if len(seen) >= TOP_DOCS:
            break
    return seen, "\n\n---\n\n".join(cp.text[d] for d in seen)


def strat_embed_docs(cp, q, sources):
    qv = cp.eng.enc.encode_queries([q])[0]
    s = np.where(cp.mask(sources), cp.D @ qv, -np.inf)
    top = [cp.ids[i] for i in np.argsort(-s)[:TOP_DOCS] if np.isfinite(s[i])]
    return top, "\n\n---\n\n".join(cp.text[d] for d in top)


def strat_chunks(cp, q, sources):
    qv = cp.eng.enc.encode_queries([q])[0]
    ok = np.array([cp.kind[d] in sources for d, _ in cp.chunks])
    s = np.where(ok, cp.C @ qv, -np.inf)
    top = [i for i in np.argsort(-s)[:TOP_CHUNKS] if np.isfinite(s[i])]
    return list(dict.fromkeys(cp.chunks[i][0] for i in top)), "\n\n---\n\n".join(cp.chunks[i][1] for i in top)


def strat_atlas(cp, q, sources, budget, use_grok=True):
    p = pack.build_context(q, budget_tokens=budget, sources=sources, engine=cp.eng, use_grok=use_grok)
    docs = list(dict.fromkeys(cp.doc_of.get(("obs:" + it["uri"]) if it["source"] == "obsidian" else it["uri"][6:])
                              for it in p["items"]))
    return [d for d in docs if d], p["context"], p


STRATS = ["keyword top-10 docs", "embedding top-10 docs", f"embedding top-{TOP_CHUNKS} chunks"] + \
         [f"atlas pack @{b}" for b in BUDGETS]


def contexts(cp, q, sources, use_grok=True):
    out = {}
    out[STRATS[0]] = strat_keyword(cp, q, sources)
    out[STRATS[1]] = strat_embed_docs(cp, q, sources)
    out[STRATS[2]] = strat_chunks(cp, q, sources)
    for b in BUDGETS:
        docs, ctx, p = strat_atlas(cp, q, sources, b, use_grok)
        out[f"atlas pack @{b}"] = (docs, ctx, p)
    return out


# ---------- grok, cached ----------

class Cache:
    def __init__(self):
        self.path = config.DATA / "cache" / "token_eval.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.lock = threading.Lock()

    def call(self, prompt, json_mode=False):
        from atlas.agent import grok

        key = hashlib.sha1(f"{config.GROK_MODEL}|{json_mode}|{prompt}".encode()).hexdigest()
        with self.lock:
            if key in self.d:
                return self.d[key]
        for attempt in range(4):
            try:
                msg = grok.chat([{"role": "user", "content": prompt}], json_mode=json_mode, temperature=0, timeout=120)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
        out = (msg.get("content") or "").strip()
        with self.lock:
            self.d[key] = out
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.d))
        return out


def _not_found(ans: str) -> bool:
    return ans == NOT_FOUND


def _ask(cache, prompt, key):
    """JSON answer -> text, or NOT_FOUND when the model says found=false."""
    from atlas.agent.grok import _parse_json

    try:
        v = _parse_json(cache.call(prompt, json_mode=True))
    except Exception:
        return NOT_FOUND
    text = str(v.get(key) or "").strip()
    return text if v.get("found") and text else NOT_FOUND


def grade(cache, q, ctxs, cp):
    from atlas.agent.grok import _parse_json

    pool = list(dict.fromkeys(d for docs, *_ in ctxs.values() for d in docs))
    # map: pull the relevant facts out of each pooled document on its own (long mixed contexts made
    # the reference miss things), reduce: answer from the relevant documents in full
    facts = {d: _ask(cache, EXTRACT.format(q=q, doc=cp.text[d]), "facts") for d in pool}
    relevant = [d for d in pool if not _not_found(facts[d])]
    ref_ctx = "\n\n---\n\n".join(cp.text[d] for d in relevant)
    ref = _ask(cache, ANSWER.format(q=q, ctx=ref_ctx), "answer") if relevant else NOT_FOUND
    res = {}
    for name, (docs, ctx, *_) in ctxs.items():
        ans = _ask(cache, ANSWER.format(q=q, ctx=ctx or "(empty)"), "answer")
        ref_nf, ans_nf = _not_found(ref), _not_found(ans)
        if ref_nf or ans_nf:  # deterministic when either side abstains
            ok = int(ref_nf and ans_nf)
        else:
            try:
                v = _parse_json(cache.call(JUDGE.format(q=q, ref=ref, cand=ans), json_mode=True))
                ok = int(v.get("correct", 0))
            except Exception:
                ok = 0
        res[name] = {"tokens": pack.count_tokens(ctx), "correct": ok, "answer": ans}
    return {"reference": ref, "ref_found": not _not_found(ref),
            "pool_docs": len(pool), "relevant_docs": len(relevant), "strategies": res}


# ---------- runs ----------

def run_graded(cp, questions, sources, label, workers=6):
    cache = Cache()

    built = {q: contexts(cp, q, sources) for q in questions}  # encoder calls stay on one thread

    def one(q):
        ctxs = built[q]
        g = grade(cache, q, ctxs, cp)
        g["question"], g["set"] = q, label
        p = ctxs["atlas pack @800"][2]
        g["atlas_answerable"] = p["answerable"]
        g["atlas_naive_tokens"] = p.get("naive_tokens")
        return g

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, questions))
    for g in out:
        print(f"[{label}] {g['question'][:50]:50s} " + " ".join(
            f"{v['tokens']:>5}/{v['correct']}" for v in g["strategies"].values()))
    return out


def run_absent(cp, topics, sources, use_grok=True):
    out = []
    for t in topics:
        ctxs = contexts(cp, t, sources, use_grok)
        row = {"topic": t, "tokens": {n: pack.count_tokens(c[1]) for n, c in ctxs.items()},
               "atlas_abstained": not ctxs["atlas pack @800"][2]["answerable"]}
        out.append(row)
    return out


def summarize(rows):
    s = {}
    for n in STRATS:
        toks = [r["strategies"][n]["tokens"] for r in rows]
        acc = float(np.mean([r["strategies"][n]["correct"] for r in rows])) if rows else 0.0
        mt = float(np.mean(toks)) if toks else 0.0
        s[n] = {"tokens": mt, "median": float(np.median(toks)) if toks else 0.0, "acc": acc,
                "acc_per_1k": acc / (mt / 1000) if mt else 0.0}
    return s


def table(s):
    lines = ["| Strategy | mean context tokens | median | accuracy | accuracy per 1k tokens |", "|---|---|---|---|---|"]
    for n, v in s.items():
        lines.append(f"| {n} | {v['tokens']:.0f} | {v['median']:.0f} | {v['acc']:.0%} | {v['acc_per_1k']:.2f} |")
    return lines


def figure(s, absent, out_png, n_q=24):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RED, BLUE, GRAY, INK, MUTED = "#b31b1b", "#4c72b0", "#a0a4aa", "#1f2328", "#6e7781"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white"})
    color = {STRATS[0]: GRAY, STRATS[1]: BLUE, STRATS[2]: BLUE}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    # label placement: atlas points alternate below/above, document baselines sit left of their marker
    offs = {STRATS[0]: (-9, -13, "right"), STRATS[1]: (-9, 6, "right"), STRATS[2]: (-6, -17, "left"),
            "atlas pack @300": (0, -17, "center"), "atlas pack @800": (0, 10, "center"),
            "atlas pack @1500": (0, 10, "center")}
    for n, v in s.items():
        c = color.get(n, RED)
        mk = "s" if "chunks" in n else "o"
        ax.scatter(v["tokens"], v["acc"] * 100, s=70, color=c, marker=mk, zorder=3, edgecolor="white", lw=0.8)
        dx, dy, ha = offs.get(n, (8, 6, "left"))
        ax.annotate(n.replace("atlas pack ", "atlas "), (v["tokens"], v["acc"] * 100), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=8.5, color=c if c != GRAY else MUTED)
    ax.set_xscale("log")
    ax.set_xlim(min(v["tokens"] for v in s.values()) / 1.6, max(v["tokens"] for v in s.values()) * 1.8)
    ax.set_ylim(0, 105)
    ax.set(xlabel="mean context tokens per question (log)", ylabel="answer accuracy (%)",
           title=f"Accuracy vs context tokens ({n_q} questions)")
    ax.grid(alpha=0.25, lw=0.6)
    names = STRATS
    mean_abs = [np.mean([a["tokens"][n] for a in absent]) if absent else 0 for n in names]
    ys = np.arange(len(names))[::-1]
    ax2.barh(ys, mean_abs, color=[color.get(n, RED) for n in names], height=0.6)
    for y, v in zip(ys, mean_abs):
        ax2.text(v + max(mean_abs) * 0.02, y, f"{v:.0f}", va="center", fontsize=8.5, color=INK)
    ax2.set_yticks(ys, [n.replace("atlas pack ", "atlas ") for n in names], fontsize=8.5)
    ax2.set_xlim(0, max(mean_abs) * 1.18 if max(mean_abs) else 1)
    ax2.set(xlabel="tokens handed to the agent", title=f"Absent topics ({len(absent)}): cost of 'nothing here'")
    fig.tight_layout()
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--out", default=str(HERE / "token_results.md"))
    ap.add_argument("--fig", default=str(config.ROOT / "assets" / "token_efficiency.png"))
    ap.add_argument("--private", default=None, help="JSON {questions:[{q, target}], absent:[...]} for a private vault")
    ap.add_argument("--json", default=None, help="also dump raw per-question results here")
    a = ap.parse_args()
    eng = get_engine(a.encoder)
    cp = Corpus(eng)
    if a.private:
        return run_private(cp, a)
    spec = json.loads((HERE / "queries.json").read_text())
    vq = json.loads((HERE / "token_queries.json").read_text())
    both = pack.SOURCES
    rows = run_graded(cp, spec["queries"], both, "inbox") + run_graded(cp, vq["vault_questions"], both, "vault")
    absent = run_absent(cp, spec["related"]["absent"] + vq["vault_absent"], both)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "absent": absent}, indent=1, default=str))
    s_all = summarize(rows)
    s_in = summarize([r for r in rows if r["set"] == "inbox"])
    s_v = summarize([r for r in rows if r["set"] == "vault"])
    n_email = sum(1 for k in cp.kind.values() if k == "gmail")
    n_note = sum(1 for k in cp.kind.values() if k == "obsidian")
    n_sec = sum(len(cp.docs[d]) for d in cp.docs if cp.kind[d] == "obsidian")
    atl = "atlas pack @800"
    ans_rate = np.mean([r["atlas_answerable"] for r in rows])
    abst = np.mean([a_["atlas_abstained"] for a_ in absent])
    naive = [r["atlas_naive_tokens"] for r in rows if r.get("atlas_naive_tokens")]
    packed = [r["strategies"][atl]["tokens"] for r in rows if r.get("atlas_naive_tokens")]
    L = ["# Token efficiency eval", "",
         f"Store: {n_email} emails (demo inbox) + {n_note} synthetic Obsidian notes ({n_sec} heading sections), "
         f"encoder `{eng.name}`, tokens counted with tiktoken cl100k_base. "
         f"{len(rows)} questions: 12 demo inbox queries (eval/queries.json) and 12 vault questions "
         f"(eval/token_queries.json). Answerer and judge: Grok `{config.GROK_MODEL}`. "
         f"Run {time.strftime('%Y-%m-%d %H:%M')}.", "",
         "Each strategy hands the agent a context; Grok answers from only that context; a separate Grok call grades "
         "the answer 0/1 against a reference answer. The reference comes from the union of documents every strategy "
         "retrieved: Grok checks each document alone for relevant facts, then answers from the full text of the "
         "relevant ones. When either side says not found, scoring is deterministic (correct only if both do).", "",
         "Caveats: 24 questions, so one question is about 4 points. A single Grok model answers, writes the "
         "reference and judges. The demo inbox queries are topic phrases (\"travel plans\"), whose references list "
         "several items; the vault questions ask for one fact. Keyword search ORs the question's content words "
         "over SQLite FTS5, a fair stand-in for grep but not for a tuned BM25 agent.", "",
         "## All 24 questions", ""] + table(s_all) + [
         "", "## Demo inbox queries (12)", ""] + table(s_in) + [
         "", "## Synthetic vault questions (12)", ""] + table(s_v) + [
         "", "## Calibrated 'nothing here'", "",
         f"Atlas abstained (answerable=false, zero items) on {abst:.0%} of {len(absent)} absent topics, and returned "
         f"items for {ans_rate:.0%} of the 24 present questions. Mean tokens handed to the agent for an absent topic:", "",
         "| Strategy | mean tokens on absent topics |", "|---|---|"]
    for n in STRATS:
        L.append(f"| {n} | {np.mean([x['tokens'][n] for x in absent]):.0f} |")
    L += ["", "## Pack vs reading the same documents whole", "",
          f"When the pack answers, it averages {np.mean(packed):.0f} tokens against {np.mean(naive):.0f} tokens for "
          f"the full text of the same top-k documents (`tokens_saved_vs_naive`), a "
          f"{np.mean(naive) / max(np.mean(packed), 1):.1f}x reduction.", "",
          "## Per question (tokens / correct)", "",
          "| set | question | " + " | ".join(STRATS) + " |", "|---|---|" + "---|" * len(STRATS)]
    for r in rows:
        L.append(f"| {r['set']} | {r['question']} | " + " | ".join(
            f"{r['strategies'][n]['tokens']} / {r['strategies'][n]['correct']}" for n in STRATS) + " |")
    L += ["", "## Absent topics (tokens)", "", "| topic | " + " | ".join(STRATS) + " | atlas abstained |",
          "|---|" + "---|" * (len(STRATS) + 1)]
    for x in absent:
        L.append(f"| {x['topic']} | " + " | ".join(str(x["tokens"][n]) for n in STRATS) +
                 f" | {'yes' if x['atlas_abstained'] else 'NO'} |")
    prev = Path(a.out).read_text() if Path(a.out).exists() else ""
    priv = prev.split("<!-- private -->", 1)[1] if "<!-- private -->" in prev else ""
    Path(a.out).write_text("\n".join(L) + "\n" + ("\n<!-- private -->" + priv if priv else ""))
    print("\n".join(table(s_all)))
    figure(s_all, absent, Path(a.fig))


def run_private(cp, a):
    """Real vault: no Grok anywhere. Quality proxy = does the context contain the target note at all."""
    spec = json.loads(Path(a.private).read_text())
    srcs = ("obsidian",)
    res = {n: {"tokens": [], "hit": []} for n in STRATS}
    for item in spec["questions"]:
        q, target = item["q"], item["target"]
        ctxs = contexts(cp, q, srcs, use_grok=False)
        for n, (docs, ctx, *_) in ctxs.items():
            res[n]["tokens"].append(pack.count_tokens(ctx))
            res[n]["hit"].append(int(any(d == f"obs:{target}" for d in docs)))
    absent = run_absent(cp, spec.get("absent", []), srcs, use_grok=False)
    n_note = sum(1 for k in cp.kind.values() if k == "obsidian")
    n_sec = sum(len(cp.docs[d]) for d in cp.docs if cp.kind[d] == "obsidian")
    L = ["<!-- private -->", "", "## Real vault (private, tokens only)", "",
         f"The user's own Obsidian vault: {n_note} notes, {n_sec} heading sections, embedded locally with "
         f"`{cp.eng.name}`. {len(spec['questions'])} questions written from note titles only; questions, titles and "
         "content are not in this repo and nothing was sent to Grok (no expansion, no answers, no judge). Quality "
         "proxy: the target note appears in the context.", "",
         "| Strategy | mean context tokens | median | target note in context | hits per 1k tokens |", "|---|---|---|---|---|"]
    for n, v in res.items():
        mt = np.mean(v["tokens"])
        hit = np.mean(v["hit"])
        L.append(f"| {n} | {mt:.0f} | {np.median(v['tokens']):.0f} | {hit:.0%} | {hit / (mt / 1000) if mt else 0:.2f} |")
    if absent:
        L += ["", f"Absent topics ({len(absent)}): Atlas abstained on "
              f"{np.mean([x['atlas_abstained'] for x in absent]):.0%}. Mean tokens: " + ", ".join(
                  f"{n} {np.mean([x['tokens'][n] for x in absent]):.0f}" for n in STRATS) + "."]
    out = Path(a.out)
    prev = out.read_text().split("<!-- private -->", 1)[0] if out.exists() else ""
    out.write_text(prev.rstrip("\n") + "\n\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
