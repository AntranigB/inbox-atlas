"""Navigate a nested markdown vault by folder, not just by note.

    from atlas.context import tree
    tree.points_of_interest("how did I calibrate the arm servos?")        # which folders and notes
    tree.points_of_interest("grip force", within="projects/robotics")    # drill into a subtree
    tree.related_folders("projects/robotics/arm")                         # folders near this one

Second brains and agent workspaces (Obsidian vaults, Hermes and OpenClaw memory folders) are deep
trees of markdown. Every folder gets a vector: the L2-normalized mean of the section vectors under it,
plus a local version that weights a folder's own notes over its descendants (half per level down).
A question is scored with the same hub-corrected region as atlas.search, the section scores are
aggregated up the tree, and the agent gets the few folders the question lives in, their best notes
and one excerpt each, or a short "not here" so it stops walking the tree.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from atlas import store
from atlas.context import pack
from atlas.search import hybrid
from atlas.search import region as R
from atlas.search.engine import get_engine

ROOT = ""  # the vault root folder
LOCAL_DECAY = 0.5  # weight of a section k levels below a folder in that folder's local vector
TOP_M = 3  # folder score = 0.6 * max z + 0.4 * mean of the top m section z
EXCERPT_TOKENS = 45


def norm_path(path: str | None) -> str:
    return "/".join(p for p in (path or "").replace("\\", "/").split("/") if p and p != ".")


def parent_of(path: str) -> str | None:
    if path == ROOT:
        return None
    return path.rsplit("/", 1)[0] if "/" in path else ROOT


def ancestors(path: str) -> list[str]:
    out, p = [], parent_of(path)
    while p is not None:
        out.append(p)
        p = parent_of(p)
    return out


def is_under(path: str, folder: str) -> bool:
    """True when path is folder itself or anywhere below it."""
    return folder == ROOT or path == folder or path.startswith(folder + "/")


def depth(path: str) -> int:
    return 0 if path == ROOT else path.count("/") + 1


def _label(path: str) -> str:
    return (path + "/") if path else "/"


def _norm(v):
    return v / max(float(np.linalg.norm(v)), 1e-8)


def _day(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d") if ts else None


@dataclass
class Folder:
    path: str
    vec: np.ndarray  # normalized mean of every section in the subtree
    local: np.ndarray  # same, weighted toward the folder's own notes
    n_notes: int
    n_direct_notes: int
    n_sections: int
    last_modified: str | None
    top_tags: list
    links: Counter = field(default_factory=Counter)  # wikilink targets used anywhere in the subtree
    stems: set = field(default_factory=set)  # note names in the subtree (what wikilinks point at)

    def stats(self) -> dict:
        return {"path": _label(self.path), "label": _label(self.path), "n_notes": self.n_notes,
                "n_direct_notes": self.n_direct_notes, "n_sections": self.n_sections,
                "last_modified": self.last_modified, "top_tags": self.top_tags}


@dataclass
class Tree:
    sec_ids: list  # obsidian section ids, in index order
    sec_rows: np.ndarray  # their row in eng.index.E
    sec_folder: list  # folder path of each section
    sec_note: list  # note path (thread) of each section
    folders: dict  # path -> Folder
    mean: np.ndarray | None = None  # mean section vector of the whole vault
    key: tuple = ()

    def folder_mask(self, folder: str) -> np.ndarray:
        return np.array([is_under(f, folder) for f in self.sec_folder], bool)


def _note_dir(note: str) -> str:
    return note.rsplit("/", 1)[0] if "/" in note else ROOT


def build_tree(eng=None) -> Tree:
    eng = eng or get_engine()
    rows = [dict(r) for r in eng.conn.execute(
        "select id, thread_id, date, labels, to_addrs from emails where source='obsidian'")]
    eng.ensure_ids([r["id"] for r in rows])  # sections ingested after the index was built
    pos = eng.index.pos
    rows = [r for r in rows if r["id"] in pos]
    rows.sort(key=lambda r: pos[r["id"]])
    sec_rows = np.array([pos[r["id"]] for r in rows], int)
    notes = [(r["thread_id"] or r["id"])[4:].split("#", 1)[0] for r in rows]
    dirs = [_note_dir(n) for n in notes]
    E = eng.index.E[sec_rows] if len(sec_rows) else np.zeros((0, eng.enc.dim), np.float32)

    paths = set()
    for d in dirs:
        paths.add(d)
        paths.update(ancestors(d))
    folders = {}
    for p in sorted(paths, key=lambda x: (depth(x), x)):
        idx = [i for i, d in enumerate(dirs) if is_under(d, p)]
        if not idx:
            continue
        V = E[idx]
        w = np.array([LOCAL_DECAY ** (depth(dirs[i]) - depth(p)) for i in idx], np.float32)
        note_set = {notes[i] for i in idx}
        tags, links, dates = Counter(), Counter(), []
        seen = set()
        for i in idx:
            r = rows[i]
            dates.append(r["date"] or 0)
            if notes[i] in seen:
                continue  # tags are per note, every section repeats them
            seen.add(notes[i])
            for t in json.loads(r["labels"] or "[]"):
                tags[t.lower()] += 1
        for i in idx:
            for l in json.loads(rows[i]["to_addrs"] or "[]"):
                links[l.strip().lower()] += 1
        folders[p] = Folder(
            path=p, vec=_norm(V.mean(0)), local=_norm((V * w[:, None]).sum(0) / w.sum()),
            n_notes=len(note_set), n_direct_notes=len({notes[i] for i in idx if dirs[i] == p}),
            n_sections=len(idx), last_modified=_day(max(dates) if dates else None),
            top_tags=[t for t, _ in tags.most_common(5)], links=links,
            stems={n.rsplit("/", 1)[-1].removesuffix(".md").lower() for n in note_set})
    return Tree(sec_ids=[r["id"] for r in rows], sec_rows=sec_rows, sec_folder=dirs, sec_note=notes,
                folders=folders, mean=E.mean(0) if len(E) else None)


_cache: dict = {}
_lock = threading.Lock()


def get_tree(eng=None) -> Tree:
    """Folder tree for this engine, rebuilt when the index or the vault changes."""
    eng = eng or get_engine()
    n = eng.conn.execute("select count(*), max(rowid) from emails where source='obsidian'").fetchone()
    key = (len(eng.index.ids), n[0], n[1])
    with _lock:
        t = _cache.get(id(eng))
        if t is None or t.key != key:
            t = build_tree(eng)
            t.key = (len(eng.index.ids), n[0], n[1])
            _cache[id(eng)] = t
        return t


# ---------- points of interest ----------

def _best_sentence(eng, reg, row, max_tokens=EXCERPT_TOKENS) -> str | None:
    """The one or two sentences of a section that scores highest against the question region (None if it has no prose)."""
    text = row.get("body") or ""
    # notes are often hard wrapped: join single line breaks, keep paragraphs and list items apart
    text = re.sub(r"(?<=\S)[ \t]*\n(?![ \t]*(\n|[-*+>#|]|\d+\.))[ \t]*", " ", text)
    ss = [s for s in pack.sentences(text) if not s.startswith("#") and re.search(r"[a-z]{3}", s)]
    if not ss:
        return None
    if len(ss) == 1:
        return pack.truncate_tokens(ss[0], max_tokens)
    S = reg.score(eng.enc.encode_docs(ss))
    order = [int(i) for i in np.argsort(-S)]
    keep = [order[0]]
    if pack.count_tokens(ss[order[0]] + " " + ss[order[1]]) <= max_tokens:
        keep.append(order[1])  # a second sentence when both fit the excerpt cap
    return pack.truncate_tokens(pack._excerpt(ss, keep), max_tokens)


def _folder_excerpt(eng, reg, t, zz, folder, used: set):
    """Best sentences from the highest scoring section inside the folder that has prose and was not
    already quoted for a folder listed above it."""
    idx = [i for i in np.flatnonzero(np.isfinite(zz)) if is_under(t.sec_folder[i], folder)]
    idx = sorted(idx, key=lambda i: -zz[i])[:5]
    for i in [i for i in idx if i not in used] + [i for i in idx if i in used]:
        row = store.get_email(eng.conn, t.sec_ids[i]) or {}
        s = _best_sentence(eng, reg, row)
        if s:
            used.add(i)
            return row.get("subject"), s
    return None, ""


def render(out: dict) -> str:
    if not out["related"]:
        return out["reason"]
    lines = []
    for i, f in enumerate(out["folders"], 1):
        tags = f" #{' #'.join(f['top_tags'][:3])}" if f.get("top_tags") else ""
        lines.append(f"{i}. {f['path']} ({f['hit_notes']}/{f['n_notes']} notes, score {f['score']:.1f}{tags})")
        for n in f["notes"]:
            lines.append(f"   - {n['uri']} (z {n['z']:.1f})")
        lines.append(f"   > {f['excerpt']}")
    return "\n".join(lines)


def points_of_interest(question: str, k_folders: int = 5, k_notes: int = 3, within: str | None = None,
                       engine=None, facets: dict | None = None, use_grok: bool = True) -> dict:
    """Which folders and notes of the vault a question lives in, ranked, with one excerpt per folder.

    within restricts everything (the yes/no included) to one subtree, for folder -> subfolder -> note.
    Returns {related, confidence, max_z, within, folders, context, tokens}. tokens counts `context`.
    """
    from atlas.agent import grok

    eng = engine or get_engine()
    t = get_tree(eng)
    within = norm_path(within)
    base = {"question": question, "within": _label(within)}
    if within and within not in t.folders:
        reason = f"No folder {_label(within)} in the vault."
        return {**base, "related": False, "confidence": 1.0, "max_z": 0.0, "folders": [], "reason": reason,
                "context": reason, "tokens": pack.count_tokens(reason)}

    exp = facets or (grok.expand(question) if use_grok else {"positive": [], "negative": []})
    pos, neg = list(exp.get("positive") or []), list(exp.get("negative") or [])
    reg, raw_all, z_all = hybrid.region_scores(eng, question, pos, neg)
    raw, z = raw_all[t.sec_rows], z_all[t.sec_rows]
    inside = t.folder_mask(within) if within else np.ones(len(t.sec_ids), bool)
    verdict = R.related_verdict(np.where(inside, raw, -1.0), np.where(inside, z, -99.0), eng.name)
    member = verdict.pop("member") & inside
    base.update(related=verdict["related"], confidence=verdict["confidence"], max_z=verdict["max_z"],
                facets=[question] + [p for p in pos if p != question])
    if not verdict["related"]:
        where = f"{_label(within)} of the vault" if within else "the vault"
        reason = (f"Nothing about this in {where}: best section z={verdict['max_z']:.1f}, below the related "
                  "threshold. Stop searching.")
        return {**base, "folders": [], "reason": reason, "context": reason, "tokens": pack.count_tokens(reason)}

    # a note's score is its best section; a note is a hit when that section is in the region
    zz = np.where(inside, z, -np.inf)
    note_best: dict = {}
    for i in np.flatnonzero(inside):
        n = t.sec_note[i]
        if n not in note_best or zz[i] > zz[note_best[n]]:
            note_best[n] = i
    hit_notes = {n for n, i in note_best.items() if member[i]}

    cands = []
    for p, f in t.folders.items():
        if not is_under(p, within) or (within and p == within and f.n_direct_notes == 0):
            continue
        idx = [note_best[n] for n in note_best if is_under(_note_dir(n), p)]
        hits = sorted((n for n in hit_notes if is_under(_note_dir(n), p)), key=lambda n: -zz[note_best[n]])
        if not hits:
            continue
        top = sorted((float(zz[i]) for i in idx), reverse=True)[:TOP_M]
        agg = 0.6 * top[0] + 0.4 * float(np.mean(top))
        precision = len(hits) / len(idx)  # specificity: a folder full of hits beats its diluted parent
        cands.append((agg * precision ** 0.5, p, hits, agg, precision))
    cands.sort(key=lambda c: -c[0])

    folders, seen_sets, quoted = [], [], set()
    for score, p, hits, agg, precision in cands:
        hs = set(hits)
        # skip an ancestor or descendant that points at exactly the same notes as a folder already listed
        if any(hs == s and (is_under(p, q) or is_under(q, p)) for q, s in seen_sets):
            continue
        seen_sets.append((p, hs))
        f = t.folders[p]
        notes = []
        for n in hits[:k_notes]:
            i = note_best[n]
            notes.append({"uri": t.sec_ids[i][4:], "note": n, "z": round(float(z[i]), 2)})
        title, excerpt = _folder_excerpt(eng, reg, t, zz, p, quoted)
        folders.append({**f.stats(), "score": round(score, 2), "max_z": round(float(zz[note_best[hits[0]]]), 2),
                        "hit_notes": len(hits), "precision": round(precision, 2), "notes": notes,
                        "title": title, "excerpt": excerpt})
        if len(folders) >= k_folders:
            break
    out = {**base, "reason": f"{len(hit_notes)} notes in {len(folders)} folders", "folders": folders}
    out["context"] = render(out)
    out["tokens"] = pack.count_tokens(out["context"])
    return out


# ---------- related folders ----------

def related_folders(path: str, k: int = 5, engine=None, local: bool = False, centered: bool = False) -> dict:
    """Nearest folders to `path` by folder vector, skipping its ancestors and descendants.

    centered=True subtracts the vault's mean section vector first, which spreads the cosines out (every
    note in one vault shares a voice) but is noisy on small vaults, so it is off by default.
    Each neighbour comes with the cosine, the tags both folders use, and the wikilinks they share
    (a link target both use, or a link from one into a note of the other)."""
    eng = engine or get_engine()
    t = get_tree(eng)
    p = norm_path(path)
    if p not in t.folders:
        reason = f"No folder {_label(p)} in the vault."
        return {"path": _label(p), "error": "not found", "related": [], "context": reason,
                "tokens": pack.count_tokens(reason)}
    f = t.folders[p]
    mu = t.mean if centered and t.mean is not None else 0.0

    def fv(g):
        return _norm((g.local if local else g.vec) - mu)

    v = fv(f)
    out = []
    for q, g in t.folders.items():
        if q == p or q == ROOT or is_under(q, p) or is_under(p, q):
            continue
        w = fv(g)
        tags = [x for x in f.top_tags if x in g.top_tags][:3]
        shared = set(f.links) & set(g.links)
        shared |= set(f.links) & g.stems
        shared |= set(g.links) & f.stems
        links = sorted(shared, key=lambda x: -(f.links.get(x, 0) + g.links.get(x, 0)))[:3]
        out.append({"path": _label(q), "cosine": round(float(v @ w), 3), "n_notes": g.n_notes,
                    "shared_tags": tags, "shared_links": links})
    out.sort(key=lambda r: -r["cosine"])
    out = out[:k]
    ctx = "\n".join(
        f"{r['path']} cos {r['cosine']:.2f}"
        + (f" tags {', '.join(r['shared_tags'])}" if r["shared_tags"] else "")
        + (f" links {', '.join(r['shared_links'])}" if r["shared_links"] else "") for r in out)
    return {"path": _label(p), "folder": f.stats(), "related": out, "context": ctx, "tokens": pack.count_tokens(ctx)}


def folder_tree(engine=None) -> list[dict]:
    """Every folder with its stats, parents before children."""
    t = get_tree(engine)
    return [t.folders[p].stats() for p in sorted(t.folders, key=lambda x: (depth(x), x))]


__all__ = ["points_of_interest", "related_folders", "folder_tree", "get_tree", "build_tree"]
