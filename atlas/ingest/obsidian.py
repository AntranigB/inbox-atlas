"""Obsidian vault source: markdown notes chunked by heading, stored next to emails with source='obsidian'.

    OBSIDIAN_VAULT=/path/to/vault uv run python -m atlas.ingest.obsidian
    uv run python -m atlas.index.build --encoder base --no-grok

Each heading section becomes one row in the emails table (no schema change):
  id         obs:<relpath>#<anchor>
  uri        <relpath>#<anchor>          (what agents pass to atlas_get)
  thread_id  obs:<relpath>               (all sections of one note share a thread)
  subject    note title > heading
  from_name  vault folder (daily, people, projects, ...)
  to_addrs   JSON list of [[wikilinks]] (thread-like links between notes)
  labels     JSON list of #tags and frontmatter tags
Everything stays local: notes are embedded with the local encoder and never sent to Grok at
ingest time. Only the few excerpts packed by atlas.context at query time ever leave the machine.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

from atlas import config, store

SOURCE = "obsidian"
SKIP_DIRS = {".obsidian", "archive", "backup", "templates", ".trash", ".git"}
DEFAULT_VAULT = "/Users/xerneas/Brain"
MAX_CHARS = 500  # split sections longer than this on paragraph boundaries (focused vectors)
MIN_CHARS = 20

HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
TAG = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]*)")
DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


def vault_path(path=None) -> Path:
    return Path(path or os.getenv("OBSIDIAN_VAULT") or DEFAULT_VAULT).expanduser()


def anchor(heading: str) -> str:
    a = re.sub(r"[^\w\s-]", "", heading.lower()).strip()
    return re.sub(r"\s+", "-", a) or "top"


def split_frontmatter(text: str):
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[3:end].strip(), text[end + 4:].lstrip("\n")
    return "", text


def parse_frontmatter(fm: str) -> dict:
    """Tiny YAML subset: key: value and key: [a, b] and '- item' lists. Enough for tags/date/title."""
    out, key = {}, None
    for line in fm.splitlines():
        if re.match(r"^\s+-\s+", line) and key:
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(line.split("-", 1)[1].strip().strip("'\""))
            continue
        m = re.match(r"^([\w-]+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if val.startswith("[") and val.endswith("]"):
            out[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        elif val:
            out[key] = val.strip("'\"")
        else:
            out[key] = []
    return out


def _date(fm: dict, rel: str, mtime: float) -> int:
    for cand in (fm.get("date"), fm.get("created"), (DATE_IN_NAME.search(rel) or [None])[0]):
        if isinstance(cand, str):
            try:
                d = dt.date.fromisoformat(cand[:10])
                return int(dt.datetime(d.year, d.month, d.day, 12, tzinfo=dt.timezone.utc).timestamp())
            except ValueError:
                pass
    return int(mtime)


def _split_long(text: str):
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > MAX_CHARS:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def chunk_note(text: str, title: str):
    """[(heading, anchor, body)] split on #, ##, ### headings. Preamble keeps the note title."""
    sections, head, buf = [], title, []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            sections.append((head, "\n".join(buf).strip()))
            head, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    sections.append((head, "\n".join(buf).strip()))
    out, seen = [], {}
    for h, body in sections:
        if len(re.sub(r"\s", "", body)) < MIN_CHARS:
            continue
        a = anchor(h)
        for j, part in enumerate(_split_long(body)):
            key = a if j == 0 else f"{a}-{j + 1}"
            n = seen.get(key, 0)
            seen[key] = n + 1
            out.append((h, key if n == 0 else f"{key}-{n}", part))
    return out


def iter_notes(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith("."))
        for f in sorted(filenames):
            if f.endswith(".md"):
                yield Path(dirpath) / f


def note_rows(root: Path, path: Path):
    rel = path.relative_to(root).as_posix()
    raw = path.read_text(encoding="utf-8", errors="replace")
    fm_text, body = split_frontmatter(raw)
    fm = parse_frontmatter(fm_text)
    title = fm.get("title") if isinstance(fm.get("title"), str) else path.stem
    tags = fm.get("tags") or []
    tags = [tags] if isinstance(tags, str) else list(tags)
    tags = list(dict.fromkeys([t.lstrip("#") for t in tags] + TAG.findall(body)))
    links = list(dict.fromkeys(l.strip() for l in WIKILINK.findall(body)))
    folder = rel.split("/")[0] if "/" in rel else "vault"
    date = _date(fm, rel, path.stat().st_mtime)
    rows = []
    for heading, anc, text in chunk_note(body, title):
        uri = f"{rel}#{anc}"
        subject = title if heading == title else f"{title} > {heading}"
        sec_links = list(dict.fromkeys(l.strip() for l in WIKILINK.findall(text))) or links
        rows.append({"id": f"obs:{uri}", "thread_id": f"obs:{rel}", "from_addr": folder, "from_name": folder,
                     "to_addrs": json.dumps(sec_links), "date": date, "subject": subject, "body": text,
                     "snippet": " ".join(text.split())[:200], "labels": json.dumps(tags), "source": SOURCE})
    return rows


def load_vault(conn, path=None, replace=True):
    """Read the vault into the store. Returns the number of sections written."""
    root = vault_path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"no Obsidian vault at {root} (set OBSIDIAN_VAULT)")
    rows = []
    for p in iter_notes(root):
        try:
            rows.extend(note_rows(root, p))
        except Exception as e:  # one bad note should not stop the ingest
            print(f"skip {p.name}: {e}", file=sys.stderr)
    if replace:
        store.delete_source(conn, SOURCE)
    store.upsert_emails(conn, rows)
    return rows


def uri_of(row: dict) -> str:
    rid = row.get("id") or ""
    return rid[4:] if rid.startswith("obs:") else f"gmail:{rid}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    conn = store.connect()
    rows = load_vault(conn, argv[0] if argv else None)
    notes = len({r["thread_id"] for r in rows})
    print(f"loaded {len(rows)} sections from {notes} notes into {config.DB_PATH}")
    print("next: uv run python -m atlas.index.build --encoder base --no-grok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
