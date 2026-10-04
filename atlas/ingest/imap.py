"""Read-only Gmail IMAP access.

HARD RULE: nothing in the user's mailbox may change. The mailbox is selected with
readonly=True (EXAMINE), bodies are fetched with BODY.PEEK[] so the Seen flag never moves,
and SafeIMAP refuses every mutating command even if some future code tries one.
"""

import imaplib
import re

from atlas import config

HOST = "imap.gmail.com"
PORT = 993
ALL_MAIL = "[Gmail]/All Mail"

# commands we allow at all; uid() is further limited to FETCH and SEARCH
_ALLOWED = {"login", "logout", "list", "select", "uid", "noop", "capability", "status"}
_ALLOWED_UID = {"FETCH", "SEARCH"}
_FORBIDDEN_WORDS = re.compile(r"\b(STORE|COPY|MOVE|EXPUNGE|DELETE|APPEND|\+FLAGS|-FLAGS|FLAGS\b(?!\.))", re.I)

SETUP_MSG = """
Gmail credentials are missing.

  1. Turn on 2-Step Verification for your Google account.
  2. Create an App Password at https://myaccount.google.com/apppasswords
  3. Make sure IMAP is enabled (Gmail settings > Forwarding and POP/IMAP).
  4. Add these lines to .env in the repo root:

       GMAIL_USER=you@gmail.com
       GMAIL_APP_PASSWORD=abcdefghijklmnop

The export is read-only: it never deletes, moves, flags or marks anything as read.
No credentials? Run demo mode instead:  uv run python -m atlas.ingest.load_fixture
"""


class ReadOnlyViolation(RuntimeError):
    pass


class SafeIMAP:
    """Thin wrapper over imaplib.IMAP4_SSL that only exposes read-only operations."""

    def __init__(self, host=HOST, port=PORT):
        self._c = imaplib.IMAP4_SSL(host, port)

    def __getattr__(self, name):
        if name.lower() not in _ALLOWED:
            raise ReadOnlyViolation(f"IMAP command {name!r} is not allowed (read-only export)")
        return getattr(self._c, name)

    def select(self, mailbox, readonly=True):
        if not readonly:
            raise ReadOnlyViolation("mailbox must be opened readonly")
        return self._c.select(mailbox, readonly=True)

    def uid(self, command, *args):
        cmd = command.upper()
        if cmd not in _ALLOWED_UID:
            raise ReadOnlyViolation(f"UID {cmd} is not allowed (read-only export)")
        if cmd == "FETCH":
            items = " ".join(str(a) for a in args[1:])
            if _FORBIDDEN_WORDS.search(items) or re.search(r"BODY\[", items, re.I):
                # BODY[] (without PEEK) would set \\Seen
                raise ReadOnlyViolation(f"FETCH items {items!r} could modify the mailbox")
        return self._c.uid(cmd, *args)


def credentials():
    user = config.env("GMAIL_USER").strip()
    pw = config.env("GMAIL_APP_PASSWORD").replace(" ", "").strip()
    return user, pw


def connect():
    user, pw = credentials()
    if not user or not pw:
        raise SystemExit(SETUP_MSG)
    c = SafeIMAP()
    c.login(user, pw)
    return c


def find_all_mail(c):
    """Locate the \\All folder (name differs by locale), default [Gmail]/All Mail."""
    try:
        typ, data = c.list()
        for line in data or []:
            s = line.decode(errors="replace") if isinstance(line, bytes) else str(line)
            if "\\All" in s:
                m = re.search(r'"([^"]+)"\s*$', s) or re.search(r"(\S+)\s*$", s)
                if m:
                    return m.group(1)
    except ReadOnlyViolation:
        raise
    except Exception:
        pass
    return ALL_MAIL


def open_all_mail(c):
    box = find_all_mail(c)
    typ, data = c.select(f'"{box}"', readonly=True)
    if typ != "OK":
        raise RuntimeError(f"could not open {box}: {data}")
    uidvalidity = None
    try:
        typ, uv = c._c.response("UIDVALIDITY")
        if uv and uv[0]:
            uidvalidity = int(uv[0])
    except Exception:
        pass
    return box, uidvalidity


def search_uids(c, since_uid=0):
    crit = f"UID {since_uid + 1}:*" if since_uid else "ALL"
    typ, data = c.uid("SEARCH", None, crit)
    if typ != "OK":
        raise RuntimeError(f"SEARCH failed: {data}")
    uids = [int(x) for x in (data[0] or b"").split()]
    return sorted(u for u in uids if u > since_uid)


_RE_UID = re.compile(rb"\bUID (\d+)")
_RE_MSGID = re.compile(rb"X-GM-MSGID (\d+)")
_RE_THRID = re.compile(rb"X-GM-THRID (\d+)")
_RE_LABELS = re.compile(rb"X-GM-LABELS \((.*?)\)(?: [A-Z]|\)|$)", re.S)


def parse_labels(s):
    if not s:
        return []
    toks = re.findall(rb'"((?:[^"\\]|\\.)*)"|(\S+)', s)
    out = []
    for q, bare in toks:
        t = (q or bare).decode("utf-8", errors="replace").replace('\\"', '"').replace("\\\\", "\\")
        out.append(t)
    return out


def _parse_fetch(data):
    """Yield (meta dict, body bytes or None) from an imaplib UID FETCH response."""
    for item in data:
        if isinstance(item, tuple):
            head, body = item[0], item[1]
        elif isinstance(item, bytes) and item not in (b")",):
            head, body = item, None
        else:
            continue
        m_uid = _RE_UID.search(head)
        m_id = _RE_MSGID.search(head)
        if not (m_uid and m_id):
            continue
        m_th = _RE_THRID.search(head)
        m_lab = _RE_LABELS.search(head)
        yield {
            "uid": int(m_uid.group(1)),
            "msgid": m_id.group(1).decode(),
            "thrid": m_th.group(1).decode() if m_th else None,
            "labels": parse_labels(m_lab.group(1)) if m_lab else [],
        }, (body if (body is not None and b"BODY[]" in head) else None)


def fetch_meta(c, uids):
    if not uids:
        return []
    typ, data = c.uid("FETCH", ",".join(map(str, uids)), "(UID X-GM-MSGID X-GM-THRID X-GM-LABELS)")
    if typ != "OK":
        raise RuntimeError(f"FETCH meta failed: {data}")
    return [m for m, _ in _parse_fetch(data)]


def fetch_raw(c, uids):
    if not uids:
        return []
    typ, data = c.uid("FETCH", ",".join(map(str, uids)), "(UID X-GM-MSGID X-GM-THRID X-GM-LABELS BODY.PEEK[])")
    if typ != "OK":
        raise RuntimeError(f"FETCH body failed: {data}")
    return [(m, b) for m, b in _parse_fetch(data) if b is not None]
