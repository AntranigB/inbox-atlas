import imaplib
from email.message import EmailMessage

import pytest

from atlas import config, store
from atlas.ingest import gmail_export as gx
from atlas.ingest import imap, sync
from atlas.ingest.clean import clean_body, parse_message

MUTATING = {"store", "copy", "move", "expunge", "delete", "append", "rename", "create", "subscribe",
            "unsubscribe", "setacl", "deleteacl", "setquota", "close"}


def make_raw(i, subject=None, body=None, html=None):
    m = EmailMessage()
    m["From"] = f"Sender {i} <s{i}@example.com>"
    m["To"] = "me@gmail.com"
    m["Subject"] = subject or f"Message number {i}"
    m["Date"] = "Fri, 02 Oct 2026 10:00:00 -0400"
    m["Message-ID"] = f"<m{i}@example.com>"
    m.set_content(body or f"Body of message {i}.")
    if html:
        m.add_alternative(html, subtype="html")
    return m.as_bytes()


class FakeIMAP:
    """Stands in for imaplib.IMAP4_SSL. Records every command."""
    calls = []

    def __init__(self, host, port):
        self.msgs = {u: make_raw(u) for u in range(1, 8)}
        FakeIMAP.calls.append(("connect", host, port))

    def login(self, u, p):
        self.calls.append(("login",))
        return "OK", [b"ok"]

    def list(self, *a):
        self.calls.append(("list",))
        return "OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\All \\HasNoChildren) "/" "[Gmail]/All Mail"']

    def select(self, box, readonly=False):
        self.calls.append(("select", box, readonly))
        return "OK", [str(len(self.msgs)).encode()]

    def response(self, code):
        return code, [b"42"]

    def uid(self, cmd, *args):
        self.calls.append(("uid", cmd.upper(), args))
        if cmd.upper() == "SEARCH":
            crit = args[-1]
            lo = int(crit.split()[1].split(":")[0]) if crit.startswith("UID") else 1
            return "OK", [" ".join(str(u) for u in self.msgs if u >= lo).encode()]
        uids = [int(x) for x in args[0].split(",")]
        out = []
        for u in uids:
            head = f'{u} (UID {u} X-GM-THRID {9000 + u} X-GM-MSGID {1000 + u} X-GM-LABELS (\\\\Inbox "My Label")'
            if "BODY.PEEK[]" in args[1]:
                out += [((head + f" BODY[] {{{len(self.msgs[u])}}}").encode(), self.msgs[u]), b")"]
            else:
                out.append((head + ")").encode())
        return "OK", out

    def logout(self):
        self.calls.append(("logout",))

    def __getattr__(self, name):
        FakeIMAP.calls.append(("FORBIDDEN", name))
        return lambda *a, **k: ("OK", [])


@pytest.fixture
def fake_gmail(tmp_path, monkeypatch):
    FakeIMAP.calls = []
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeIMAP)
    monkeypatch.setenv("GMAIL_USER", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
    raw = tmp_path / "raw" / "gmail"
    monkeypatch.setattr(gx, "RAW_DIR", raw)
    monkeypatch.setattr(gx, "META_PATH", raw / "_meta.jsonl")
    monkeypatch.setattr(gx, "STATE_PATH", raw / "_state.json")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    return raw


def assert_read_only(calls):
    names = {c[0].lower() if c[0] != "uid" else c[1].lower() for c in calls}
    assert not any(c[0] == "FORBIDDEN" for c in calls), calls
    assert not (names & MUTATING), names
    for c in calls:
        if c[0] == "select":
            assert c[2] is True
        if c[0] == "uid":
            assert c[1] in ("FETCH", "SEARCH")
            if c[1] == "FETCH":
                assert "BODY[]" not in c[2][1] and "FLAGS" not in c[2][1]


def test_export_is_read_only_and_resumable(fake_gmail):
    conn = store.connect(":memory:")
    gx.export(max_n=5)
    assert_read_only(FakeIMAP.calls)
    assert ("select", '"[Gmail]/All Mail"', True) in FakeIMAP.calls
    files = sorted(p.name for p in fake_gmail.glob("*.eml"))
    assert files == [f"{1000 + u}.eml" for u in (3, 4, 5, 6, 7)]  # newest 5
    gx.parse_all(conn)
    e = store.get_email(conn, "1007")
    assert e["thread_id"] == "9007" and "My Label" in e["labels"] and e["from_addr"] == "s7@example.com"
    # second run downloads nothing new for the same set
    FakeIMAP.calls = []
    gx.export(max_n=5)
    assert not [c for c in FakeIMAP.calls if c[0] == "uid" and "BODY.PEEK[]" in str(c[2])]
    assert_read_only(FakeIMAP.calls)


def test_sync_new_returns_new_ids(fake_gmail, monkeypatch):
    conn = store.connect(":memory:")
    gx.export(max_n=3)
    gx.parse_all(conn)
    FakeIMAP.calls = []
    orig = FakeIMAP.__init__

    def init(self, h, p):
        orig(self, h, p)
        self.msgs[8] = make_raw(8, "Fresh mail")
    monkeypatch.setattr(FakeIMAP, "__init__", init)
    ids = sync.sync_new(conn)
    assert ids == ["1008"]
    assert store.get_email(conn, "1008")["subject"] == "Fresh mail"
    assert (fake_gmail / "1008.eml").exists()
    assert sync.sync_new(conn) == []
    assert_read_only(FakeIMAP.calls)


def test_sync_without_creds(monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "")
    assert sync.sync_new() == []


def test_safe_imap_refuses_mutation(monkeypatch):
    FakeIMAP.calls = []
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeIMAP)
    c = imap.SafeIMAP()
    for bad in ("store", "copy", "move", "expunge", "delete", "append"):
        with pytest.raises(imap.ReadOnlyViolation):
            getattr(c, bad)
    for cmd in ("STORE", "COPY", "MOVE", "EXPUNGE"):
        with pytest.raises(imap.ReadOnlyViolation):
            c.uid(cmd, "1", "+FLAGS (\\Deleted)")
    with pytest.raises(imap.ReadOnlyViolation):
        c.uid("FETCH", "1", "(BODY[])")
    with pytest.raises(imap.ReadOnlyViolation):
        c.select("INBOX", readonly=False)
    assert not [x for x in FakeIMAP.calls if x[0] == "FORBIDDEN"]


def test_missing_creds_message(monkeypatch, capsys):
    monkeypatch.setenv("GMAIL_USER", "")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "")
    assert gx.main([]) == 1
    assert "App Password" in capsys.readouterr().out


# ---------- cleaning ----------

def test_strip_quoted_reply_and_signature():
    body = """Sounds good, see you at 5.

--
Alex Chen
CS PhD student

On Fri, Oct 2, 2026 at 3:14 PM Bob <bob@x.com> wrote:
> Want to grab dinner?
> -Bob
"""
    assert clean_body(body) == "Sounds good, see you at 5."


def test_strip_gmail_wrapped_attribution():
    body = "Yes!\n\nOn Fri, Oct 2, 2026 at 3:14 PM Some Very Long Name <\nsomeone@example.com> wrote:\n\n> hi"
    assert clean_body(body) == "Yes!"


def test_strip_outlook_and_forward_and_iphone():
    assert clean_body("Approved.\n\n-----Original Message-----\nFrom: x\nSent: y\n\nold") == "Approved."
    assert clean_body("FYI\n\n---------- Forwarded message ---------\nFrom: a\nDate: b\n\nstuff") == "FYI"
    assert clean_body("ok will do\n\nSent from my iPhone") == "ok will do"
    assert clean_body("Hi\n\nFrom: Bob <b@x.com>\nSent: Monday\nTo: me\nSubject: hi\n\nold") == "Hi"
    # a lone From: in prose is kept
    assert "From: the dean" in clean_body("Message\nFrom: the dean of students\nhello")


def test_strip_tracking_and_long_urls():
    body = ("Your order shipped! Track it here https://click.mail.example.com/ls/click?upn=aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789\n"
            "Unsubscribe | Update your preferences\n​​\nView this email in your browser")
    out = clean_body(body)
    assert "Your order shipped" in out and "[click.mail.example.com]" in out
    assert "Unsubscribe" not in out and "browser" not in out and "​" not in out


def test_parse_html_only_and_encoded_header():
    m = EmailMessage()
    m["From"] = "=?utf-8?b?Sm9zw6kgR2FyY8OtYQ==?= <jose@example.com>"
    m["Subject"] = "=?utf-8?q?Caf=C3=A9_meetup?="
    m["Date"] = "Sat, 03 Oct 2026 09:00:00 +0000"
    m.set_content("<html><head><style>p{color:red}</style></head><body><p>Join us   at the <b>café</b>.</p>"
                  "<div class='gmail_quote'>On Fri someone wrote: old stuff</div></body></html>", subtype="html")
    r = parse_message(m.as_bytes(), "123", "456", ["\\Inbox"])
    assert r["from_name"] == "José García" and r["subject"] == "Café meetup"
    assert r["body"] == "Join us at the café." and r["date"] == 1791018000
    assert r["id"] == "123" and r["thread_id"] == "456" and r["snippet"] == r["body"]


def test_parse_prefers_plain_and_fallback_ids():
    raw = make_raw(1, "Re: Lunch", "plain wins", "<p>html loses</p>")
    r = parse_message(raw)
    assert r["body"] == "plain wins" and len(r["id"]) == 16
    r2 = parse_message(make_raw(2, "Lunch", "x"))
    assert r2["thread_id"] == r["thread_id"]  # Re: stripped for subject thread hash
