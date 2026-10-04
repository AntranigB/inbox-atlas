"""Encoder calls stay on one thread, and agent searches use the shared query expansion."""

import threading
from concurrent.futures import ThreadPoolExecutor

from atlas.agent import grok, tools
from atlas.model import encoder


def test_serial_runs_every_call_on_one_thread():
    seen, active, peak = set(), [0], [0]
    lock = threading.Lock()

    def work(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        seen.add(threading.current_thread().name)
        with lock:
            active[0] -= 1
        return encoder.serial(lambda: i * 2)  # nested call must not deadlock

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda i: encoder.serial(work, i), range(32)))
    assert out == [i * 2 for i in range(32)]
    assert len(seen) == 1 and peak[0] == 1


def test_topic_uses_expansion_facets(monkeypatch):
    monkeypatch.setattr(grok, "expand", lambda q: {"positive": ["Codeforces round", "hackathon"],
                                                   "negative": ["online coding assessment"]})
    pos, neg = tools._facets_for("coding competitions", ["coding competition"], ["job", "job interview"])
    assert pos == ["Codeforces round", "hackathon"]
    assert neg == ["online coding assessment", "job interview"]  # one word negatives are dropped


def test_no_topic_keeps_agent_facets(monkeypatch):
    monkeypatch.setattr(grok, "expand", lambda q: {"positive": [], "negative": []})
    assert tools._facets_for("", ["a"], ["b"]) == (["a"], ["b"])
    assert tools._facets_for("x", ["a"], ["b"]) == (["a"], ["b"])  # expansion unavailable
