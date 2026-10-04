"""Evidence must come from a labeled source, not a coincidental distractor."""

import pytest

from eval.private_eval import evidence_from_blocks, run


def test_evidence_cannot_be_borrowed_from_unrelated_email():
    item = {"targets": ["workshop"], "evidence": ["Room 205", "2pm"]}
    assert not evidence_from_blocks([("workshop", "Room 205"), ("distractor", "2pm")], item)
    assert evidence_from_blocks([("workshop", "Room 205 at 2pm")], item)
    assert not evidence_from_blocks([("distractor", "Room 205 at 2pm")], item)


def test_facts_can_span_retained_chunks_of_expected_sources():
    item = {"targets": ["workshop"], "evidence": ["Room 205", "2pm"]}
    assert evidence_from_blocks([("workshop", "Room 205"), ("workshop", "2pm")], item)
    # A regex cannot manufacture a phrase across two disconnected excerpts.
    assert not evidence_from_blocks([("workshop", "Room"), ("workshop", "205")],
                                    {"targets": ["workshop"], "evidence": [r"Room\s+205"]})


def test_invalid_manifest_or_budgets_fail_before_loading_corpus():
    with pytest.raises(ValueError, match="present question"):
        run(None, {"questions": []}, None)
    with pytest.raises(ValueError, match="Budgets"):
        run(None, {"questions": [{}]}, None, budgets=[-1])


def test_compact_baseline_accounting_and_encoder_restoration(monkeypatch):
    from types import SimpleNamespace
    import json
    from eval import private_eval as ev
    from atlas.context import pack

    encoder = SimpleNamespace(name="test", dim=2)
    engine = SimpleNamespace(enc=encoder, name="test")
    cp = SimpleNamespace(ids=["one"], text={"one": "Room 205"}, doc_of={"one": "one"})
    monkeypatch.setattr(ev, "Corpus", lambda eng: cp)
    def strategy(*args, **kwargs):
        return ["one"], "Room 205", [("one", "Room 205")]
    for name in ("strat_keyword", "strat_embed_docs", "strat_chunks"):
        monkeypatch.setattr(ev, name, strategy)
    result = {"answerable": True, "context": "Room 205", "items": [
        {"source": "gmail", "uri": "gmail:one"}]}
    monkeypatch.setattr(pack, "build_context", lambda *args, **kwargs: result)
    monkeypatch.setattr(pack, "render", lambda items: "Room 205")
    baseline = SimpleNamespace(build_context=pack.build_context, render=pack.render,
                               tool_payload=lambda p: {"context": p["context"]})
    manifest = {"questions": [{"q": "where?", "targets": ["one"], "evidence": ["Room 205"]}]}
    report = ev.run(engine, manifest, baseline, budgets=[20])
    expected = pack.count_tokens(json.dumps({"context": "Room 205"}, separators=(",", ":")))
    assert report["summary"]["original @20"]["mean_payload_tokens"] == expected
    assert report["summary"]["original @20"]["evidence_hits"] == 1
    assert engine.enc is encoder
    monkeypatch.setattr(ev, "Corpus", lambda eng: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError):
        ev.run(engine, manifest, baseline)
    assert engine.enc is encoder
