"""Offline tests for the training branch: parsing, metrics, region encoder (CPU, random weights)."""

import random

import numpy as np
import pytest

from train.common import clean_body, doc_text, is_reply, norm_subject
from train.metrics import auroc, ece, fit_platt, retrieval


def test_clean_body_strips_quotes_and_forwards():
    body = "Sounds good, see you at 3.\n\n-----Original Message-----\nFrom: Bob\nSent: Monday\nold stuff here"
    assert clean_body(body) == "Sounds good, see you at 3."
    fwd = ("\n---------------------- Forwarded by Jeff Dasovich/NA/Enron on 05/01/2001 ----\n\n"
           "The ISO price cap order is attached and worth reading in full today.")
    out = clean_body(fwd)
    assert "price cap order" in out and "Forwarded by" not in out


def test_subject_threading():
    assert norm_subject("RE: Fw: Gas deal  update") == "gas deal update"
    assert is_reply("Re: hi") and not is_reply("Fwd: hi")
    assert doc_text({"subject": "s", "from_name": "a", "body": "b"}, "nosubj") == "a\nb"


def test_enron_label_filter():
    from train.enron import good_label, humanize

    assert humanize("California_Energy") == "california energy"
    assert good_label("california energy")
    assert not good_label("8 00") and not good_label("sent") and not good_label("2001 07")


def test_retrieval_metrics():
    S = np.array([[0.9, 0.1, 0.5], [0.1, 0.2, 0.3]])
    r = retrieval(S, [{0}, {2}], k=10)
    assert r["recall@10"] == 1.0 and r["ndcg@10"] == 1.0
    assert auroc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    y = np.array([0, 0, 1, 1] * 50)
    s = np.array([-2.0, -1.0, 1.0, 2.0] * 50)
    a, c = fit_platt(y, s)
    p = 1 / (1 + np.exp(-(a * s + c)))
    assert ece(y, p) < 0.2


def test_build_anchors_respects_splits():
    from train.mine import build_anchors

    base = {"body": "x", "from_name": "f"}
    emails = [
        {**base, "id": "a", "split": "train", "topics": ["gas"], "subject": "pipeline capacity report",
         "is_reply": False, "parent_id": None, "thread_id": "t1"},
        {**base, "id": "b", "split": "train", "topics": [], "subject": "Re: pipeline capacity report",
         "is_reply": True, "parent_id": "a", "thread_id": "t1"},
        {**base, "id": "c", "split": "test_topic", "topics": ["held"], "subject": "a heldout subject",
         "is_reply": False, "parent_id": None, "thread_id": "t2"},
    ]
    topics = {"gas": {"split": "train"}, "held": {"split": "heldout"}}
    an = build_anchors(emails, topics, random.Random(0))
    assert {x["kind"] for x in an} == {"topic", "subject", "reply"}
    assert all(x["pos"] != "c" for x in an)


def test_region_encoder_build_score_prob(tmp_path):
    torch = pytest.importorskip("torch")
    from atlas.model.region_encoder import LearnedRegionModel, SetRegionNet

    torch.manual_seed(0)
    m = LearnedRegionModel(SetRegionNet(d=32, h=32, heads=4, m=4, k=4), T=2.0)
    p = tmp_path / "region.pt"
    m.save(p)
    m2 = LearnedRegionModel.load(p)
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(3, 32))
    reg = m2.build(pos, rng.normal(size=(1, 32)))
    E = rng.normal(size=(10, 32)).astype(np.float32)
    s, pr = reg.score(E), reg.prob(E)
    assert s.shape == (10,) and pr.shape == (10,)
    assert np.all((pr >= 0) & (pr <= 1))
    # anchors start at the positive centroid, so the centroid direction scores highest
    c = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    assert reg.score(c.mean(0, keepdims=True))[0] > s.max()
    d = reg.describe()
    assert d["kind"] == "learned" and d["k"] == 4 and len(d["anchor_facets"]) == 4
    assert m2.build(pos, None).score(E).shape == (10,)
    with pytest.raises(ValueError):
        m2.build(rng.normal(size=(2, 16)), None)


def test_build_personal_from_fixture(tmp_path, monkeypatch):
    import json
    import sys

    from atlas import store

    rows = [json.loads(l) for l in open("tests/fixtures/mailbox.jsonl")]
    for r in rows:  # pretend the fixture topics are user Gmail labels
        r["labels"] = json.dumps(["INBOX", "CATEGORY_UPDATES", r["topic"]])
        r.pop("topic")
    db = tmp_path / "mail.sqlite"
    conn = store.connect(db)
    store.upsert_emails(conn, rows)
    conn.close()
    import train.build_personal as bp

    monkeypatch.setattr(bp, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--min-label", "1"])
    bp.main()
    out = [json.loads(l) for l in open(tmp_path / "datasets" / "personal" / "emails.jsonl")]
    assert len(out) == len(rows)
    labs = {l for e in out for l in e["topics"]}
    assert "inbox" not in labs and not any(l.startswith("category") for l in labs)
    assert "hackathon" in labs


def test_region_facets_numpy_matches_torch():
    torch = pytest.importorskip("torch")
    from atlas.model.region_encoder import LearnedRegionModel, SetRegionNet, membership

    torch.manual_seed(0)
    net = SetRegionNet(d=16, h=16, heads=4, m=4, k=2, facets=True).eval()
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(size=(3, 16)).astype(np.float32), rng.normal(size=(2, 16)).astype(np.float32)
    E = rng.normal(size=(7, 16)).astype(np.float32)
    reg = LearnedRegionModel(net).build(pos, neg)
    with torch.no_grad():
        P, N = torch.from_numpy(pos)[None], torch.from_numpy(neg)[None]
        a, tau, b, fx = net(P, torch.ones(1, 3, dtype=torch.bool), N, torch.ones(1, 2, dtype=torch.bool))
        m = membership(torch.from_numpy(E), a, tau, b, fx)[0].numpy()
    assert np.allclose(reg.score(E), m, atol=1e-3)
    assert reg.describe()["facet_tau"] is not None


def test_region_encoder_learns_toy_sets():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from atlas.model.region_encoder import SetRegionNet, membership

    torch.manual_seed(0)
    d = 16
    centers = F.normalize(torch.randn(6, d), dim=-1)
    net = SetRegionNet(d=d, h=32, heads=4, m=4, k=2, tau0=10.0, b0=8.0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(150):
        t = torch.randint(0, 6, (16,))
        P = F.normalize(centers[t].unsqueeze(1) + 0.1 * torch.randn(16, 3, d), dim=-1)
        mem = F.normalize(centers[t].unsqueeze(1) + 0.15 * torch.randn(16, 8, d), dim=-1)
        oth = F.normalize(centers[(t + 1) % 6].unsqueeze(1) + 0.15 * torch.randn(16, 8, d), dim=-1)
        Y = torch.cat([torch.ones(16, 8), torch.zeros(16, 8)], 1)
        a, tau, b, fx = net(P, torch.ones(16, 3, dtype=torch.bool), None, None)
        loss = F.binary_cross_entropy_with_logits(membership(torch.cat([mem, oth], 1), a, tau, b, fx), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.3
