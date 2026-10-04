"""Stage B: freeze the student, cache embeddings, train the Set Transformer region encoder.

Per training example: a topic set T is sampled; the region encoder sees 1-8 of T's phrases
(positives, with phrase dropout) and 0-3 phrases of confuser topics (negatives). Emails scored:
members of T, hard non-members (nearest to T's phrase centroid), emails near the negative
phrases, and random emails. Loss:
  BCE(m(e), member)                                   membership with hard negatives
  + neg_weight * softplus(m(e)) on emails near negative phrases (must sit below the boundary)
  + kd_weight * KL(softmax(teacher) || softmax(m)) over the folder label's teacher candidates
Then a scalar temperature T is fit on val-split emails so sigmoid(m/T) is calibrated.

usage: uv run python -m train.stage_b --model models/atlas-embed --run b1 [--ds enron]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from train.common import DATASETS, RUNS, doc_text, norm_subject, read_jsonl, sha
from train.metrics import auroc, retrieval
from train.topic_sets import folder_sets, grok_sets, load_grok


def embed_cache(enc, emails, path):
    if path.exists():
        return np.load(path).astype(np.float32)
    E = enc.encode_docs([doc_text(e) for e in emails])
    np.save(path, E.astype(np.float16))
    return E


_grok_cache = {}


def build_sets(emails, topics, grok, enc, pool_split):
    """Topic sets with members restricted to emails of the given splits. Grok phrase clusters are
    computed once over train + val emails, then members are filtered per split."""
    pool = [e for e in emails if e["split"] in pool_split]
    sets = folder_sets(pool, topics, grok)
    if "g" not in _grok_cache:
        t0 = time.time()
        _grok_cache["g"] = grok_sets([e for e in emails if e["split"] in ("train", "val")], grok, enc.encode_queries)
        print(f"grok clusters {len(_grok_cache['g'])} in {time.time() - t0:.0f}s", flush=True)
    ids = {e["id"] for e in pool}
    for g in _grok_cache["g"]:
        mem = [m for m in g["members"] if m in ids]
        sets.append({**g, "members": mem})
    return [s for s in sets if len(s["members"]) >= 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/atlas-embed")
    ap.add_argument("--run", default="b1")
    ap.add_argument("--ds", default="enron")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--neg-weight", type=float, default=1.0)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--no-neg-phrases", action="store_true")
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--no-facets", action="store_true", help="anchors only, phrases are not used as extra anchors")
    ap.add_argument("--pu", action="store_true", help="draw negatives only from emails labeled by the same source")
    ap.add_argument("--out", default=None, help="region.pt path, default <model>/region.pt")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from atlas.model.region_encoder import LearnedRegionModel, SetRegionNet, membership, slice_fx
    from atlas.model.trained import load, load_base

    torch.manual_seed(0)
    rng = random.Random(0)
    run = RUNS / a.run
    run.mkdir(parents=True, exist_ok=True)
    d = DATASETS / a.ds
    mp = Path(a.model)
    enc = load(mp) if mp.exists() and (mp / "atlas.json").exists() else load_base()
    emails = read_jsonl(d / "emails.jsonl")
    topics = json.loads((d / "topics.json").read_text()) if (d / "topics.json").exists() else {}
    grok = load_grok(d / "grok_labels.jsonl")
    teacher = {} if a.no_distill or not (d / "teacher.json").exists() else json.loads((d / "teacher.json").read_text())
    t0 = time.time()
    tag = (mp.name if mp.exists() else "base")
    E_all = embed_cache(enc, emails, d / f"emb_{tag}.npy")
    print(f"embeddings {E_all.shape} in {time.time() - t0:.0f}s", flush=True)
    idx = {e["id"]: i for i, e in enumerate(emails)}

    sets = [s for s in build_sets(emails, topics, grok, enc, {"train"}) if s["split"] == "train"]
    val_sets = [s for s in build_sets(emails, topics, grok, enc, {"val"}) if s["split"] == "train"]
    print(f"train sets {len(sets)} (folder {sum(s['src'] == 'folder' for s in sets)}), val sets {len(val_sets)}", flush=True)

    phrases = sorted({p for s in sets + val_sets for p in s["phrases"] + s["subjects"]})
    PV = enc.encode_queries(phrases)
    pidx = {p: i for i, p in enumerate(phrases)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    PVt = torch.from_numpy(PV).to(dev)
    Et = torch.from_numpy(E_all).to(dev)
    train_rows = torch.tensor([i for i, e in enumerate(emails) if e["split"] == "train"], device=dev)

    def prep(ss):
        for s in ss:
            s["mi"] = [idx[m] for m in s["members"] if m in idx]
            s["mset"] = set(s["mi"])
            s["pi"] = [pidx[p] for p in s["phrases"]]
            s["si"] = [pidx[p] for p in s["subjects"]]
            c = PVt[s["pi"]].mean(0)
            s["c"] = F.normalize(c, dim=0)

    prep(sets)
    prep(val_sets)
    # region-val: 10% of train topics never seen in region training, scored on val emails, for
    # model selection (step 0 = the initialization, which is close to the heuristic region)
    rv_names = {s["name"] for s in sets if int(sha("rv:" + s["name"]), 16) % 10 == 0}
    sets = [s for s in sets if s["name"] not in rv_names]
    val_rows = torch.tensor([i for i, e in enumerate(emails) if e["split"] == "val"], device=dev)
    vlist = val_rows.tolist()
    rrng = random.Random(7)
    rv = []
    for s in val_sets:
        if s["name"] in rv_names and len(s["mi"]) >= 3:
            pool = s["pi"][1:] + s["si"]
            ph = [s["pi"][0]] + rrng.sample(pool, min(4, len(pool)))
            y = np.array([1.0 if r in s["mset"] else 0.0 for r in vlist])
            rv.append((ph, y))
    print(f"region-val topics {len(rv)}, training topics {len(sets)}", flush=True)

    def rv_eval():
        net.eval()
        au, nd = [], []
        with torch.no_grad():
            Ev = Et[val_rows]
            for ph, y in rv:
                A, tau, b, fx = net(PVt[ph].unsqueeze(0), torch.ones(1, len(ph), dtype=torch.bool, device=dev), None, None)
                m = membership(Ev, A, tau, b, fx)[0].cpu().numpy()
                au.append(auroc(y, m))
                nd.append(retrieval(m[None], [set(np.where(y > 0)[0])])["ndcg@10"])
        net.train()
        return {"auroc": float(np.mean(au)), "ndcg@10": float(np.mean(nd)), "score": float(np.mean(au) + np.mean(nd))}
    C = torch.stack([s["c"] for s in sets])
    conf = (C @ C.T).fill_diagonal_(-1).topk(20, dim=1).indices.cpu().numpy()
    # Labels are partial (only folder-filed or Grok-labeled emails carry topics), so an unlabeled
    # email near a topic is often a true member. Negatives for a set come only from emails labeled
    # by the same source with --pu (positive-unlabeled correction; it did not help in b5-pu).
    pools = {}
    for src in ("folder", "grok"):
        if not a.pu:
            rows = [i for i, e in enumerate(emails) if e["split"] == "train"]
        elif src == "folder":
            rows = [i for i, e in enumerate(emails) if e["split"] == "train" and e.get("topics")]
        else:
            rows = [i for i, e in enumerate(emails) if e["split"] == "train" and e["id"] in grok]
        pools[src] = np.array(rows)
    hard = [None] * len(sets)
    for src, rows in pools.items():
        ii = [i for i, s_ in enumerate(sets) if s_["src"] == src]
        Ep = Et[torch.from_numpy(rows).to(dev)]
        for c0 in range(0, len(ii), 256):
            chunk = ii[c0 : c0 + 256]
            top = (C[chunk] @ Ep.T).topk(min(400, len(rows)), dim=1).indices.cpu().numpy()
            for j, row in zip(chunk, top):
                hard[j] = [int(r) for r in rows[row] if int(r) not in sets[j]["mset"]]
    pool_lists = {k: [int(x) for x in v] for k, v in pools.items()}
    print(f"prepared sets in {time.time() - t0:.0f}s", flush=True)

    net = SetRegionNet(d=E_all.shape[1], facets=not a.no_facets).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    trn = [int(x) for x in train_rows.cpu().numpy()]

    def sample(i, s, train=True):
        pool = s["pi"] + s["si"]
        k = rng.randint(1, min(8, len(pool)))
        P = rng.sample(pool, k)
        if rng.random() < 0.7 and s["pi"] and s["pi"][0] not in P:  # usually include the topic name
            P[0] = s["pi"][0]
        nn_ = 0 if a.no_neg_phrases else rng.randint(0, 3)
        N, near_neg = [], []
        if nn_ and train:
            cs = rng.sample(list(conf[i]), nn_)
            for c in cs:
                cs_ = sets[c]
                N.append(rng.choice(cs_["pi"]))
                near_neg += [m for m in cs_["mi"] if m not in s["mset"]]
        mem = rng.sample(s["mi"], min(24, len(s["mi"])))
        hn = rng.sample(hard[i], min(24, len(hard[i])))
        nn_e = rng.sample(near_neg, min(8, len(near_neg)))
        rnd = rng.sample(pool_lists[s["src"]], 64 - len(mem) - len(hn) - len(nn_e))
        rnd = [x for x in rnd if x not in s["mset"]]
        ids = mem + hn + nn_e + rnd
        y = [1.0] * len(mem) + [0.0] * (len(ids) - len(mem))
        isneg = [0.0] * (len(mem) + len(hn)) + [1.0] * len(nn_e) + [0.0] * len(rnd)
        return P, N, ids, y, isneg

    def collate(items):
        B = len(items)
        pm = max(len(x[0]) for x in items)
        nm = max(len(x[1]) for x in items)
        em = max(len(x[2]) for x in items)
        # build padded index arrays in numpy, then one host-to-device copy each (the old per-row
        # torch.tensor calls made this phase CPU bound with the GPU mostly idle)
        Pi = np.zeros((B, pm), np.int64)
        Pm = np.zeros((B, pm), bool)
        Ni = np.zeros((B, max(nm, 1)), np.int64)
        Nm = np.zeros((B, max(nm, 1)), bool)
        Ei = np.zeros((B, em), np.int64)
        Y = np.zeros((B, em), np.float32)
        W = np.zeros((B, em), np.float32)
        NG = np.zeros((B, em), np.float32)
        for b, (p, n, ids, y, ng) in enumerate(items):
            Pi[b, : len(p)] = p
            Pm[b, : len(p)] = True
            Ni[b, : len(n)] = n
            Nm[b, : len(n)] = True
            Ei[b, : len(ids)] = ids
            Y[b, : len(y)] = y
            W[b, : len(y)] = 1.0
            NG[b, : len(ng)] = ng
        t = lambda x: torch.from_numpy(x).to(dev, non_blocking=True)  # noqa: E731
        Pm_t, Nm_t = t(Pm), t(Nm)
        P = PVt[t(Pi)] * Pm_t.unsqueeze(-1)
        N = PVt[t(Ni)] * Nm_t.unsqueeze(-1)
        if nm == 0:
            N, Nm_t = N[:, :0], Nm_t[:, :0]
        return P, Pm_t, N, Nm_t, Et[t(Ei)], t(Y), t(W), t(NG)

    tset = {i: teacher.get(s["name"]) for i, s in enumerate(sets) if s["src"] == "folder" and teacher.get(s["name"])}
    print(f"teacher lists for {len(tset)} sets", flush=True)
    log = open(run / "log.jsonl", "a")
    import copy

    best = rv_eval()
    best_state, best_step = copy.deepcopy(net.state_dict()), 0
    print("region-val step 0", json.dumps(best), flush=True)
    net.train()
    ema = None
    for step in range(1, a.steps + 1):
        bi = [rng.randrange(len(sets)) for _ in range(a.bs)]
        items = [sample(i, sets[i]) for i in bi]
        P, Pm, N, Nm, E, Y, W, NG = collate(items)
        if net.training:
            P = F.normalize(P + 0.02 * torch.randn_like(P), dim=-1) * Pm.unsqueeze(-1)
        A, tau, b, fx = net(P, Pm, N, Nm)
        m = membership(E, A, tau, b, fx)
        pw = (1 - Y) * 1.0 + Y * 2.0  # members are rarer, upweight
        l_bce = (F.binary_cross_entropy_with_logits(m, Y, reduction="none") * W * pw).sum() / W.sum()
        l_neg = (F.softplus(m) * NG).sum() / NG.sum().clamp(min=1)
        l_kd = torch.tensor(0.0, device=dev)
        kd_rows = [(j, i) for j, i in enumerate(bi) if i in tset]
        if kd_rows and a.kd_weight > 0:
            terms = []
            for j, i in kd_rows[:16]:
                t = tset[i]
                cid = [idx[c] for c in t if c in idx]
                tv = torch.tensor([t[emails[c]["id"]] for c in cid], device=dev)
                mm = membership(Et[cid].unsqueeze(0), A[j : j + 1], tau[j : j + 1], b[j : j + 1], slice_fx(fx, j))[0]
                terms.append(F.kl_div(F.log_softmax(mm, -1), F.log_softmax(tv, -1), log_target=True, reduction="sum"))
            l_kd = torch.stack(terms).mean()
        loss = l_bce + a.neg_weight * l_neg + a.kd_weight * l_kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        lv = float(loss.detach())
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step % 100 == 0:
            rec = {"step": step, "loss": round(float(loss), 4), "ema": round(ema, 4), "bce": round(float(l_bce), 4),
                   "neg": round(float(l_neg), 4), "kd": round(float(l_kd), 4), "tau": round(float(tau.mean()), 2),
                   "b": round(float(b.mean()), 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % 250 == 0 or step == a.steps:
            v = rv_eval()
            print(f"region-val step {step}", json.dumps(v), flush=True)
            if v["score"] > best["score"]:
                best, best_state, best_step = v, copy.deepcopy(net.state_dict()), step
    print(f"best region-val step {best_step}", json.dumps(best), flush=True)
    net.load_state_dict(best_state)

    # calibration: score all val emails for sampled val sets, fit scalar T by NLL
    net.eval()
    ms, ys = [], []
    with torch.no_grad():
        for s in val_sets[:300]:
            pool = s["pi"] + s["si"]
            for _ in range(2):
                k = rng.randint(1, min(8, len(pool)))
                P = PVt[rng.sample(pool, k)].unsqueeze(0)
                A, tau, b, fx = net(P, torch.ones(1, k, dtype=torch.bool, device=dev), None, None)
                m = membership(Et[val_rows], A, tau, b, fx)[0]
                y = torch.from_numpy(np.isin(np.asarray(vlist), list(s["mset"])).astype(np.float32)).to(dev)
                ms.append(m.cpu())
                ys.append(y.cpu())
    M = torch.cat(ms)
    Yv = torch.cat(ys)
    # sets are sampled roughly balanced in training, the inbox is not: fit a temperature T and a
    # scalar offset on the natural val distribution, P(member) = sigmoid((m - shift) / T)
    from train.metrics import ece, fit_platt

    sa, sc = fit_platt(Yv.numpy(), M.numpy())
    best_T, shift = 1.0 / max(sa, 1e-6), -sc / max(sa, 1e-6)
    best_nll = float(F.binary_cross_entropy_with_logits((M - shift) / best_T, Yv))
    p = torch.sigmoid((M - shift) / best_T).numpy()
    cal = {"T": best_T, "shift": shift, "nll": best_nll, "val_auroc": auroc(Yv.numpy(), M.numpy()), "val_ece": ece(Yv.numpy(), p),
           "val_ece_T1": ece(Yv.numpy(), torch.sigmoid(M).numpy()), "n": int(len(Yv)), "pos_rate": float(Yv.mean())}
    print("calibration", json.dumps(cal), flush=True)
    out = Path(a.out) if a.out else mp / "region.pt"
    net_cpu = net.cpu()
    model = LearnedRegionModel(net_cpu, best_T, shift=shift, meta={"run": a.run, "encoder": str(mp), "calibration": cal,
                                                 "steps": a.steps, "best_step": best_step, "region_val": best, "args": vars(a)})
    model.save(out)
    (run / "done.json").write_text(json.dumps({"out": str(out), "calibration": cal}, indent=1))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
