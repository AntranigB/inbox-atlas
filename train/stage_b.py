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

from train.common import DATASETS, RUNS, doc_text, norm_subject, read_jsonl
from train.topic_sets import folder_sets, grok_sets, load_grok


def embed_cache(enc, emails, path):
    if path.exists():
        return np.load(path).astype(np.float32)
    E = enc.encode_docs([doc_text(e) for e in emails])
    np.save(path, E.astype(np.float16))
    return E


def build_sets(emails, topics, grok, enc, pool_split):
    """Topic sets with members restricted to emails of the given splits."""
    pool = [e for e in emails if e["split"] in pool_split]
    sets = folder_sets(pool, topics, grok)
    sets += grok_sets(pool, grok, enc.encode_queries)
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
    ap.add_argument("--out", default=None, help="region.pt path, default <model>/region.pt")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from atlas.model.region_encoder import LearnedRegionModel, SetRegionNet, membership
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
    C = torch.stack([s["c"] for s in sets])
    conf = (C @ C.T).fill_diagonal_(-1).topk(20, dim=1).indices.cpu().numpy()
    hard = []
    tr_np = train_rows.cpu().numpy()
    for i in range(0, len(sets), 256):
        sims = C[i : i + 256] @ Et[train_rows].T
        top = sims.topk(400, dim=1).indices
        for j, row in enumerate(top.cpu().numpy()):
            s = sets[i + j]
            hard.append([int(r) for r in tr_np[row] if int(r) not in s["mset"]])
    print(f"prepared sets in {time.time() - t0:.0f}s", flush=True)

    net = SetRegionNet(d=E_all.shape[1]).to(dev)
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
        rnd = rng.sample(trn, 64 - len(mem) - len(hn) - len(nn_e))
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
        D = PVt.shape[1]
        P = torch.zeros(B, pm, D, device=dev)
        Pm = torch.zeros(B, pm, dtype=torch.bool, device=dev)
        N = torch.zeros(B, nm, D, device=dev)
        Nm = torch.zeros(B, nm, dtype=torch.bool, device=dev)
        Ei = torch.zeros(B, em, dtype=torch.long, device=dev)
        Y = torch.zeros(B, em, device=dev)
        W = torch.zeros(B, em, device=dev)
        NG = torch.zeros(B, em, device=dev)
        for b, (p, n, ids, y, ng) in enumerate(items):
            P[b, : len(p)] = PVt[p]
            Pm[b, : len(p)] = True
            if n:
                N[b, : len(n)] = PVt[n]
                Nm[b, : len(n)] = True
            Ei[b, : len(ids)] = torch.tensor(ids, device=dev)
            Y[b, : len(y)] = torch.tensor(y, device=dev)
            W[b, : len(y)] = 1.0
            NG[b, : len(ng)] = torch.tensor(ng, device=dev)
        # phrase dropout as embedding noise
        return P, Pm, N, Nm, Et[Ei], Y, W, NG

    tset = {i: teacher.get(s["name"]) for i, s in enumerate(sets) if s["src"] == "folder" and teacher.get(s["name"])}
    print(f"teacher lists for {len(tset)} sets", flush=True)
    log = open(run / "log.jsonl", "a")
    net.train()
    ema = None
    for step in range(1, a.steps + 1):
        bi = [rng.randrange(len(sets)) for _ in range(a.bs)]
        items = [sample(i, sets[i]) for i in bi]
        P, Pm, N, Nm, E, Y, W, NG = collate(items)
        if net.training:
            P = F.normalize(P + 0.02 * torch.randn_like(P), dim=-1) * Pm.unsqueeze(-1)
        A, tau, b = net(P, Pm, N, Nm)
        m = membership(E, A, tau, b)
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
                mm = membership(Et[cid].unsqueeze(0), A[j : j + 1], tau[j : j + 1], b[j : j + 1])[0]
                terms.append(F.kl_div(F.log_softmax(mm, -1), F.log_softmax(tv, -1), log_target=True, reduction="sum"))
            l_kd = torch.stack(terms).mean()
        loss = l_bce + a.neg_weight * l_neg + a.kd_weight * l_kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
        if step % 100 == 0:
            rec = {"step": step, "loss": round(float(loss), 4), "ema": round(ema, 4), "bce": round(float(l_bce), 4),
                   "neg": round(float(l_neg), 4), "kd": round(float(l_kd), 4), "tau": round(float(tau.mean()), 2),
                   "b": round(float(b.mean()), 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

    # calibration: score all val emails for sampled val sets, fit scalar T by NLL
    net.eval()
    val_rows = torch.tensor([i for i, e in enumerate(emails) if e["split"] == "val"], device=dev)
    ms, ys = [], []
    with torch.no_grad():
        for s in val_sets[:300]:
            pool = s["pi"] + s["si"]
            for _ in range(2):
                k = rng.randint(1, min(8, len(pool)))
                P = PVt[rng.sample(pool, k)].unsqueeze(0)
                A, tau, b = net(P, torch.ones(1, k, dtype=torch.bool, device=dev), None, None)
                m = membership(Et[val_rows], A, tau, b)[0]
                y = torch.tensor([1.0 if int(r) in s["mset"] else 0.0 for r in val_rows.tolist()], device=dev)
                ms.append(m.cpu())
                ys.append(y.cpu())
    M = torch.cat(ms)
    Yv = torch.cat(ys)
    best_T, best_nll = 1.0, 1e9
    for lt in np.linspace(-2, 3, 101):
        T = float(math.exp(lt))
        nll = float(F.binary_cross_entropy_with_logits(M / T, Yv))
        if nll < best_nll:
            best_T, best_nll = T, nll
    from train.metrics import auroc, ece

    p = torch.sigmoid(M / best_T).numpy()
    cal = {"T": best_T, "nll": best_nll, "val_auroc": auroc(Yv.numpy(), M.numpy()), "val_ece": ece(Yv.numpy(), p),
           "val_ece_T1": ece(Yv.numpy(), torch.sigmoid(M).numpy()), "n": int(len(Yv)), "pos_rate": float(Yv.mean())}
    print("calibration", json.dumps(cal), flush=True)
    out = Path(a.out) if a.out else mp / "region.pt"
    net_cpu = net.cpu()
    model = LearnedRegionModel(net_cpu, best_T, {"run": a.run, "encoder": str(mp), "calibration": cal,
                                                 "steps": a.steps, "args": vars(a)})
    model.save(out)
    (run / "done.json").write_text(json.dumps({"out": str(out), "calibration": cal}, indent=1))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
