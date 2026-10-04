"""Stage A: train atlas-embed (frozen bge-base + LoRA) with InfoNCE + Matryoshka + listwise
distillation from a frozen cross-encoder teacher.

Loss per Matryoshka dim d in --dims (embeddings truncated to d and renormalized):
  InfoNCE over in-batch docs + mined hard negatives (false negatives masked: same label,
  same thread, same doc), scale 20 (temperature 0.05)
  + kd_weight * KL(softmax(teacher logits / 1) || softmax(student logits)) over each anchor's
  own candidates (positive + its hard negatives), when teacher.json exists.
Batches are homogeneous by anchor kind so 'subject' batches (docs rendered without the subject)
cannot be solved by spotting the missing subject line.

usage: uv run python -m train.stage_a --run a1 [--ds enron] [--no-distill] [--no-negs] [--no-matryoshka]
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from atlas.model.encoder import BGE_BASE, QUERY_PREFIX
from train.common import DATASETS, MODELS, RUNS, doc_text, read_jsonl
from train.metrics import retrieval

LORA_TARGETS = r".*attention\.(self\.(query|key|value)|output\.dense)"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ds", default="enron", help="comma separated dataset names under data/datasets")
    ap.add_argument("--out", default=None, help="checkpoint dir, default models/<run>")
    ap.add_argument("--init-adapter", default=None, help="continue from an existing checkpoint dir (stage C)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--nneg", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--dims", default="768,256,64")
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--q-len", type=int, default=64)
    ap.add_argument("--doc-len", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--no-negs", action="store_true")
    ap.add_argument("--no-matryoshka", action="store_true")
    ap.add_argument("--kinds", default="", help="restrict anchor kinds, comma separated")
    ap.add_argument("--kind-weights", default="topic:1,subject:1,reply:0.5,grok_topic:2,grok_query:1.5")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Data:
    def __init__(self, names, use_teacher):
        self.emails, self.anchors, self.mined, self.teacher = {}, [], {}, {}
        self.val = []
        for n in names:
            d = DATASETS / n
            for e in read_jsonl(d / "emails.jsonl"):
                self.emails[e["id"]] = e
                if e["split"] == "val":
                    self.val.append(e)
            self.anchors += read_jsonl(d / "anchors.jsonl")
            self.mined.update(json.loads((d / "mined.json").read_text()))
            if use_teacher and (d / "teacher.json").exists():
                self.teacher.update(json.loads((d / "teacher.json").read_text()))
        self.anchors = [x for x in self.anchors if x["pos"] in self.emails]

    def negs(self, x, n, rng):
        if n == 0:
            return []
        pos = self.emails[x["pos"]]
        t = self.teacher.get(x["q"])
        tpos = t.get(x["pos"]) if t else None
        out = []
        cands = self.mined.get(x["q"], [])
        start = 0 if tpos is not None else 2  # without a teacher, skip the top ranks (likely false negs)
        for c in cands[start:]:
            e = self.emails.get(c)
            if e is None or c == x["pos"] or e["thread_id"] == pos["thread_id"]:
                continue
            if x.get("label") and x["label"] in e.get("topics", []):
                continue
            if tpos is not None and (c not in t or t[c] >= tpos):  # unscored, or teacher says as relevant as the positive  # teacher says it is as relevant as the positive
                continue
            out.append(c)
        if len(out) >= n:
            return rng.sample(out[:30], n) if len(out[:30]) >= n else out[:n]
        # pad with random docs
        allids = self._allids if hasattr(self, "_allids") else None
        if allids is None:
            self._allids = allids = [i for i, e in self.emails.items() if e["split"] == "train"]
        while len(out) < n:
            out.append(rng.choice(allids))
        return out


def make_batches(anchors, bs, rng, weights):
    by = defaultdict(list)
    for x in anchors:
        by[x["kind"]].append(x)
    batches = []
    for k, xs in by.items():
        rng.shuffle(xs)
        w = weights.get(k, 1.0)
        n = int(len(xs) * w)
        xs = (xs * math.ceil(max(w, 1)))[:n]
        for i in range(0, len(xs) - bs + 1, bs):
            batches.append(xs[i : i + bs])
    rng.shuffle(batches)
    return batches


def val_eval(enc, data, max_docs=6000):
    rng = random.Random(1)
    docs = data.val[:]
    rng.shuffle(docs)
    docs = docs[:max_docs]
    D = enc.encode_docs([doc_text(e) for e in docs])
    res = {}
    labels = sorted({l for e in docs for l in e["topics"]})
    rel = [{j for j, e in enumerate(docs) if l in e["topics"]} for l in labels]
    keep = [i for i, r in enumerate(rel) if len(r) >= 3]
    if keep:
        Q = enc.encode_queries([labels[i] for i in keep])
        res["topic"] = retrieval(Q @ D.T, [rel[i] for i in keep])
    subj = [j for j, e in enumerate(docs) if len(e["subject"]) >= 8 and not e["is_reply"]][:1500]
    if subj:
        Dn = enc.encode_docs([doc_text(docs[j], "nosubj") for j in range(len(docs))])
        Q = enc.encode_queries([docs[j]["subject"] for j in subj])
        res["subject"] = retrieval(Q @ Dn.T, [{j} for j in subj])
    res["score"] = float(np.mean([v["ndcg@10"] for v in res.values() if isinstance(v, dict)]))
    return res


def main():
    a = parse()
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    from atlas.model.trained import HFEncoder, cls_embed

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    run = RUNS / a.run
    run.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else MODELS / a.run
    out.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = open(run / "log.jsonl", "a")

    data = Data(a.ds.split(","), use_teacher=not a.no_distill)
    anchors = data.anchors
    if a.kinds:
        ks = set(a.kinds.split(","))
        anchors = [x for x in anchors if x["kind"] in ks]
    weights = {k: float(v) for k, v in (p.split(":") for p in a.kind_weights.split(","))}
    print(f"anchors {len(anchors)}, teacher queries {len(data.teacher)}, val emails {len(data.val)}", flush=True)
    dims = [768] if a.no_matryoshka else [int(x) for x in a.dims.split(",")]
    nneg = 0 if a.no_negs else a.nneg

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(BGE_BASE)
    base = AutoModel.from_pretrained(BGE_BASE)
    if a.init_adapter:
        model = PeftModel.from_pretrained(base, Path(a.init_adapter) / "adapter", is_trainable=True)
    else:
        cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
        model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(dev)

    batches_per_epoch = len(make_batches(anchors, a.bs, random.Random(0), weights))
    total = a.max_steps or int(batches_per_epoch * a.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * a.warmup), total)
    enc = HFEncoder(model, tok, max_len=a.doc_len, batch_size=128, device=dev)

    def save(tag):
        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        cfg = {"name": "atlas-embed", "base": BGE_BASE, "dims": [768, 256, 64], "max_len": 512, "pooling": "cls",
               "query_prefix": QUERY_PREFIX, "run": a.run, "tag": tag, "lora_r": a.r, "lora_alpha": a.alpha}
        (out / "atlas.json").write_text(json.dumps(cfg, indent=1))

    base_val = val_eval(enc, data)  # step 0 = base model (LoRA B is zero-initialized)
    print("val step 0", json.dumps(base_val), flush=True)
    log.write(json.dumps({"step": 0, "val": base_val}) + "\n")
    best = base_val["score"]
    save("step0")

    step, t0 = 0, time.time()
    ema = None
    model.train()
    while step < total:
        for batch in make_batches(anchors, a.bs, rng, weights):
            if step >= total:
                break
            B = len(batch)
            qq = batch[0]["qq"]
            qtexts = [(QUERY_PREFIX + x["q"]) if x["qq"] else x["q"] for x in batch]
            pos_ids = [x["pos"] for x in batch]
            neg_ids = [n for x in batch for n in data.negs(x, nneg, rng)]
            dids = pos_ids + neg_ids
            mode = batch[0]["mode"]
            dtexts = [doc_text(data.emails[i], mode) for i in dids]
            # mask false negatives
            mask = torch.zeros(B, len(dids), dtype=torch.bool)
            for i, x in enumerate(batch):
                pe = data.emails[x["pos"]]
                for j, did in enumerate(dids):
                    if j == i:
                        continue
                    e = data.emails[did]
                    if did == x["pos"] or e["thread_id"] == pe["thread_id"] or (x.get("label") and x["label"] in e.get("topics", [])):
                        mask[i, j] = True
            qe = tok(qtexts, padding=True, truncation=True, max_length=a.q_len if qq else a.doc_len, return_tensors="pt").to(dev)
            de = tok(dtexts, padding=True, truncation=True, max_length=a.doc_len, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Qv = model(**qe).last_hidden_state[:, 0].float()
                Dv = model(**de).last_hidden_state[:, 0].float()
            mask = mask.to(dev)
            labels = torch.arange(B, device=dev)
            l_nce, l_kd = 0.0, 0.0
            # teacher targets over each anchor's own candidates
            own = [[i] + [B + i * nneg + k for k in range(nneg)] for i in range(B)]
            tt, has = [], []
            for i, x in enumerate(batch):
                t = data.teacher.get(x["q"])
                if t is not None and nneg and all(dids[j] in t for j in own[i]):
                    tt.append([t[dids[j]] for j in own[i]])
                    has.append(i)
            for d in dims:
                q = F.normalize(Qv[:, :d], dim=-1)
                D = F.normalize(Dv[:, :d], dim=-1)
                S = (q @ D.T) * a.scale
                l_nce = l_nce + F.cross_entropy(S.masked_fill(mask, -1e4), labels)
                if has:
                    idx = torch.tensor([own[i] for i in has], device=dev)
                    s_own = S[torch.tensor(has, device=dev)[:, None], idx]
                    t_own = torch.tensor(tt, device=dev, dtype=torch.float32)
                    l_kd = l_kd + F.kl_div(F.log_softmax(s_own, -1), F.log_softmax(t_own, -1), log_target=True, reduction="batchmean")
            l_nce = l_nce / len(dims)
            l_kd = l_kd / len(dims) if has else torch.tensor(0.0, device=dev)
            loss = l_nce + a.kd_weight * l_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            lv = float(loss.detach())
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            if step % 25 == 0:
                rec = {"step": step, "total": total, "loss": round(lv, 4), "ema": round(ema, 4), "nce": round(float(l_nce), 4),
                       "kd": round(float(l_kd), 4), "kind": batch[0]["kind"], "lr": sched.get_last_lr()[0],
                       "s_per_step": round((time.time() - t0) / step, 3), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                eta = (total - step) * rec["s_per_step"] / 60
                print(json.dumps(rec), f"eta {eta:.0f}m", flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % a.eval_every == 0 or step == total:
                model.eval()
                v = val_eval(enc, data)
                model.train()
                print(f"val step {step}", json.dumps(v), flush=True)
                log.write(json.dumps({"step": step, "val": v}) + "\n")
                if v["score"] > best:
                    best = v["score"]
                    save(f"step{step}")
                    print(f"saved best {best:.4f}", flush=True)
    # merged copy for inference without peft
    from peft import PeftModel as PM

    mb = PM.from_pretrained(AutoModel.from_pretrained(BGE_BASE), out / "adapter").merge_and_unload()
    mb.save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    (run / "done.json").write_text(json.dumps({"best_val": best, "steps": step, "minutes": (time.time() - t0) / 60}))
    print("done", best, flush=True)


if __name__ == "__main__":
    main()
