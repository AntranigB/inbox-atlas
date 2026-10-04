"""atlas-region: a Set Transformer that reads a SET of topic-phrase embeddings (positives and
negatives, each tagged with a learned type embedding) and outputs a region in embedding space.

Architecture (Lee et al. 2019, "Set Transformer"): input projection + type embedding,
2 ISAB layers (16 inducing points), PMA with K=4 seeds. Heads on the K pooled vectors give
  a_k   unit anchors (residual around the normalized positive mean)
  tau_k per-anchor temperatures (softplus)
  b     a scalar boundary
Membership logit for an email embedding e:  m(e) = logsumexp_k(tau_k * cos(e, a_k)) - b.
A scalar temperature T fit on held-out data makes sigmoid(m / T) a calibrated P(member).

Inference is numpy in, numpy out, and runs on CPU or MPS:
    model = LearnedRegionModel.load("models/atlas-embed/region.pt")
    region = model.build(pos_vecs, neg_vecs)
    region.score(E); region.prob(E); region.describe()
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, h, heads):
        super().__init__()
        self.att = nn.MultiheadAttention(h, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))

    def forward(self, q, kv, kv_pad=None):
        x = self.ln1(q + self.att(q, kv, kv, key_padding_mask=kv_pad, need_weights=False)[0])
        return self.ln2(x + self.ff(x))


class ISAB(nn.Module):
    def __init__(self, h, heads, m):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, h) / math.sqrt(h))
        self.mab0 = MAB(h, heads)
        self.mab1 = MAB(h, heads)

    def forward(self, x, pad):
        H = self.mab0(self.I.expand(x.shape[0], -1, -1), x, pad)
        return self.mab1(x, H)


class PMA(nn.Module):
    def __init__(self, h, heads, k):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, h) / math.sqrt(h))
        self.mab = MAB(h, heads)

    def forward(self, x, pad):
        return self.mab(self.S.expand(x.shape[0], -1, -1), x, pad)


class SetRegionNet(nn.Module):
    def __init__(self, d=768, h=256, heads=4, m=16, k=4, tau0=20.0, b0=16.0):
        super().__init__()
        self.cfg = dict(d=d, h=h, heads=heads, m=m, k=k, tau0=tau0, b0=b0)
        self.inp = nn.Linear(d, h)
        self.kind_emb = nn.Embedding(2, h)  # 0 = positive phrase, 1 = negative phrase
        self.isab = nn.ModuleList([ISAB(h, heads, m), ISAB(h, heads, m)])
        self.pma = PMA(h, heads, k)
        self.anchor = nn.Linear(h, d)
        self.tau = nn.Linear(h, 1)
        self.bias = nn.Linear(h, 1)
        nn.init.zeros_(self.anchor.weight)
        nn.init.zeros_(self.anchor.bias)
        nn.init.zeros_(self.tau.weight)
        nn.init.constant_(self.tau.bias, math.log(math.expm1(tau0)))
        nn.init.zeros_(self.bias.weight)
        nn.init.constant_(self.bias.bias, b0)

    def forward(self, P, Pm, N, Nm):
        """P: (B, p, d) positive phrase vecs, Pm: (B, p) bool valid. N, Nm likewise (n may be 0).
        Returns anchors (B, K, d) unit, tau (B, K), b (B,)."""
        x = self.inp(P) + self.kind_emb.weight[0]
        valid = Pm
        if N is not None and N.shape[1] > 0:
            x = torch.cat([x, self.inp(N) + self.kind_emb.weight[1]], 1)
            valid = torch.cat([Pm, Nm], 1)
        pad = ~valid
        for layer in self.isab:
            x = layer(x, pad)
        z = self.pma(x, pad)  # (B, K, h)
        w = Pm.float().unsqueeze(-1)
        center = F.normalize((F.normalize(P, dim=-1) * w).sum(1) / w.sum(1).clamp(min=1), dim=-1)
        a = F.normalize(center.unsqueeze(1) + self.anchor(z), dim=-1)
        tau = F.softplus(self.tau(z)).squeeze(-1) + 1.0
        b = self.bias(z.mean(1)).squeeze(-1)
        return a, tau, b


def membership(E, a, tau, b):
    """E: (B, n, d) or (n, d) shared; returns (B, n) logits m(e)."""
    E = F.normalize(E, dim=-1)
    if E.dim() == 2:
        cos = torch.einsum("nd,bkd->bnk", E, a)
    else:
        cos = torch.einsum("bnd,bkd->bnk", E, a)
    return torch.logsumexp(tau.unsqueeze(1) * cos, -1) - b.unsqueeze(1)


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class LearnedRegion:
    kind = "learned"

    def __init__(self, anchors, tau, b, T, pos_vecs):
        self.anchors = anchors  # (K, d)
        self.tau = tau  # (K,)
        self.b = float(b)
        self.T = float(T)
        self._pos = pos_vecs

    def score(self, E):
        E = _norm(E)
        z = (E @ self.anchors.T) * self.tau
        mx = z.max(1, keepdims=True)
        return (mx[:, 0] + np.log(np.exp(z - mx).sum(1))) - self.b

    def prob(self, E):
        return 1.0 / (1.0 + np.exp(-np.clip(self.score(E) / self.T, -50, 50)))

    def describe(self):
        sims = self.anchors @ self._pos.T if len(self._pos) else np.zeros((len(self.anchors), 0))
        return {
            "kind": "learned",
            "k": int(len(self.anchors)),
            "tau": [round(float(t), 3) for t in self.tau],
            "bias": round(self.b, 3),
            "temperature": round(self.T, 3),
            "anchor_facets": [int(s.argmax()) if s.size else None for s in sims],
            "anchor_facet_cos": [round(float(s.max()), 3) if s.size else None for s in sims],
        }


class LearnedRegionModel:
    def __init__(self, net, T=1.0, meta=None, device="cpu"):
        self.net = net.eval().to(device)
        self.T = float(T)
        self.meta = meta or {}
        self.device = device
        self.dim = net.cfg["d"]

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = SetRegionNet(**ck["config"])
        net.load_state_dict(ck["state_dict"])
        return cls(net, ck.get("T", 1.0), ck.get("meta"), device)

    def save(self, path):
        torch.save({"config": self.net.cfg, "state_dict": self.net.state_dict(), "T": self.T, "meta": self.meta}, path)

    def build(self, pos_vecs, neg_vecs=None):
        P = _norm(pos_vecs)
        if P.shape[1] != self.dim:
            raise ValueError(f"region model expects dim {self.dim}, got {P.shape[1]} (do not truncate with ATLAS_DIM)")
        N = _norm(neg_vecs) if neg_vecs is not None and len(neg_vecs) else np.zeros((0, self.dim), np.float32)
        with torch.no_grad():
            Pt = torch.from_numpy(P)[None].to(self.device)
            Nt = torch.from_numpy(N)[None].to(self.device)
            a, tau, b = self.net(Pt, torch.ones(1, len(P), dtype=torch.bool, device=self.device),
                                 Nt, torch.ones(1, len(N), dtype=torch.bool, device=self.device))
        return LearnedRegion(a[0].cpu().numpy(), tau[0].cpu().numpy(), float(b[0]), self.T, P)
