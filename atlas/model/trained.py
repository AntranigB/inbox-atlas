"""atlas-embed: frozen bge-base backbone + LoRA adapter, Matryoshka dims, CLS pooling.

A checkpoint dir (models/atlas-embed/) holds:
  atlas.json   {"base", "dims", "max_len", "pooling", ...}
  adapter/     peft LoRA adapter (needs peft to load)
  merged/      optional full model with LoRA merged in (needs only transformers)
  region.pt    optional learned region encoder (atlas/model/region_encoder.py)
"""

import json
from pathlib import Path

import numpy as np

from atlas.model.encoder import BGE_BASE, QUERY_PREFIX, Encoder, _normalize


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cls_embed(model, enc):
    """CLS pooling as bge does, L2 normalized. Returns a torch tensor."""
    import torch.nn.functional as F

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return F.normalize(out.last_hidden_state[:, 0].float(), dim=-1)


def truncate(x, dim):
    """Matryoshka truncation followed by renormalization (numpy)."""
    if not dim or dim >= x.shape[1]:
        return x
    return _normalize(x[:, :dim])


class HFEncoder(Encoder):
    def __init__(self, model, tokenizer, dim=None, name="atlas-embed", max_len=512, batch_size=64, device=None):
        import torch

        self.model = model.eval()
        self.tok = tokenizer
        self.device = device or next(model.parameters()).device
        self.full_dim = model.config.hidden_size
        self.dim = dim or self.full_dim
        self.name = name if not dim else f"{name}-{dim}"
        self.max_len = max_len
        self.batch_size = batch_size
        self.amp = str(self.device).startswith("cuda")
        self._torch = torch

    def _encode(self, texts):
        torch = self._torch
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        order = np.argsort([-len(t) for t in texts])  # length bucketing for speed
        out = np.zeros((len(texts), self.full_dim), np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                idx = order[i : i + self.batch_size]
                enc = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    v = cls_embed(self.model, enc)
                out[idx] = v.cpu().numpy()
        return truncate(out, self.dim)

    def encode_docs(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode([QUERY_PREFIX + t for t in texts])


def load_base(model_name=BGE_BASE, dim=None, device=None, name="base", **kw):
    from transformers import AutoModel, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return HFEncoder(model, tok, dim=dim, name=name, device=device, **kw)


def load(path, dim=None, device=None, **kw):
    """Load a trained atlas-embed checkpoint dir. Merges LoRA into the base for inference."""
    from transformers import AutoModel, AutoTokenizer

    path = Path(path)
    cfg = json.loads((path / "atlas.json").read_text())
    device = device or pick_device()
    base = cfg.get("base", BGE_BASE)
    if (path / "merged").exists():
        model = AutoModel.from_pretrained(path / "merged")
        tok = AutoTokenizer.from_pretrained(path / "merged")
    else:
        from peft import PeftModel

        model = AutoModel.from_pretrained(base)
        model = PeftModel.from_pretrained(model, path / "adapter").merge_and_unload()
        tok = AutoTokenizer.from_pretrained(base)
    model = model.to(device)
    return HFEncoder(model, tok, dim=dim, name=cfg.get("name", "atlas-embed"),
                     max_len=cfg.get("max_len", 512), device=device, **kw)
