"""Encoder loading. The training branch adds the trained LoRA + Matryoshka checkpoint loader."""

import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from atlas import config

BGE_BASE = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Torch on Apple MPS aborts the whole process ("A command encoder is already encoding to this
# command buffer") when two threads run the model at once, and FastAPI runs sync routes on a
# thread pool. Every model forward pass goes through this one worker thread instead.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas-encode")
_on_worker = threading.local()


def serial(fn, *args, **kw):
    """Run fn on the single encoder thread (directly if already on it)."""
    if getattr(_on_worker, "yes", False):
        return fn(*args, **kw)

    def run():
        _on_worker.yes = True
        return fn(*args, **kw)

    return _worker.submit(run).result()


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class Encoder:
    name = "encoder"
    dim = 0

    def encode_docs(self, texts):
        raise NotImplementedError

    def encode_queries(self, texts):
        raise NotImplementedError


class HashEncoder(Encoder):
    """Deterministic bag-of-words hashing. Only for offline tests."""

    name = "hash"

    def __init__(self, dim=256):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v

    def encode_docs(self, texts):
        return _normalize([self._vec(t) for t in texts])

    encode_queries = encode_docs


class STEncoder(Encoder):
    def __init__(self, model_name=BGE_BASE, dim=None, name="base"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, truncate_dim=dim)
        self.dim = dim or self.model.get_sentence_embedding_dimension()
        self.name = name if not dim else f"{name}-{dim}"

    def encode_docs(self, texts):
        return _normalize(serial(self.model.encode, list(texts), batch_size=64, normalize_embeddings=True))

    def encode_queries(self, texts):
        return self.encode_docs([QUERY_PREFIX + t for t in texts])


_cache = {}


def load_encoder(spec=None):
    spec = spec or config.ENCODER
    if spec in _cache:
        return _cache[spec]
    if spec == "hash":
        enc = HashEncoder()
    elif spec == "base":
        enc = STEncoder(BGE_BASE, config.DIM)
    else:
        from atlas.model import trained  # added by the training branch

        enc = trained.load(spec, config.DIM)
    _cache[spec] = enc
    return enc
