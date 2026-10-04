---
base_model: BAAI/bge-base-en-v1.5
library_name: peft
tags:
- lora
- sentence-embeddings
- email
- retrieval
---

# atlas-embed (LoRA adapter)

The trained encoder behind Inbox Atlas. A LoRA adapter (r=16, alpha=32, 1.18M trainable
parameters) on a frozen `BAAI/bge-base-en-v1.5` backbone, trained on 224,867 deduplicated Enron
emails with InfoNCE on mined hard negatives, Matryoshka heads at 768, 256 and 64 dims, and listwise
distillation from a frozen `BAAI/bge-reranker-v2-m3` cross-encoder.

Use it from the repo root:

```bash
ATLAS_ENCODER=models/atlas-embed uv run python -m atlas.index.build --encoder models/atlas-embed
ATLAS_ENCODER=models/atlas-embed uv run atlas up
```

Results on unseen emails (nDCG@10): vague queries 0.692 (base 0.635), subject to body 0.408
(base 0.342). Full training details are in TRAINING.md.
