"""Local retrieval/evidence benchmark with a frozen, user-supplied question manifest.

No LLM calls. Reports target-document retrieval and required evidence retention,
NOT generated-answer accuracy. Compare the current pack with an archived pack.py:

 ATLAS_DATA=data/private uv run python -m eval.private_eval \
   --questions data/private/questions.json --baseline data/private/baseline_pack.py \
   --out data/private/results.json

Manifest: {"sources": ["gmail"], "questions": [{"q": "...", "targets": ["message id"],
"evidence": ["regex for fact 1", "regex for fact 2"]}], "absent": ["topic", ...]}.
All required facts must appear, with at least one target document retrieved. Include
alternative valid source IDs in targets. Keep private manifests/results under data/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from atlas.context import pack
from atlas.search.engine import get_engine
from eval.token_eval import Corpus, strat_chunks, strat_embed_docs, strat_keyword


class CachedEncoder:
    """Reuse identical embeddings across strategies; no change to model/scoring."""

    def __init__(self, enc):
        self.enc, self.name, self.dim = enc, enc.name, enc.dim
        self.cache = {"docs": {}, "queries": {}}

    def _encode(self, texts, kind):
        texts = list(texts)
        cache = self.cache[kind]
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            values = getattr(self.enc, "encode_" + kind)(missing)
            cache.update(zip(missing, values))
        return np.asarray([cache[t] for t in texts], np.float32).reshape(-1, self.dim)

    def encode_docs(self, texts):
        return self._encode(texts, "docs")

    def encode_queries(self, texts):
        return self._encode(texts, "queries")


def evidence_hit(ctx, docs, item):
    return bool(set(docs) & set(item["targets"])) and all(
        re.search(pattern, ctx, re.I) is not None for pattern in item["evidence"])


def old_payload(p):
    return {k: p.get(k) for k in ("answerable", "confidence", "reason", "context", "items", "tokens",
                                  "tokens_saved_vs_naive")} | {
        "region": {k: p["region"].get(k) for k in ("facets", "size", "max_z")}}


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("atlas_baseline_pack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(eng, manifest, baseline, budgets=(300, 800, 1500)):
    eng.enc = CachedEncoder(eng.enc)
    cp = Corpus(eng)
    sources = tuple(manifest.get("sources", ["gmail"]))
    rows = []
    cases = [(q, False) for q in manifest["questions"]] + [
        ({"q": q, "targets": [], "evidence": []}, True) for q in manifest.get("absent", [])]
    # Validate the frozen labels against the corpus before looking at retrieval.
    for item in manifest["questions"]:
        targets = item["targets"]
        if not targets or not item["evidence"]:
            raise ValueError("Each present question needs targets and required evidence")
        if not evidence_hit("\n".join(cp.text.get(t, "") for t in targets), targets, item):
            raise ValueError("A manifest's required evidence is absent from its source documents")
    for number, (item, absent) in enumerate(cases, 1):
        q, results = item["q"], {}
        for name, func in (("keyword docs", strat_keyword), ("embedding docs", strat_embed_docs),
                           ("embedding chunks", strat_chunks)):
            docs, ctx = func(cp, q, sources)
            results[name] = {"context_tokens": pack.count_tokens(ctx), "target_hit": bool(set(docs) & set(item["targets"])),
                             "evidence_hit": evidence_hit(ctx, docs, item) if not absent else None,
                             "abstained": not bool(docs)}
        for version, module in (("original", baseline), ("improved", pack)):
            for budget in budgets:
                p = module.build_context(q, budget_tokens=budget, sources=sources, engine=eng, use_grok=False)
                docs = list(dict.fromkeys(cp.doc_of.get("obs:" + it["uri"] if it["source"] == "obsidian" else it["uri"][6:])
                                         for it in p["items"]))
                payload = old_payload(p) if version == "original" else pack.tool_payload(p)
                results[f"{version} @{budget}"] = {
                    "context_tokens": pack.count_tokens(p["context"]),
                    "payload_tokens": pack.count_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
                    "target_hit": bool(set(docs) & set(item["targets"])),
                    "evidence_hit": evidence_hit(p["context"], docs, item) if not absent else None,
                    "abstained": not p["answerable"], "budget_exceeded": pack.count_tokens(p["context"]) > budget,
                    "context": p["context"], "docs": docs}
        rows.append({"question": q, "absent": absent, "strategies": results})
        print(f"Evaluated {number}/{len(cases)}", flush=True)
    summary = {}
    for name in rows[0]["strategies"]:
        present = [r["strategies"][name] for r in rows if not r["absent"]]
        absent = [r["strategies"][name] for r in rows if r["absent"]]
        summary[name] = {"mean_context_tokens": float(np.mean([r["context_tokens"] for r in present])),
                         "target_hits": sum(r["target_hit"] for r in present),
                         "evidence_hits": sum(r["evidence_hit"] for r in present),
                         "present_count": len(present), "absent_count": len(absent),
                         "absent_abstentions": sum(r["abstained"] for r in absent),
                         "present_abstentions": sum(r["abstained"] for r in present)}
        if "payload_tokens" in present[0]:
            summary[name]["mean_payload_tokens"] = float(np.mean([r["payload_tokens"] for r in present]))
            summary[name]["budget_exceeded"] = sum(r["budget_exceeded"] for r in present + absent)
    return {"encoder": eng.name, "documents": len(cp.ids), "llm_calls": 0,
            "metric": "required evidence retention, not generated-answer accuracy", "summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.questions).read_text())
    result = run(get_engine(), manifest, load_baseline(args.baseline))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    out.chmod(0o600)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
