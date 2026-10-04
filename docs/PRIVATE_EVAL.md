# Local retrieval and token evaluation

`eval/private_eval.py` compares keyword search, full-document embedding search, embedding chunks,
an archived context pack, and the current context pack at 300/800/1500-token budgets.
It makes no LLM calls: embeddings run locally, facets are disabled, and required facts are checked
against a question manifest. A first run may download the public embedding model.

## Prepare a private corpus

Keep exports, questions, and per-question results under the gitignored `data/` directory. Use
`ATLAS_DATA` to isolate the benchmark from your normal inbox database. For Gmail API or connector
MIME-tree exports (snake_case and camelCase are supported):

```bash
mkdir -p data/private-eval
chmod 700 data/private-eval
ATLAS_DATA=data/private-eval uv run python -m atlas.ingest.gmail_json data/private-eval/messages.json
git show HEAD:atlas/context/pack.py > data/private-eval/baseline_pack.py
```

Archive the baseline before making changes, or replace HEAD with the original revision. The JSON
importer reads existing exports only; it does not access Gmail, mark messages read, or download
attachments. Use only exports you intend to include. A query scoped to a recent sample tests that
sample, not the whole mailbox. Do not infer mailbox-wide absence from a sampled corpus.

## Freeze the question manifest

Write questions and expected evidence before comparing retrieval results. Questions should include
paraphrases, dates, amounts, multi-fact requests, and absent topics. For example (fictional data):

```json
{
  "sources": ["gmail"],
  "questions": [
    {
      "q": "Where and when is the robotics workshop?",
      "targets": ["fixture-workshop-message"],
      "evidence": ["Room 205", "November 12", "2pm"]
    }
  ],
  "absent": ["What is my yacht registration number?"]
}
```

Each evidence entry is a case-insensitive Python regular expression; **all** entries must match.
Use alternatives for equivalent wording/formatting. `targets` contains one or more acceptable
email IDs, or `obs:<note path>` for Obsidian. The runner verifies facts exist in the specified
source documents before retrieving. Absent topics should be checked against the sampled corpus.

```bash
ATLAS_DATA=data/private-eval ATLAS_ENCODER=base ATLAS_DB=sqlite \
  uv run python -m eval.private_eval \
  --questions data/private-eval/questions.json \
  --baseline data/private-eval/baseline_pack.py \
  --out data/private-eval/results.json
```

## Interpret the results

- `target_hits`: at least one labeled source document appears in the returned context.
- `evidence_hits`: a target was retrieved and all required fact patterns occur in the context.
  This is an evidence-retention proxy, not an end-to-end answer correctness score. Matching facts
  are not guaranteed to belong to the same source, and patterns can miss valid paraphrases.
- `mean_context_tokens`: context text only, including its rendered citations.
- `mean_payload_tokens`: serialized compact JSON tool result, including metadata; excludes MCP
  framing, request prompts, query expansion, and answer generation. The original payload repeats
  excerpts in `context` and `items`; the current MCP payload emits them once.
- `absent_abstentions` and `present_abstentions`: show both successful abstentions and false negatives.

No model API cost comparison is claimed. All methods use the same imported, cleaned corpus and
base encoder. Cached embeddings only reuse identical inputs across methods. The full-document
and chunk baselines match the repository's existing evaluator (top 10 documents and top 8
roughly 120-word chunks). Their budgets differ; report both tokens and evidence retained.
If you revise the implementation after inspecting failures, those questions are development data;
use new questions or another mailbox/time period for an independent follow-up test.
