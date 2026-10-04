# Gmail token-budget and encoder comparison

## Scope

Local evaluation of 200 imported Gmail messages, 34 factual questions, and 8 absent-topic probes. These questions were used in earlier development and are not an independent test set. No email contents, addresses, question text, IDs, or per-question results are included here.

Seven context budgets were tested with three encoder configurations, comparing original context packing (`8d39ce7`) with the improved branch implementation. This is 1,764 context-pack evaluations (42 cases × 7 budgets × 2 implementations × 3 encoders). Both implementations use the same current search engine. Query expansion and answer generation were disabled; no LLM API calls were made.

Evidence scoring requires every labeled fact to survive in excerpts from an expected source document. An unrelated email containing a matching value cannot receive credit. Response tokens count normalized serialized JSON including metadata, excluding MCP framing, prompts and generated answers.

## Base encoder: budget curve

| Context budget | Original evidence /34 | Improved evidence /34 | Original response tokens | Improved response tokens | Improved context tokens |
|---:|---:|---:|---:|---:|---:|
| 150 | 17 | 20 | 286.1 | 136.2 | 96.1 |
| 300 | 18 | 21 | 353.2 | 168.5 | 127.5 |
| 500 | 24 | 27 | 516.2 | 249.5 | 206.7 |
| 800 | 24 | 27 | 544.5 | 262.6 | 219.4 |
| 1000 | 24 | 27 | 561.8 | 270.2 | 226.9 |
| 1500 | 25 | 28 | 1670.6 | 816.4 | 759.5 |
| 2500 | 25 | 28 | 1734.1 | 855.9 | 798.3 |

Token counts are means over the 34 factual questions. At 800 tokens, the improved response uses 51.8% fewer tokens than the original while retaining all required evidence for 27 rather than 24 questions. On this sample, 500 and 800 budgets have identical complete-evidence hits; 500 saves a further 5.0% in mean response tokens. Going to 1,500 retains one more complete answer but more than triples the mean response size versus 500.

## Encoder tradeoffs with improved packing

| Encoder | Dimensions | Budget | Evidence /34 | Mean response tokens | Correct absent abstentions /8 | False abstentions /34 |
|---|---:|---:|---:|---:|---:|---:|
| base | 768 | 500 | 27 | 249.5 | 7 | 2 |
| base | 768 | 800 | 27 | 262.6 | 7 | 2 |
| base | 768 | 1500 | 28 | 816.4 | 7 | 2 |
| atlas-embed | 768 | 500 | 23 | 185.1 | 8 | 2 |
| atlas-embed | 768 | 800 | 23 | 195.4 | 8 | 2 |
| atlas-embed | 768 | 1500 | 26 | 682.3 | 8 | 2 |
| atlas-embed-256 | 256 | 500 | 23 | 172.3 | 8 | 5 |
| atlas-embed-256 | 256 | 800 | 23 | 174.4 | 8 | 5 |
| atlas-embed-256 | 256 | 1500 | 24 | 545.8 | 8 | 5 |

The base encoder remains the default: the trained encoder produces shorter responses but retains fewer complete answers here. The 256-dimensional trained encoder needs one third as much raw vector storage as a 768-dimensional encoder, but its additional false abstentions are a material tradeoff. Reduced vector size is not a threefold reduction in model inference cost or response tokens.

## Larger-context reference methods

| Base-encoder strategy | Evidence /34 | Mean context tokens |
|---|---:|---:|
| keyword docs | 34 | 3169.1 |
| embedding docs | 34 | 2640.9 |
| embedding chunks | 33 | 1023.3 |

These reference strategies have different retrieval limits (10 full documents or 8 chunks of roughly 120 words), so smaller context should not be presented as equal accuracy. Full documents retained more evidence. Required-evidence retention is not generated-answer accuracy, and token counts are not total API cost.

All 882 improved-pack evaluations respected their requested context budget. Abstention performance is based on only eight absent topics. The corpus is a recent sample, not the whole mailbox; these results cannot establish mailbox-wide absence or a general performance guarantee.

## Reproduction

See [PRIVATE_EVAL.md](PRIVATE_EVAL.md) for the importer, manifest format, source-aware scoring, and commands. Use `--budgets 150 300 500 800 1000 1500 2500`. Run with `ATLAS_ENCODER=base`, `ATLAS_ENCODER=models/atlas-embed`, and the trained model plus `ATLAS_DIM=256`. Keep corpus, manifest, checkpoint, and search revision fixed. The shipped trained checkpoint is `a2/step6148` (BGE base with merged LoRA). Private local reports retain manifest and code hashes.

The private corpus and question manifests are intentionally excluded, so the exact private results cannot be independently reproduced from this repository alone. Use the evaluator on an authorized corpus and freeze new questions before tuning.

A separate base-encoder control compared the last uploaded pack (`1556f50`) with this round: all 294 query/budget pairs had identical contexts, evidence scores, and response-token counts. This round adds testing coverage and fixes empty-inbox handling; the measured extraction/token gains originated in the earlier pack improvement.
