# Token efficiency eval

Store: 24 emails (demo inbox) + 14 synthetic Obsidian notes (123 heading sections), encoder `base`, tokens counted with tiktoken cl100k_base. 24 questions: 12 demo inbox queries (eval/queries.json) and 12 vault questions (eval/token_queries.json). Answerer and judge: Grok `grok-4.20-0309-non-reasoning`. Run 2026-10-04 01:35.

Each strategy hands the agent a context; Grok answers from only that context; a separate Grok call grades the answer 0/1 against a reference answer. The reference comes from the union of documents every strategy retrieved: Grok checks each document alone for relevant facts, then answers from the full text of the relevant ones. When either side says not found, scoring is deterministic (correct only if both do).

Caveats: 24 questions, so one question is about 4 points. A single Grok model answers, writes the reference and judges. The demo inbox queries are topic phrases ("travel plans"), whose references list several items; the vault questions ask for one fact. Keyword search ORs the question's content words over SQLite FTS5, a fair stand-in for grep but not for a tuned BM25 agent.

## All 24 questions

| Strategy | mean context tokens | median | accuracy | accuracy per 1k tokens |
|---|---|---|---|---|
| keyword top-10 docs | 4445 | 4142 | 67% | 0.15 |
| embedding top-10 docs | 4270 | 4293 | 92% | 0.21 |
| embedding top-8 chunks | 740 | 720 | 75% | 1.01 |
| atlas pack @300 | 156 | 137 | 71% | 4.53 |
| atlas pack @800 | 249 | 240 | 75% | 3.01 |
| atlas pack @1500 | 465 | 460 | 79% | 1.70 |

## Demo inbox queries (12)

| Strategy | mean context tokens | median | accuracy | accuracy per 1k tokens |
|---|---|---|---|---|
| keyword top-10 docs | 3724 | 3462 | 33% | 0.09 |
| embedding top-10 docs | 4773 | 4715 | 83% | 0.17 |
| embedding top-8 chunks | 705 | 690 | 50% | 0.71 |
| atlas pack @300 | 142 | 125 | 58% | 4.10 |
| atlas pack @800 | 225 | 125 | 58% | 2.59 |
| atlas pack @1500 | 463 | 413 | 67% | 1.44 |

## Synthetic vault questions (12)

| Strategy | mean context tokens | median | accuracy | accuracy per 1k tokens |
|---|---|---|---|---|
| keyword top-10 docs | 5165 | 5958 | 100% | 0.19 |
| embedding top-10 docs | 3767 | 3730 | 100% | 0.27 |
| embedding top-8 chunks | 776 | 760 | 100% | 1.29 |
| atlas pack @300 | 170 | 176 | 83% | 4.89 |
| atlas pack @800 | 274 | 278 | 92% | 3.35 |
| atlas pack @1500 | 467 | 500 | 92% | 1.96 |

## Calibrated 'nothing here'

Atlas abstained (answerable=false, zero items) on 93% of 14 absent topics, and returned items for 92% of the 24 present questions. Mean tokens handed to the agent for an absent topic:

| Strategy | mean tokens on absent topics |
|---|---|
| keyword top-10 docs | 1858 |
| embedding top-10 docs | 4039 |
| embedding top-8 chunks | 742 |
| atlas pack @300 | 31 |
| atlas pack @800 | 31 |
| atlas pack @1500 | 40 |

## Pack vs reading the same documents whole

When the pack answers, it averages 270 tokens against 2978 tokens for the full text of the same top-k documents (`tokens_saved_vs_naive`), a 11.0x reduction.

## Per question (tokens / correct)

| set | question | keyword top-10 docs | embedding top-10 docs | embedding top-8 chunks | atlas pack @300 | atlas pack @800 | atlas pack @1500 |
|---|---|---|---|---|---|---|---|
| inbox | coding competition | 38 / 0 | 2142 / 1 | 428 / 1 | 241 / 0 | 427 / 0 | 655 / 0 |
| inbox | money I owe someone | 3359 / 1 | 2152 / 0 | 491 / 1 | 119 / 1 | 119 / 1 | 236 / 1 |
| inbox | anything about housing next year | 7195 / 1 | 4096 / 0 | 705 / 0 | 131 / 1 | 131 / 1 | 240 / 1 |
| inbox | travel plans | 2012 / 1 | 5701 / 1 | 910 / 1 | 261 / 0 | 453 / 0 | 747 / 0 |
| inbox | upcoming exams | 4544 / 0 | 2343 / 1 | 585 / 1 | 63 / 1 | 63 / 1 | 123 / 1 |
| inbox | internship hunting | 921 / 0 | 4575 / 1 | 605 / 0 | 57 / 1 | 57 / 1 | 104 / 1 |
| inbox | stuff I bought online | 3564 / 0 | 5478 / 1 | 675 / 0 | 41 / 1 | 41 / 1 | 382 / 1 |
| inbox | workout schedule | 5927 / 0 | 4490 / 1 | 870 / 0 | 238 / 0 | 319 / 0 | 811 / 1 |
| inbox | catching up with my parents | 883 / 0 | 7865 / 1 | 718 / 1 | 167 / 1 | 391 / 1 | 908 / 1 |
| inbox | my research lab work | 7976 / 0 | 6157 / 1 | 969 / 0 | 251 / 0 | 560 / 0 | 878 / 0 |
| inbox | tech news digest | 1938 / 1 | 4855 / 1 | 553 / 1 | 110 / 1 | 110 / 1 | 444 / 1 |
| inbox | winter break trip | 6337 / 0 | 7425 / 1 | 951 / 0 | 28 / 0 | 28 / 0 | 28 / 0 |
| vault | What is my flight confirmation code for the Japan trip? | 1879 / 1 | 1521 / 1 | 783 / 1 | 261 / 1 | 565 / 1 | 803 / 1 |
| vault | How much do I owe Marcus and how do I pay him? | 7059 / 1 | 2836 / 1 | 641 / 1 | 251 / 0 | 272 / 1 | 593 / 1 |
| vault | When is my dentist appointment? | 1935 / 1 | 1501 / 1 | 619 / 1 | 28 / 0 | 28 / 0 | 28 / 0 |
| vault | When is chapter 2 of my thesis due to Prof. Okafor? | 4858 / 1 | 5449 / 1 | 936 / 1 | 267 / 1 | 283 / 1 | 840 / 1 |
| vault | What will my rent be after the increase? | 7254 / 1 | 3153 / 1 | 707 / 1 | 143 / 1 | 155 / 1 | 237 / 1 |
| vault | What oven temperatures do I bake sourdough at? | 795 / 1 | 3664 / 1 | 863 / 1 | 260 / 1 | 406 / 1 | 524 / 1 |
| vault | Which PhD programs did I decide to apply to? | 3739 / 1 | 3989 / 1 | 841 / 1 | 100 / 1 | 111 / 1 | 111 / 1 |
| vault | What is my goal time for the half marathon? | 7748 / 1 | 3795 / 1 | 887 / 1 | 265 / 1 | 517 / 1 | 585 / 1 |
| vault | How much am I contributing to my Roth IRA each month? | 7807 / 1 | 5261 / 1 | 720 / 1 | 209 / 1 | 209 / 1 | 312 / 1 |
| vault | When should I transplant the tomatoes outside? | 3613 / 1 | 3158 / 1 | 856 / 1 | 78 / 1 | 301 / 1 | 476 / 1 |
| vault | What is my usual coffee order? | 7820 / 1 | 4568 / 1 | 720 / 1 | 50 / 1 | 50 / 1 | 208 / 1 |
| vault | How many new Anki cards per day did I settle on? | 7474 / 1 | 6308 / 1 | 736 / 1 | 131 / 1 | 391 / 1 | 891 / 1 |

## Absent topics (tokens)

| topic | keyword top-10 docs | embedding top-10 docs | embedding top-8 chunks | atlas pack @300 | atlas pack @800 | atlas pack @1500 | atlas abstained |
|---|---|---|---|---|---|---|---|
| yacht maintenance | 1717 | 7048 | 858 | 28 | 28 | 28 | yes |
| wedding planning | 2896 | 5711 | 724 | 28 | 28 | 28 | yes |
| pet vet appointment | 1935 | 2339 | 753 | 28 | 28 | 28 | yes |
| jury duty summons | 0 | 3827 | 876 | 28 | 28 | 28 | yes |
| mortgage refinancing | 0 | 6074 | 709 | 28 | 28 | 28 | yes |
| knitting patterns | 1441 | 6074 | 789 | 28 | 28 | 28 | yes |
| car insurance claim | 883 | 3133 | 758 | 28 | 28 | 28 | yes |
| baby shower gifts | 913 | 4543 | 790 | 28 | 28 | 28 | yes |
| golf tee times | 3600 | 2084 | 627 | 73 | 73 | 190 | NO |
| skydiving lessons | 0 | 7300 | 1029 | 28 | 28 | 28 | yes |
| When is my car due for an oil change? | 2814 | 2330 | 578 | 28 | 28 | 28 | yes |
| What did the vet say about my cat? | 1572 | 5119 | 737 | 28 | 28 | 28 | yes |
| What is my wifi password at the cabin? | 1090 | 472 | 617 | 28 | 28 | 28 | yes |
| Who is catering my sister's wedding? | 7145 | 496 | 540 | 28 | 28 | 28 | yes |

<!-- private -->

## Real vault (private, tokens only)

The user's own Obsidian vault: 419 notes, 2382 heading sections, embedded locally with `base`. 12 questions written from note titles only; questions, titles and content are not in this repo and nothing was sent to Grok (no expansion, no answers, no judge). Quality proxy: the target note appears in the context.

| Strategy | mean context tokens | median | target note in context | hits per 1k tokens |
|---|---|---|---|---|
| keyword top-10 docs | 10907 | 8578 | 92% | 0.08 |
| embedding top-10 docs | 7400 | 6610 | 100% | 0.14 |
| embedding top-8 chunks | 1185 | 1216 | 100% | 0.84 |
| atlas pack @300 | 268 | 272 | 92% | 3.42 |
| atlas pack @800 | 625 | 661 | 100% | 1.60 |
| atlas pack @1500 | 1102 | 1094 | 100% | 0.91 |

Absent topics (4): Atlas abstained on 100%. Mean tokens: keyword top-10 docs 13695, embedding top-10 docs 6053, embedding top-8 chunks 966, atlas pack @300 26, atlas pack @800 26, atlas pack @1500 26.
