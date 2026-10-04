# Inbox Atlas retrieval eval

Encoder `base`, 24 emails, 12 handwritten queries, pool = top 20 of keyword + embedding + region, Grok judge (`grok-4.20-0309-non-reasoning`). Run 2026-10-03 22:57.

| Method | Recall@10 | nDCG@10 |
|---|---|---|
| keyword (BM25) | 0.175 | 0.217 |
| embedding (cosine) | 0.811 | 0.828 |
| region, query only | 0.839 | 0.846 |
| region, no negatives | 0.887 | 0.885 |
| region, no hub z (raw) | 0.874 | 0.880 |
| region (full) | 0.867 | 0.876 |
| hybrid (region + BM25 RRF) | 0.886 | 0.820 |

## Per query nDCG@10

| Query | relevant | keyword (BM25) | embedding (cosine) | region, query only | region, no negatives | region, no hub z (raw) | region (full) | hybrid (region + BM25 RRF) |
|---|---|---|---|---|---|---|---|---|
| coding competition | 10 | 0.22 | 0.93 | 0.93 | 0.94 | 0.93 | 0.94 | 0.87 |
| money I owe someone | 3 | 0.47 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 |
| anything about housing next year | 2 | 0.61 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| travel plans | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| upcoming exams | 4 | 0.00 | 0.54 | 0.53 | 0.63 | 0.65 | 0.53 | 0.33 |
| internship hunting | 3 | 0.00 | 0.70 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 |
| stuff I bought online | 3 | 0.00 | 0.47 | 0.61 | 0.65 | 0.63 | 0.65 | 0.76 |
| workout schedule | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| catching up with my parents | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| my research lab work | 6 | 0.30 | 0.41 | 0.39 | 0.63 | 0.58 | 0.63 | 0.70 |
| tech news digest | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| winter break trip | 2 | 0.00 | 0.88 | 0.92 | 1.00 | 1.00 | 1.00 | 0.50 |

## Related? yes/no accuracy: 19/20 = 95%

Rule: top hub corrected z >= 3.0 and raw >= 0.5.

| Topic | truth | answer | max z | count | |
|---|---|---|---|---|---|
| programming contests | present | YES | 10.31 | 9 | ok |
| paying back a friend | present | YES | 8.14 | 2 | ok |
| where I will live next year | present | YES | 4.59 | 2 | ok |
| flights | present | YES | 4.84 | 3 | ok |
| midterm exam | present | YES | 4.06 | 1 | ok |
| software engineering internship | present | YES | 4.44 | 2 | ok |
| package delivery | present | YES | 5.82 | 1 | ok |
| spin class | present | YES | 4.34 | 1 | ok |
| lab meeting | present | NO | 2.99 | 0 | MISS |
| weekly newsletter | present | YES | 4.7 | 2 | ok |
| yacht maintenance | absent | NO | -1.21 | 0 | ok |
| wedding planning | absent | NO | 2.47 | 0 | ok |
| pet vet appointment | absent | NO | 0.02 | 0 | ok |
| jury duty summons | absent | NO | 0.01 | 0 | ok |
| mortgage refinancing | absent | NO | 2.07 | 0 | ok |
| knitting patterns | absent | NO | -1.35 | 0 | ok |
| car insurance claim | absent | NO | 0.87 | 0 | ok |
| baby shower gifts | absent | NO | -1.22 | 0 | ok |
| golf tee times | absent | NO | 2.98 | 0 | ok |
| skydiving lessons | absent | NO | 0.83 | 0 | ok |
