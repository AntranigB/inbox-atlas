# Training: atlas-embed + atlas-region

Runbook for training the Inbox Atlas models on the GPU PC (`ssh archy`, RTX 4070 SUPER 12 GB).
Results live at the bottom of this file and in `runs/<run>/metrics.json`.

## What we train

**atlas-embed** (the student encoder)
- FROZEN `BAAI/bge-base-en-v1.5` backbone, CLS pooling, L2 norm, bge query prefix on the query side.
- Trainable LoRA (peft, r=16, alpha=32, dropout 0.05) on the attention `query`, `key`, `value`
  and attention `output.dense` projections: 1.18M trainable parameters out of 110M.
- Loss per Matryoshka dim in {768, 256, 64} (truncate, renormalize):
  InfoNCE over in-batch docs + mined hard negatives (scale 20, false negatives masked when they
  share the label or the thread), plus listwise KL distillation from a FROZEN cross-encoder teacher
  (`BAAI/bge-reranker-v2-m3`, temperature 1) over each anchor's own candidates. Teacher scores
  also act as a false-negative filter: a mined candidate the teacher rates at least as relevant as
  the positive is never used as a negative.
- Prior art, stated honestly: listwise distillation from a cross-encoder into a bi-encoder is
  RocketQAv2 (Ren et al. 2021) and margin-MSE distillation (Hofstaetter et al. 2020); the teacher
  denoising of hard negatives is RocketQA (Qu et al. 2021); Matryoshka is Kusupati et al. 2022;
  LLM-generated training queries are GPL / InPars / Promptagator. What is ours is the combination
  with folder and Grok topic supervision and the region model below.

**atlas-region** (the novel piece, `atlas/model/region_encoder.py`)
- A Set Transformer (Lee et al. 2019): input projection + learned type embedding (positive or
  negative phrase), 2 ISAB layers with 16 inducing points, PMA with K=4 seeds.
- It reads the SET of topic-phrase embeddings Grok writes at query time (from the frozen student)
  and outputs a region: K unit anchors `a_k`, per-anchor temperatures `tau_k` (softplus), a bias `b`.
  Membership logit `m(e) = logsumexp_k(tau_k * cos(e, a_k)) - b`.
- Losses: BCE membership against topic-set ground truth with hard negatives (nearest non-members),
  a negative-set term (emails near the negative phrases must score below the boundary), and
  listwise KL to the frozen teacher's scores for the topic name.
- Set augmentation: 1 to 8 positives, 0 to 3 negatives, random subsets (phrase dropout) and small
  embedding noise, so it does not care how many phrases Grok writes.
- After training, a scalar temperature T and a scalar offset are fit on the val split (every val
  email scored against sampled val topic sets, so the natural ~0.1% base rate) so
  `sigmoid((m - shift) / T)` is a calibrated P(member). A temperature alone was not enough: training
  sets are sampled near balanced, and the first smoke run had ECE 0.42 with T only. Saved to
  `models/atlas-embed/region.pt` with its config.

## Data

| Source | What | Labels |
|---|---|---|
| Enron (CMU, 517k files, 233k after dedupe) | `train/enron.py` | user-created folder names (generic ones dropped, >= 20 emails), humanized: "california_energy" -> "california energy" |
| Grok labels | `train/grok_label.py` (20k Enron emails, cap is a flag) | 2-3 abstract topic phrases + 2 queries (vague, specific) per email |
| Personal Gmail | `train/build_personal.py` from `data/mail.sqlite` | Gmail labels + Grok labels |

Free pairs: subject -> body (doc rendered without its subject, batches kept homogeneous so the
missing subject line is not a shortcut), reply -> parent. Topic sets for the region model: folder
labels (members = the folder) and clusters of Grok topic phrases (k-means micro clusters, then
average-link agglomerative over them).

Splits (never leak): ~15% of folder labels are held out entirely (`test_topic`, every thread that
touches one goes there), then 10% of remaining threads (`test_thread`), 5% of threads (`val`, model
selection and calibration), rest `train`. Only `train` emails are ever used as positives,
negatives or teacher candidates.

Privacy: Grok labeling sends the subject and the first 800 chars of each labeled email to xAI.
Raw data is never modified or deleted.

## Setup on the PC (once)

```bash
ssh archy
mkdir -p ~/Documents && cd ~/Documents && gh repo clone xerneas3318/inbox-atlas
cd inbox-atlas && git checkout training
# only the xAI key (run this on the Mac, it never prints the key)
grep '^XAI_API_KEY=' /Users/xerneas/Documents/bigred/nomad/.env | ssh archy 'cat >> ~/Documents/inbox-atlas/.env'
uv venv --python 3.12
uv sync --extra train --extra dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.14.1+cu130 True
```

Always `git pull` before launching anything. Everything runs inside tmux so it survives ssh drops.

## Stages

| Step | Command | Time on 4070 SUPER |
|---|---|---|
| Parse Enron | `uv run python -m train.enron --download` | 2 min (download 450 MB + 28-way parse) |
| Grok labels | `tmux new -d -s atlas-grok 'uv run python -m train.grok_label --cap 20000 2>&1 \| tee -a runs/grok_enron.log'` | 18 min, network bound, runs alongside GPU work |
| Mine anchors + negatives | `uv run python -m train.mine --ds enron` | 6 min (embeds 175k train docs with base bge, mines 132k unique queries) |
| Teacher scores | `uv run python -m train.teacher --ds enron` | 36 min for 491k pairs (bge-reranker-v2-m3, bf16, 140 to 220 pairs/s) |
| Stage A (student LoRA) | `uv run python -m train.stage_a --run a2 --out models/atlas-embed` | 0.84 s/step at bs 32 x (1 pos + 3 negs), 4.2 GB with gradient checkpointing (without it, bs 32 OOMs on 12 GB); 6.1k steps per epoch with Grok anchors = 86 min |
| Stage B (region) | `uv run python -m train.stage_b --model models/atlas-embed --run b1` | 5 min embed cache + set prep + 4k steps, about 15 min |
| Stage C (personal, optional) | `uv run python -m train.build_personal && uv run python -m train.mine --ds personal && uv run python -m train.stage_a --run c1 --ds personal --no-distill --init-adapter models/atlas-embed --out models/atlas-embed-personal --epochs 1` | minutes for a few thousand emails |
| Transfer eval | `uv run python -m train.eval --ds personal --run eval-personal --models models/atlas-embed` | 1 to 3 min |
| Eval | `uv run python -m train.eval --run eval1 --models models/atlas-embed` | 7 min per encoder (BM25 + base + each checkpoint) |

The whole Stage A chain is scripted: `train/run_stage_a.sh <run>` (parses, mines, then trains with
distillation if `teacher.json` exists, otherwise as the no-distill ablation). The full pipeline
after Grok labels: `train/run_full.sh`.

Ablations (same data, one flag each):
`--no-distill`, `--no-negs`, `--no-matryoshka` on `train.stage_a`; `--no-neg-phrases`,
`--no-distill` on `train.stage_b`. `train/run_ablations.sh` runs them all: the region ablations
(saved as `models/atlas-embed/region_<name>.pt`, picked up by eval automatically), then the
student ablations at half an epoch each next to a half-epoch full-recipe reference (`abl-full`),
then `runs/eval-abl/metrics.md`. Note `--no-distill` on stage A removes the teacher entirely,
including its false-negative filter.

## Monitoring

```bash
ssh archy
tmux ls                                 # atlas-train, atlas-grok
tmux attach -t atlas-train              # Ctrl-b d to detach
tail -f ~/Documents/inbox-atlas/runs/<run>/train.log
grep '"step"' ~/Documents/inbox-atlas/runs/<run>/train.log | tail   # loss, ema, kd, s/step, eta
grep 'val step' ~/Documents/inbox-atlas/runs/<run>/train.log         # val nDCG@10, best is saved
nvidia-smi                              # expect ~100% util, ~4 GB for stage A
```

## Personal Gmail data

The ingest branch writes `data/mail.sqlite` on the Mac. Sync it to the PC (run on the Mac):

```bash
rsync -av /Users/xerneas/Documents/bigred/inbox-atlas/data/mail.sqlite archy:~/Documents/inbox-atlas/data/
```

Then on the PC: `uv run python -m train.build_personal` writes `data/datasets/personal/`
(Gmail labels as free topics, split by thread). `uv run python -m train.grok_label --ds personal --cap 5000`
adds Grok labels. The personal set doubles as the zero-shot transfer eval: models trained only
on 2001 corporate Enron mail, evaluated on a 2026 student's Gmail.

## Copy the trained model back to the Mac

```bash
rsync -av archy:~/Documents/inbox-atlas/models/atlas-embed /Users/xerneas/Documents/bigred/inbox-atlas/models/
```

Then `ATLAS_ENCODER=models/atlas-embed uv run python server.py`. The loader uses `merged/` (plain
transformers, no peft needed) and falls back to `adapter/` + peft. `ATLAS_DIM=256` or `64` gives
the Matryoshka-truncated encoder (the region model needs the full 768).

## Results

Dataset after cleaning: 224,867 deduped Enron emails, 426 folder topics (63 held out), splits
train 175,353 / val 10,817 / test_thread 20,779 / test_topic 17,918. 19,980 emails Grok-labeled.
Eval corpus for held-out tasks: 38,697 unseen emails. Recall@10 is hits in the top 10 divided by
min(relevant, 10).

### Baselines (`runs/eval-base`)

| task (n queries) | BM25 R@10 / nDCG@10 | base bge R@10 / nDCG@10 |
|---|---|---|
| topic, seen folders, train emails (254) | 0.135 / 0.121 | 0.131 / 0.120 |
| topic, seen folders, held-out threads (308) | 0.090 / 0.085 | 0.091 / 0.087 |
| topic, held-out folders (63) | 0.263 / 0.267 | 0.290 / 0.310 |
| subject -> body, held-out threads (3000) | 0.490 / 0.388 | 0.438 / 0.342 |
| Grok specific query -> email (4280) | 0.868 / 0.753 | 0.733 / 0.593 |
| Grok vague query -> email (4280) | 0.810 / 0.652 | 0.788 / 0.635 |

Heuristic region (PLAN.md 3b) on base bge, held-out folders: AUROC 0.819, nDCG@10 0.261.

Honest notes: folder names are a hard, noisy query set (many are cryptic codes like "esvl" or
person names), so absolute numbers are low for every system. BM25 beats base bge on the Grok
"specific" queries because Grok wrote them while reading the email, so they share its words;
the vague queries are the fairer semantic test. Pooled ECE at the natural ~0.1% base rate is
near zero for anything that predicts "no" everywhere, so we also report ECE over the top 200
emails per topic, where membership decisions actually happen.

### First run (a1, superseded)

Stage A without teacher on the first parse (before the folder-label cleanup), val nDCG@10 over
1500 steps: subject -> body 0.444 -> 0.519, folder topics 0.140 -> 0.156. Stopped at step 2100
to free the GPU for the teacher; it validated the pipeline end to end (adapter, merge, loader).

### Main run: atlas-embed (a2), `runs/eval2`

a2: 1 epoch, 6,148 steps, 89 min, InfoNCE + Matryoshka + teacher KD, anchors = folder topics
13.9k, subject 60k, reply 30k, Grok topic 32.6k, Grok query 28.5k; teacher scores on 30.3k
queries. Val score (mean nDCG@10 of folder topics and subject) 0.290 -> 0.339; best checkpoint
saved at the last eval.

| task (nDCG@10) | BM25 | base bge | atlas-embed 768 | 256 | 64 |
|---|---|---|---|---|---|
| topic, seen folders, train emails | 0.121 | 0.120 | **0.135** | 0.122 | 0.072 |
| topic, seen folders, held-out threads | 0.085 | **0.087** | 0.086 | 0.075 | 0.054 |
| topic, held-out folders | 0.267 | **0.310** | 0.258 | 0.242 | 0.141 |
| subject -> body, held-out threads | 0.388 | 0.342 | **0.408** | 0.381 | 0.271 |
| Grok specific query -> email | **0.753** | 0.593 | 0.706 | 0.649 | 0.422 |
| Grok vague query -> email | 0.652 | 0.635 | **0.692** | 0.659 | 0.467 |

(Recall@10 moves the same way; full tables in `runs/eval2/metrics.md`.)

What this says, honestly:
- On unseen emails, atlas-embed beats base bge on subject -> body (+0.066 nDCG@10) and on Grok
  queries (+0.113 specific, +0.057 vague) and is the best system on vague queries, beating BM25.
  On specific queries BM25 still wins (they share the email's words by construction).
- On folder-name queries it does NOT generalize: no gain on seen folders over held-out threads,
  and it is worse than base bge on held-out folders (0.258 vs 0.310). Folder names are short,
  noisy codes; training on them seems to fit the training folders rather than teach "topic name
  -> email" in general. The Grok topic phrases are the better topic supervision.
- Matryoshka: 256 dims keeps most of the gain (still beats base 768 on subject and both Grok
  query tasks), 64 dims loses a lot. Base bge was not trained for truncation, so its 256/64
  numbers would be lower still (not measured yet).

Region membership on held-out folders (positives = folder name + Grok phrases of 3 seed emails,
seeds excluded from scoring; negatives = 2 nearest train folder names):

| encoder / region | AUROC | nDCG@10 | ECE all | ECE top 200 | Brier |
|---|---|---|---|---|---|
| base bge / heuristic | 0.819 | **0.261** | 0.0012 | 0.065 | 0.00279 |
| atlas-embed / heuristic | **0.843** | 0.249 | 0.0014 | 0.055 | **0.00275** |
| atlas-embed / learned v1 (4 anchors only) | 0.781 | 0.122 | 0.0025 | **0.044** | 0.00280 |

- The trained encoder makes the heuristic region better at separating members (AUROC +0.024).
- The first learned region (anchors only) is better calibrated at the top of the ranking but
  ranks worse than the heuristic. Diagnosis: with only 4 anchors it cannot express "close to ANY
  one of these phrases", which the heuristic's max-facet term does. v2 adds the input phrases as
  extra anchors (shared learned temperature) plus a learned gate for the negative phrases, so the
  heuristic is a special case of the model. Results below when `runs/eval3` lands.

### v2 region and region ablations, `runs/eval3`

v2 (phrases as extra anchors + negative gate) adds model selection on region-val topics: 10% of
the training topics are never used for region training and are scored on val emails every 250
steps; step 0 (the initialization) is a candidate.

| encoder / region | AUROC | nDCG@10 | ECE top 200 | Brier |
|---|---|---|---|---|
| base bge / heuristic | 0.819 | **0.261** | 0.065 | 0.00279 |
| atlas-embed / heuristic | **0.843** | 0.249 | 0.055 | 0.00275 |
| atlas-embed / learned v2 (`region.pt`) | 0.841 | 0.242 | **0.046** | 0.00275 |
| v2, lr 3e-5 / no teacher KD / no negative phrases / PU negatives | 0.841 | 0.242 | 0.045 to 0.046 | 0.00275 |
| atlas-embed / learned v1 (anchors only, trained 4k steps) | 0.781 | 0.122 | 0.044 | 0.00280 |

The negative result, stated plainly: on held-out topics, every region-training run got worse than
its own initialization (region-val AUROC 0.918 at step 0, 0.86 to 0.89 after 250+ steps, with
lr 3e-4 or 3e-5, with or without teacher KD, negative phrases, or positive-unlabeled negatives),
so selection always kept step 0. With only ~680 training topic sets and partial labels (most
on-topic emails carry no folder or Grok label, so they are trained as negatives), the Set
Transformer fits the training topics instead of learning a transferable rule. What `region.pt`
adds over the heuristic today is calibration: a fitted temperature and offset give a usable
P(member) (ECE over the top 200 per topic 0.046 vs 0.055 for a Platt-scaled heuristic) at the same
ranking quality (AUROC 0.841 vs 0.843). The ablation rows are identical because they all
selected the same initialization. More topics (Grok-label all of Enron, personal Gmail labels)
and a ranking loss are the next things to try.

### Student ablations

Pending: `runs/eval-abl/metrics.md` (abl-full, no distill, no negatives, no Matryoshka; 0.35
epoch each, about 30 min each, started 04:35).

### Efficiency notes

Stage A, the teacher and embedding passes keep the GPU at 90 to 100%. Stage B (the region model
is 1.5M params on cached embeddings) and eval are CPU bound Python (set sampling, BM25, label
loops), so the GPU sits mostly idle during those phases; the batch collate was vectorized after
this showed up. Running several stage B jobs at once (as during the region sweep) multiplies the
CPU load.
