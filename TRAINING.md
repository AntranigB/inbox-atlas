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
- After training a scalar temperature T is fit on the val split so `sigmoid(m / T)` is a calibrated
  P(member). Saved to `models/atlas-embed/region.pt` with its config.

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
| Mine anchors + negatives | `uv run python -m train.mine --ds enron` | 3 min (embeds 185k train docs with base bge) |
| Teacher scores | `uv run python -m train.teacher --ds enron` | see log, about 30 to 50 min |
| Stage A (student LoRA) | `uv run python -m train.stage_a --run a2 --out models/atlas-embed` | about 0.86 s/step, 2.8k steps per epoch = 40 min |
| Stage B (region) | `uv run python -m train.stage_b --model models/atlas-embed --run b1` | 5 min embed cache + 5 to 10 min train |
| Stage C (personal, optional) | `uv run python -m train.stage_a --run c1 --ds personal --init-adapter models/atlas-embed --out models/atlas-embed-personal --epochs 1` | depends on inbox size |
| Eval | `uv run python -m train.eval --run eval1 --models models/atlas-embed` | 5 to 10 min |

The whole Stage A chain is scripted: `train/run_stage_a.sh <run>` (parses, mines, then trains with
distillation if `teacher.json` exists, otherwise as the no-distill ablation). The full pipeline
after Grok labels: `train/run_full.sh`.

Ablations (same data, one flag each):
`--no-distill`, `--no-negs`, `--no-matryoshka` on `train.stage_a`; `--no-neg-phrases`,
`--no-distill` on `train.stage_b`.

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

Filled in from `runs/*/metrics.json` as runs finish.
