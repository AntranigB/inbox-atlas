#!/usr/bin/env bash
# Full Enron pipeline with Grok labels, teacher distillation, region training, eval and ablations.
# Run from the repo root:  tmux new -d -s atlas-train 'bash train/run_full.sh'
# Each stage logs to runs/<stage>/train.log. Stages whose output exists are skipped, so the
# script can be re-launched after a crash.
set -euo pipefail
stage() { local name=$1; shift; mkdir -p runs/$name; echo "== $name: $*" | tee -a runs/$name/train.log; "$@" 2>&1 | tee -a runs/$name/train.log; }
D=data/datasets/enron

[ -f $D/.labels_v2 ] || { stage data uv run python -m train.enron; touch $D/.labels_v2; rm -f $D/mined.json $D/teacher.json; }
# wait for the Grok labeler (runs in its own tmux session) so its anchors are mined in
while tmux has-session -t atlas-grok 2>/dev/null; do sleep 20; done
[ -f $D/mined.json ] || stage mine uv run python -m train.mine --ds enron
[ -f $D/teacher.json ] || stage teacher uv run python -m train.teacher --ds enron
[ -f runs/a2/done.json ] || stage a2 uv run python -m train.stage_a --run a2 --out models/atlas-embed
[ -f runs/b2/done.json ] || stage b2 uv run python -m train.stage_b --model models/atlas-embed --run b2
stage eval2 uv run python -m train.eval --run eval2 --models models/atlas-embed
# ablations (same data and steps, one change each)
[ -f runs/abl-nodistill/done.json ] || stage abl-nodistill uv run python -m train.stage_a --run abl-nodistill --no-distill --out models/abl-nodistill
[ -f runs/abl-nonegs/done.json ] || stage abl-nonegs uv run python -m train.stage_a --run abl-nonegs --no-negs --out models/abl-nonegs
[ -f runs/abl-nomat/done.json ] || stage abl-nomat uv run python -m train.stage_a --run abl-nomat --no-matryoshka --out models/abl-nomat
stage eval-abl uv run python -m train.eval --run eval-abl --no-base --models models/atlas-embed,models/abl-nodistill,models/abl-nonegs,models/abl-nomat
