#!/usr/bin/env bash
# Ablations after the main model exists (models/atlas-embed with region.pt).
# Region ablations are cheap (cached embeddings). Student ablations use half an epoch each, with
# a matching half-epoch full-recipe run (abl-full) as the reference, so they fit the night.
set -euo pipefail
stage() { local name=$1; shift; mkdir -p runs/$name; echo "== $name: $*" | tee -a runs/$name/train.log; "$@" 2>&1 | tee -a runs/$name/train.log; }
M=models/atlas-embed
[ -f runs/b-nonegphrase/done.json ] || stage b-nonegphrase uv run python -m train.stage_b --model $M --run b-nonegphrase --no-neg-phrases --out $M/region_nonegphrase.pt
[ -f runs/b-nodistill/done.json ] || stage b-nodistill uv run python -m train.stage_b --model $M --run b-nodistill --no-distill --out $M/region_nodistill.pt
stage eval3 uv run python -m train.eval --run eval3 --models $M
for v in full nodistill nonegs nomat; do
  flag=""
  [ $v = nodistill ] && flag="--no-distill"
  [ $v = nonegs ] && flag="--no-negs"
  [ $v = nomat ] && flag="--no-matryoshka"
  [ -f runs/abl-$v/done.json ] || stage abl-$v uv run python -m train.stage_a --run abl-$v --epochs 0.5 $flag --out models/abl-$v
done
stage eval-abl uv run python -m train.eval --run eval-abl --no-base --models models/abl-full,models/abl-nodistill,models/abl-nonegs,models/abl-nomat
