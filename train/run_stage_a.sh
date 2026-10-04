#!/usr/bin/env bash
# Stage A pipeline on Enron. Run from the repo root inside tmux:
#   tmux new -d -s atlas-train 'bash train/run_stage_a.sh a1 2>&1 | tee -a runs/a1/train.log'
set -euo pipefail
RUN=${1:-a1}
mkdir -p runs/$RUN
[ -f data/datasets/enron/emails.jsonl ] || uv run python -m train.enron --download
[ -f data/datasets/enron/mined.json ] || uv run python -m train.mine --ds enron
if [ -f data/datasets/enron/teacher.json ]; then
  uv run python -m train.stage_a --run $RUN --ds enron --out models/atlas-embed
else
  # no teacher scores yet: this run doubles as the no-distill ablation
  uv run python -m train.stage_a --run $RUN --ds enron --no-distill --out models/$RUN
fi
