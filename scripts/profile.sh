#!/bin/sh
# Switch the live app between the demo data (judging) and your personal data.
#   sh scripts/profile.sh demo       fictional inbox + synthetic vault, Tiger Cloud
#   sh scripts/profile.sh personal   your real Gmail, local SQLite only (nothing uploaded)
set -e
cd "$(dirname "$0")/.."
uv run atlas down >/dev/null 2>&1 || true
ATLAS_DATA=data/personal uv run atlas down >/dev/null 2>&1 || true
case "$1" in
  demo)
    nohup uv run atlas up > data/atlas-up.log 2>&1 &
    ;;
  personal)
    ATLAS_DATA=data/personal ATLAS_DB=sqlite nohup uv run atlas up > data/atlas-up.log 2>&1 &
    ;;
  *) echo "usage: sh scripts/profile.sh demo|personal"; exit 1 ;;
esac
echo "atlas restarting with the $1 profile; check with: uv run atlas status"
