#!/usr/bin/env bash
# One command to rebuild every model, result table and figure, then check them.
#   bash reproduce.sh            # uses the frozen data snapshot in data/snapshot (exact reproduction)
#   bash reproduce.sh nflverse   # re-downloads the full 2021-2025 nflverse files (~95 MB) first
set -euo pipefail
SOURCE="${1:-snapshot}"
python fourth_down.py build --source "$SOURCE" | tee results/build_log.txt
python verify_results.py
echo
echo "Example recommendation:"
python fourth_down.py predict --ydstogo 4 --yardline_100 35 --score_diff -7 --qtr 4 --seconds_left 900
