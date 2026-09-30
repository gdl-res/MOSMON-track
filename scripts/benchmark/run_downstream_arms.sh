#!/usr/bin/env bash
# Re-run the existing species analysis once per tracker arm, sequentially.
#
# One arm at a time, and one output directory per arm: extract_track_features
# caches to <out>/track_features.parquet keyed on the run list and options, and
# the key does NOT include the tracker -- a shared directory would silently
# serve one arm's features to another.
#
# bytetrack_iou is deliberately absent: it is bit-identical to bytetrack on the frozen
# detection cache and is not a distinct tracker.
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python}"
O="${OUT:?set OUT to the benchmark output root}"
ARMS="${ARMS:-botsort bytetrack ocsort}"
cd "$REPO" || exit 1
for arm in $ARMS; do
  if [ -f "$O/downstream/$arm/classification_results.json" ]; then
    echo "=== $(date '+%F %T')  $arm already done -- skipping"; continue
  fi
  echo "=== $(date '+%F %T')  species-analysis $arm  START"
  systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=2G -- \
    "$PY" -u -m mosmon_tracking.cli species-analysis \
      --batch "$O/$arm" --out "$O/downstream/$arm" \
      --config configs/tracker_benchmark.yaml
  echo "=== $(date '+%F %T')  species-analysis $arm  rc=$?"
done
echo "=== $(date '+%F %T')  ALL DOWNSTREAM ARMS DONE"
