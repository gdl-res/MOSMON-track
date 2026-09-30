#!/usr/bin/env bash
# Replay the frozen detection cache through one tracker arm at a time.
#
# Why not just `run_all_trackers.py --stage replay`: that builds all five arms
# into a single shared-decode loop and holds every arm's state plus the whole
# detection table in memory at once. On 31 GB that exhausted RAM and swap on the
# SECOND video (2026-08-27) and thrashed to a standstill. One arm at a time fits;
# the cost is decoding each video once per arm instead of once for all five.
#
# Resume semantics -- the subtle part. `benchmark_batch` keys its skip-manifest
# by VIDEO only (`_manifests/<video_id>.json`), not by (video, arm). So a second
# arm run with --skip-existing against the same root would skip every video the
# first arm finished and silently emit a benchmark with one complete arm. This
# script therefore rotates `_manifests` to `_manifests_<arm>` after each arm
# completes, which:
#   * lets --skip-existing resume correctly WITHIN an arm, and
#   * gives the next arm a clean slate, and
#   * marks a finished arm so re-running this script resumes at the right arm.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python}"
VIDEOS="${VIDEOS:?set VIDEOS to the folder of source videos}"
OUT="${OUT:?set OUT to the benchmark output root}"
MODEL="weights/mosmon_yolov11x_overlap_tiling_best_v1.pt"
ARMS="${ARMS:-bytetrack_iou bytetrack ocsort botsort deepocsort}"

# DeepOCSORT's ReID encoder runs through onnxruntime-gpu, whose CUDA provider
# dlopen()s libcublas/libcublasLt/libcurand/libcudart. Those ship inside the pip
# nvidia-* wheels and are NOT on the default loader path, so the provider silently
# fails to load and ORT falls back to CPUExecutionProvider -- measured 6.1x slower
# on realistic 256x128 crops at batch 256 (464 ms vs 76 ms). torch works regardless
# because it preloads its own copies, which is why this hides until ReID runs.
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO" || exit 1

for arm in $ARMS; do
  if [ -d "$OUT/_manifests_$arm" ]; then
    echo "=== $arm already complete ($(ls "$OUT/_manifests_$arm" | wc -l) videos) -- skipping"
    continue
  fi
  echo "=== $(date '+%F %T')  arm $arm  START"
  # Run under a memory-capped systemd scope. On 2026-08-27 the replay ballooned
  # to 26 GB on a dense Culex video and the OOM killer took the job three times
  # -- but not before the thrashing froze the desktop and killed the terminal.
  # The cause is fixed (see detections_io.read_detections; peak is now ~3.6 GB),
  # this is the backstop: if anything runs away again the kernel kills the job
  # ALONE and the machine stays usable. --scope keeps the caller's environment,
  # so the LD_LIBRARY_PATH above still reaches DeepOCSORT's ONNX provider, and
  # it propagates the child's exit status, so the `rc` resume logic below works.
  #
  # 20 G against a ~6 G expected peak is deliberate headroom: deepocsort holds
  # ReID crops and has not yet run over the dense videos at all.
  systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=2G -- \
    "$PY" -u scripts/benchmark/run_all_trackers.py \
      --videos "$VIDEOS" --model "$MODEL" --out "$OUT" \
      --stage replay --arms "$arm" --skip-existing
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $(date '+%F %T')  arm $arm FAILED rc=$rc -- stopping; rerun this script to resume"
    # 137 = SIGKILL, which here almost always means the scope hit MemoryMax.
    [ $rc -eq 137 ] && echo "=== rc=137 (SIGKILL): hit the 20G scope limit; check 'journalctl -k -g oom'"
    exit $rc
  fi
  mv "$OUT/_manifests" "$OUT/_manifests_$arm" 2>/dev/null
  [ -f "$OUT/benchmark_index.csv" ] && mv "$OUT/benchmark_index.csv" "$OUT/benchmark_index_$arm.csv"
  echo "=== $(date '+%F %T')  arm $arm  DONE"
done
echo "=== $(date '+%F %T')  ALL ARMS COMPLETE"
