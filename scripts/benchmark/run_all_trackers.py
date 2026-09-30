"""Run the whole fixed-detector benchmark: detect once, replay every arm.

Two stages, both resumable with --skip-existing:

  1. detect  -- one detector pass per video into a shared detection cache
  2. replay  -- every tracker arm over that cache, from a single video decode

Stage 1 is the expensive one and only has to happen once, ever. Re-running the
benchmark with different tracker settings reuses it, which is the point: no
tracker comparison should ever be contaminated by a second detector pass.

Example
-------
    python scripts/benchmark/run_all_trackers.py \
        --videos data/raw_videos \
        --model weights/mosmon_yolov11x_overlap_tiling_best_v1.pt \
        --out outputs/tracker_benchmark \
        --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="Folder of source videos.")
    ap.add_argument("--model", required=True, help="YOLO .pt weights.")
    ap.add_argument("--out", required=True, help="Benchmark root directory.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "tracker_benchmark.yaml"))
    ap.add_argument("--arms", default=None, help="Comma-separated arm subset.")
    ap.add_argument("--stage", choices=["all", "detect", "replay"], default="all")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-native", action="store_true",
                    help="Skip the tracker-native (Experiment A) tables.")
    args = ap.parse_args(argv)

    from mosmon_tracking.benchmark.batch import benchmark_batch
    from mosmon_tracking.benchmark.detect import detect_batch
    from mosmon_tracking.config import load_config

    cfg = load_config(args.config)
    if not cfg.model.paths:
        cfg.model.paths = [args.model]
    out = Path(args.out)
    detections = out / "detections_canonical"

    if args.stage in ("all", "detect"):
        print(f"== stage 1/2: detection cache -> {detections}")
        idx = detect_batch(args.videos, args.model, cfg, detections,
                           skip_existing=args.skip_existing)
        failed = int((idx["status"] == "failed").sum()) if not idx.empty else 0
        print(f"   {len(idx)} videos, {failed} failed")
        if failed:
            print("   detection failures recorded as *.failure.json; not replaying them")

    if args.stage in ("all", "replay"):
        print(f"== stage 2/2: replaying arms -> {out}")
        arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
        idx = benchmark_batch(detections, cfg, out, videos_dir=args.videos,
                              arm_names=arms, skip_existing=args.skip_existing,
                              write_native=not args.no_native)
        if idx.empty:
            print("   nothing to replay")
            return 1
        failed = int((idx["status"] == "failed").sum())
        print(f"   {len(idx)} videos, {failed} failed")
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
