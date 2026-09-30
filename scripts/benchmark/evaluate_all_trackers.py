"""Roll the benchmark up: tables, downstream comparison, and the report.

Runs after scripts/benchmark/run_all_trackers.py. The downstream stage re-runs
the existing species analysis once per arm, into a separate output directory per
arm -- the track-feature cache key does not include the tracker, so a shared
directory would silently serve one arm's features to another.

Example
-------
    python scripts/benchmark/evaluate_all_trackers.py \
        --benchmark outputs/tracker_benchmark \
        --out       outputs/tracker_benchmark \
        --downstream
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="Benchmark root (from run_all_trackers).")
    ap.add_argument("--out", default=None, help="Output root (default: --benchmark).")
    ap.add_argument("--detections", default=None,
                    help="Detection cache (default: <benchmark>/detections_canonical).")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "tracker_benchmark.yaml"))
    ap.add_argument("--downstream", action="store_true",
                    help="Also re-run the species analysis for every arm (slow).")
    ap.add_argument("--permutations", type=int, default=None,
                    help="Override species_analysis.n_permutations (lower = faster).")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)

    from mosmon_tracking.benchmark.evaluate import evaluate
    from mosmon_tracking.benchmark.report import write_report
    from mosmon_tracking.config import load_config

    bench = Path(args.benchmark)
    out = Path(args.out or args.benchmark)
    dets = Path(args.detections or (bench / "detections_canonical"))
    cfg = load_config(args.config)

    print(f"== evaluating {bench}")
    result = evaluate(bench, dets, out,
                      reference_arm=cfg.benchmark.reference_arm,
                      n_boot=args.bootstrap)
    print(f"   {result['summary']['n_arms']} arms x "
          f"{result['summary']['n_videos']} videos")

    downstream = None
    if args.downstream:
        downstream = run_downstream(bench, out, cfg, args.permutations)

    prov = provenance(cfg, bench, dets)
    path = write_report(result, out, downstream=downstream, provenance=prov)
    print(f"== report: {path}")
    return 0


def run_downstream(bench: Path, out: Path, cfg, permutations: int | None) -> dict:
    """Re-run the species analysis per arm, then compare across arms."""
    from mosmon_tracking.benchmark.analyze_downstream_stability import (
        arm_cohorts,
        classification_table,
        cohort_summary,
        descriptor_stability,
        load_video_features,
    )
    from mosmon_tracking.benchmark.benchmark_utils import find_arm_dirs
    from mosmon_tracking.species_analysis import species_analysis, video_feature_columns
    from mosmon_tracking.video_io import save_table

    if permutations:
        cfg.species_analysis.n_permutations = permutations

    arms = [p.name for p in find_arm_dirs(bench)]
    dirs: dict[str, Path] = {}
    for arm in arms:
        target = out / "downstream" / arm
        print(f"   species-analysis [{arm}] -> {target}")
        try:
            species_analysis(bench / arm, target, cfg)
            dirs[arm] = target
        except Exception as exc:  # noqa: BLE001 - recorded, other arms continue
            print(f"     FAILED: {exc}")
            (out / "downstream").mkdir(parents=True, exist_ok=True)
            (out / "downstream" / f"{arm}.failure.json").write_text(
                json.dumps({"arm": arm, "error": str(exc)}, indent=2), encoding="utf-8")

    cohorts = arm_cohorts(bench, cfg)
    summary = cohort_summary(cohorts)
    save_table(cohorts, out / "downstream" / "cohorts.parquet")
    save_table(summary, out / "downstream" / "cohort_summary.parquet")

    vf = load_video_features(dirs)
    stability = pd_empty()
    if not vf.empty:
        cols = video_feature_columns(vf)
        stability = descriptor_stability(vf, cols)
        save_table(vf, out / "downstream" / "video_features_all_arms.parquet")
        save_table(stability, out / "downstream" / "descriptor_stability.parquet")

    clf = classification_table(dirs)
    save_table(clf, out / "downstream" / "classification_by_arm.parquet")
    return {"cohorts": summary, "classification": clf, "stability": stability}


def pd_empty():
    import pandas as pd

    return pd.DataFrame()


def provenance(cfg, bench: Path, dets: Path) -> dict:
    import platform
    import subprocess
    import sys as _sys

    import ultralytics

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    try:
        import torch

        torch_v, cuda_v = torch.__version__, torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        torch_v = cuda_v = gpu = None

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": _sys.version.split()[0],
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch_v, "cuda": cuda_v, "gpu": gpu,
        "benchmark_dir": str(bench), "detections_dir": str(dets),
        "arms": cfg.benchmark.arms,
        "detector": {"imgsz": cfg.model.imgsz, "conf": cfg.model.conf,
                     "iou": cfg.model.iou, "max_det": cfg.model.max_det,
                     "agnostic_nms": cfg.model.agnostic_nms,
                     "frame_stride": cfg.video.frame_stride},
        "command": " ".join(_sys.argv),
    }


if __name__ == "__main__":
    raise SystemExit(main())
