"""Per-video benchmark orchestration: replay every arm and write run folders.

Each arm's raw table is routed through the *existing* pipeline
(:func:`mosmon_tracking.batch.pipeline_from_raw`), unchanged. That is deliberate:
§19-§20 require that downstream methodology is identical across arms, and the
cheapest way to guarantee it is to reuse one implementation rather than
reimplement cleaning per arm.

Two reconstruction variants are written per arm:

* **Experiment B, common reconstruction** -- the full MOSMON cleaning,
  interpolation and smoothing every arm receives identically. This is the run
  folder that feeds species-analysis.
* **Experiment A, tracker-native** -- only the minimum filtering needed to
  compute descriptors at all. This measures what each arm directly produces,
  before the reconstruction pipeline has a chance to normalise differences away.

Never compare A of one arm against B of another.
"""

from __future__ import annotations

import copy
import gc
import json
import time
from pathlib import Path

import pandas as pd

from ..config import Config
from ..video_io import save_table
from ..yolo_tracker import empty_raw_table
from .benchmark_utils import check_track_table, summarise
from .detections_io import read_detections
from .identity_agreement import pairwise_agreement
from .run_tracker_benchmark import replay, set_class_names


def native_config(cfg: Config) -> Config:
    """Config for Experiment A: the least processing descriptors still allow.

    No interpolation and no smoothing, so the trajectory is exactly what the
    tracker emitted. Track-length and confidence gates are dropped to their
    floor rather than to zero: a one-observation track has no displacement, no
    speed and no duration, so it cannot enter any descriptor anyway.
    """
    out = copy.deepcopy(cfg)
    out.postprocess.interpolate_gaps = False
    out.postprocess.smoothing = "none"
    out.tracker.min_track_length_frames = 2
    out.tracker.min_mean_confidence = 0.0
    return out


def run_video(
    detections_path: str | Path,
    arms: list,
    cfg: Config,
    out_dir: str | Path,
    video_path: str | Path | None = None,
    replay_conf: float | None = None,
    write_native: bool = True,
    model_names: dict | None = None,
) -> dict:
    """Replay one video's detection cache through every arm; write run folders."""
    from ..batch import pipeline_from_raw
    from ..calibration import Calibrator

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = read_detections(detections_path, conf_min=replay_conf,
                          half=bool(cfg.model.half))
    if det.empty:
        raise ValueError(f"no detections in {detections_path}")

    video_id = str(det["video_id"].iloc[0])
    if model_names:
        set_class_names(model_names)

    t0 = time.time()
    results, stats = replay(
        det, arms,
        video_path=video_path,
        model_name=str(det["model_name"].iloc[0]),
        model_path=str(det["model_path"].iloc[0]),
    )
    replay_seconds = time.time() - t0

    # The detection table is dead once every arm has consumed it, and it is big:
    # a dense video caches 30 M+ boxes. Holding it through the postprocessing of
    # every arm is what turned a tight run into an OOM kill.
    del det
    gc.collect()

    calibrator = Calibrator.from_config(cfg.calibration)
    native_cfg = native_config(cfg)
    per_arm: dict[str, dict] = {}
    reports = []

    for name in list(results):
        res = results[name]
        arm_dir = out_dir / name / video_id
        arm_dir.mkdir(parents=True, exist_ok=True)

        # The mapping from shared detections to this arm's track ids. This is
        # what makes the cross-arm identity comparison possible at all.
        save_table(res.assignments, arm_dir / "assignments.parquet")

        t1 = time.time()
        summary = pipeline_from_raw(res.tracks, cfg, arm_dir, video_path=video_path,
                                    calibrator=calibrator)
        post_seconds = time.time() - t1

        native = {}
        if write_native:
            native = _write_native(res.tracks, native_cfg, arm_dir, calibrator)

        rep = check_track_table(res.tracks, name)
        reports.append(rep)

        timing = {
            "association_seconds": res.association_seconds,
            "postprocess_seconds": post_seconds,
            "frames": res.n_frames,
            "association_fps": (res.n_frames / res.association_seconds
                                if res.association_seconds > 0 else None),
        }
        (arm_dir / "arm_metadata.json").write_text(json.dumps({
            "arm": name,
            "video_id": video_id,
            "params": res.params,
            "provenance": res.provenance,
            "timing": timing,
            "detections": stats["detections"],
            "checks": rep.as_dict(),
            "native": native,
        }, indent=2, default=str), encoding="utf-8")
        # pipeline_from_raw returns {video_info, model_settings, qc, behaviour}.
        per_arm[name] = {
            "timing": timing,
            "checks": rep.as_dict(),
            "n_raw_rows": res.n_output_rows,
            "behaviour": (summary or {}).get("behaviour", {}),
            "qc": {k: (summary or {}).get("qc", {}).get(k) for k in (
                "n_unique_track_ids_raw", "n_unique_track_ids_clean",
                "median_track_length", "interpolation_fraction",
                "large_jump_fraction", "duplicate_track_proxy",
                "duplicate_id_proxy")},
        }
        # This arm's raw table has now been written twice over (Experiment B and
        # Experiment A) and is no longer needed; the small ``assignments`` table
        # is kept because the cross-arm agreement is computed after the loop.
        res.tracks = empty_raw_table()
        del summary
        gc.collect()

    agreement = pairwise_agreement({n: r.assignments for n, r in results.items()})
    save_table(agreement, out_dir / "_agreement" / f"{video_id}.parquet")

    manifest = {
        "video_id": video_id,
        "video_path": str(video_path) if video_path else None,
        "detections_path": str(detections_path),
        "replay_conf": replay_conf,
        "detections": stats["detections"],
        "decode_seconds": stats["decode_seconds"],
        "replay_seconds": replay_seconds,
        "n_frames_replayed": stats["n_frames_replayed"],
        "arms": per_arm,
        "checks": summarise(reports),
    }
    (out_dir / "_manifests").mkdir(parents=True, exist_ok=True)
    (out_dir / "_manifests" / f"{video_id}.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def _write_native(raw: pd.DataFrame, native_cfg: Config, arm_dir: Path,
                  calibrator) -> dict:
    """Experiment A: minimally filtered, un-smoothed, un-interpolated tracks."""
    from .. import features as feat
    from ..track_postprocess import clean_tracks

    clean, _dropped = clean_tracks(raw, native_cfg)
    if clean.empty:
        return {"n_tracks": 0, "n_rows": 0}
    clean = feat.add_cross_track_features(clean, native_cfg)
    clean = feat.refine_movement_state(clean, native_cfg)
    save_table(clean, arm_dir / "tracks_native.parquet", keep_categories=True)
    summary = feat.compute_track_summary(clean, native_cfg, calibrator)
    save_table(summary, arm_dir / "track_summary_native.parquet")
    return {"n_tracks": int(len(summary)), "n_rows": int(len(clean))}
