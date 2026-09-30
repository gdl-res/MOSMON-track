"""The §6.5 reproduction gate, as a callable check."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..video_io import load_table
from .detections_io import read_detections
from .reproduction import compare_track_tables, verdict
from .run_tracker_benchmark import replay


def validate_arm(
    detections_dir: str | Path,
    video: str | Path,
    reference: str | Path,
    cfg: Config,
    arm: str = "botsort",
    tracker_yaml: str = "configs/tracker_botsort.yaml",
    max_frames: int | None = None,
) -> dict:
    """Replay one arm and compare it to a reference track table.

    The cached pool is filtered to ``benchmark.reference_conf`` first, so the arm
    sees exactly the detections the reference run saw rather than the wider pool
    the benchmark normally uses.
    """
    from ..trackers import build_arm

    video = Path(video)
    det_path = Path(detections_dir)
    if det_path.is_dir():
        det_path = det_path / f"{video.stem}.parquet"
    ref_conf = getattr(cfg.benchmark, "reference_conf", 0.15)

    det = read_detections(det_path, conf_min=ref_conf, half=bool(cfg.model.half))
    if max_frames:
        det = det[det["processed_frame"] < max_frames]
    if det.empty:
        raise ValueError(f"no detections at conf>{ref_conf} in {det_path}")

    adapter = build_arm(arm, tracker_yaml, device=cfg.model.device)
    results, _stats = replay(det, [adapter], video_path=video)
    mine = results[arm].tracks

    ref = load_table(Path(reference))
    if ref is None or ref.empty:
        raise ValueError(f"reference table is empty: {reference}")
    if max_frames:
        last = int(det["source_frame"].max())
        ref = ref[ref["frame_idx"] <= last]

    cmp = compare_track_tables(mine, ref)
    passed, message = verdict(cmp)
    return {
        "arm": arm,
        "video": str(video),
        "reference": str(reference),
        "reference_conf": ref_conf,
        "max_frames": max_frames,
        "comparison": cmp,
        "passed": passed,
        "message": message,
    }
