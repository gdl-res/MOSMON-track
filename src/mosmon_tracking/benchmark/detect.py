"""Build the frozen detection cache: one detector pass per video, no tracker."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import pandas as pd

from ..config import Config
from ..video_io import find_videos, probe_video
from ..yolo_tracker import run_detection_only
from .benchmark_utils import check_detection_table
from .detections_io import detection_fingerprint, write_detections


def detect_video(video_path: str | Path, model_path: str | Path, cfg: Config,
                 out_dir: str | Path) -> dict:
    """Detect one video and write ``<out_dir>/<video_id>.parquet`` + a sidecar."""
    video_path, model_path = Path(video_path), Path(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = video_path.stem

    t0 = time.time()
    det = run_detection_only(video_path, model_path, cfg)
    elapsed = time.time() - t0

    path = write_detections(det, out_dir / f"{video_id}.parquet")
    info = probe_video(video_path)
    report = check_detection_table(det, max_det=cfg.model.max_det)
    meta = {
        "video_id": video_id,
        "video_path": str(video_path),
        "model_path": str(model_path),
        "model_name": model_path.stem,
        "detections_path": str(path),
        "n_detections": int(len(det)),
        "n_frames": int(det["processed_frame"].nunique()) if len(det) else 0,
        "elapsed_seconds": elapsed,
        "video": {"width": info.width, "height": info.height, "fps": info.fps,
                  "frame_count": info.frame_count, "duration_s": info.duration_s},
        "detector": {
            "imgsz": cfg.model.imgsz, "conf": cfg.model.conf, "iou": cfg.model.iou,
            "agnostic_nms": cfg.model.agnostic_nms, "max_det": cfg.model.max_det,
            "half": cfg.model.half, "classes": cfg.model.classes,
            "frame_stride": cfg.video.frame_stride,
        },
        "fingerprint": detection_fingerprint(det),
        "checks": report.as_dict(),
    }
    (out_dir / f"{video_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def detect_batch(videos: str | Path, model_path: str | Path, cfg: Config,
                 out_dir: str | Path, skip_existing: bool = False) -> pd.DataFrame:
    """Detect every video under ``videos`` into one shared detection cache."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = find_videos(videos, cfg.input)
    rows = []
    for i, vp in enumerate(paths, 1):
        vid = Path(vp).stem
        target = out_dir / f"{vid}.parquet"
        meta_path = out_dir / f"{vid}.meta.json"
        if skip_existing and target.exists() and meta_path.exists():
            rows.append({"video_id": vid, "status": "skipped",
                         **_brief(json.loads(meta_path.read_text()))})
            print(f"[{i}/{len(paths)}] skip {vid}", flush=True)
            continue
        print(f"[{i}/{len(paths)}] detect {vid}", flush=True)
        try:
            meta = detect_video(vp, model_path, cfg, out_dir)
            rows.append({"video_id": vid, "status": "ok", **_brief(meta)})
        except Exception as exc:  # noqa: BLE001 - recorded, batch continues
            (out_dir / f"{vid}.failure.json").write_text(
                json.dumps({"video_id": vid, "error": str(exc),
                            "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8")
            rows.append({"video_id": vid, "status": "failed", "error": str(exc)})
            print(f"    FAILED: {exc}", flush=True)
    index = pd.DataFrame(rows)
    if not index.empty:
        index.to_csv(out_dir / "detection_index.csv", index=False)
    return index


def _brief(meta: dict) -> dict:
    return {
        "n_detections": meta.get("n_detections"),
        "n_frames": meta.get("n_frames"),
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "checks_ok": meta.get("checks", {}).get("ok"),
    }
