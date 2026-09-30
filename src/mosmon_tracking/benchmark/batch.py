"""Batch orchestration over the whole detection cache."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from ..config import Config
from .run import run_video


def resolve_arms(cfg: Config, names: list[str] | None = None) -> dict[str, str]:
    """Arm name -> tracker YAML, from ``benchmark.arms`` in the config."""
    arms = dict(getattr(cfg.benchmark, "arms", {}) or {})
    if not arms:
        raise ValueError("config has no benchmark.arms; nothing to run")
    if names:
        missing = [n for n in names if n not in arms]
        if missing:
            raise ValueError(f"unknown arms {missing}; config defines {sorted(arms)}")
        arms = {n: arms[n] for n in names}
    return arms


def build_arms(cfg: Config, names: list[str] | None = None, device: str | None = None):
    """Instantiate every arm, failing loudly if an appearance arm loses its ReID."""
    from ..trackers import build_arm

    device = device if device is not None else cfg.model.device
    return [build_arm(name, yaml, device=device,
                      reid_fallback=getattr(cfg.benchmark, "reid_fallback", None))
            for name, yaml in resolve_arms(cfg, names).items()]


def find_detection_caches(detections_dir: str | Path) -> list[Path]:
    d = Path(detections_dir)
    return sorted(p for p in d.glob("*.parquet") if not p.name.startswith("_"))


def _find_video(videos_dir: str | Path | None, video_id: str, cfg: Config) -> Path | None:
    if not videos_dir:
        return None
    root = Path(videos_dir)
    if root.is_file():
        return root
    for ext in cfg.input.video_extensions:
        for cand in (root.rglob(f"{video_id}{ext}") if cfg.input.recursive
                     else root.glob(f"{video_id}{ext}")):
            return cand
        for cand in (root.rglob(f"{video_id}{ext.upper()}") if cfg.input.recursive
                     else root.glob(f"{video_id}{ext.upper()}")):
            return cand
    return None


def benchmark_batch(
    detections_dir: str | Path,
    cfg: Config,
    out_dir: str | Path,
    videos_dir: str | Path | None = None,
    arm_names: list[str] | None = None,
    skip_existing: bool = False,
    write_native: bool = True,
) -> pd.DataFrame:
    """Replay every cached detection table through every arm."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = find_detection_caches(detections_dir)
    arms = build_arms(cfg, arm_names)
    need_video = [a.name for a in arms if a.needs_frames]
    replay_conf = getattr(cfg.benchmark, "replay_conf", None)

    model_names = _model_names(cfg)
    rows = []
    for i, cache in enumerate(caches, 1):
        video_id = cache.stem
        manifest_path = out_dir / "_manifests" / f"{video_id}.json"
        if skip_existing and manifest_path.exists():
            rows.append({"video_id": video_id, "status": "skipped"})
            print(f"[{i}/{len(caches)}] skip {video_id}", flush=True)
            continue

        video_path = _find_video(videos_dir, video_id, cfg)
        if need_video and video_path is None:
            msg = (f"arms {need_video} need frames but no video found for "
                   f"{video_id!r} under {videos_dir!r}")
            (out_dir / "_manifests").mkdir(parents=True, exist_ok=True)
            manifest_path.with_suffix(".failure.json").write_text(
                json.dumps({"video_id": video_id, "error": msg}, indent=2),
                encoding="utf-8")
            rows.append({"video_id": video_id, "status": "failed", "error": msg})
            print(f"[{i}/{len(caches)}] FAILED {video_id}: {msg}", flush=True)
            continue

        print(f"[{i}/{len(caches)}] replay {video_id}", flush=True)
        try:
            man = run_video(cache, arms, cfg, out_dir, video_path=video_path,
                            replay_conf=replay_conf, write_native=write_native,
                            model_names=model_names)
            rows.append({
                "video_id": video_id, "status": "ok",
                "n_detections": man["detections"]["n_detections"],
                "n_frames": man["n_frames_replayed"],
                "replay_seconds": man["replay_seconds"],
                "decode_seconds": man["decode_seconds"],
                "checks_ok": man["checks"]["all_ok"],
            })
        except Exception as exc:  # noqa: BLE001 - recorded, batch continues
            (out_dir / "_manifests").mkdir(parents=True, exist_ok=True)
            manifest_path.with_suffix(".failure.json").write_text(
                json.dumps({"video_id": video_id, "error": str(exc),
                            "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8")
            rows.append({"video_id": video_id, "status": "failed", "error": str(exc)})
            print(f"    FAILED: {exc}", flush=True)

    index = pd.DataFrame(rows)
    if not index.empty:
        index.to_csv(out_dir / "benchmark_index.csv", index=False)
    return index


def _model_names(cfg: Config) -> dict | None:
    """Class names from the configured model, so track tables carry labels."""
    paths = list(cfg.model.paths or [])
    if not paths:
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(paths[0])).names
    except Exception:
        return None
