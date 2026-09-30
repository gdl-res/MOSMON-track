"""Pre-run checks: de-risk a long batch before committing the machine to it.

Verifies that every video is readable, the model files exist, the GPU is present
(when the config asks for CUDA), the config has no silently-ignored typo keys, and
that there's enough free disk — then estimates total runtime from the calibration.
Returns a structured report and a list of hard blockers.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from . import runtime
from .config import Config, find_unknown_keys
from .metadata import parse_filename
from .video_io import find_videos, probe_video


def _gpu_status(cfg: Config) -> dict:
    want_cuda = str(cfg.model.device).lower() not in ("cpu", "")
    info: dict[str, Any] = {"requested_device": cfg.model.device, "cuda_available": None}
    try:
        import torch
        avail = bool(torch.cuda.is_available())
        info["cuda_available"] = avail
        if avail:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info["vram_free_gb"] = round(free / 1e9, 2)
            info["vram_total_gb"] = round(total / 1e9, 2)
    except Exception as exc:  # torch missing or driver issue
        info["error"] = str(exc)
    info["blocker"] = bool(want_cuda and info.get("cuda_available") is False)
    return info


def _mean_bundle_size_bytes(out_dir: Path) -> int | None:
    """Mean size of any finished bundles under ``out_dir`` (for disk estimates)."""
    sizes = []
    for d in out_dir.glob("*"):
        if d.is_dir() and (d / "video_summary.json").exists():
            sizes.append(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()))
    return int(sum(sizes) / len(sizes)) if sizes else None


def run_preflight(videos_root: str | Path, models: list[str | Path], cfg: Config,
                  out_dir: str | Path, config_path: str | Path | None = None) -> dict:
    """Run all pre-run checks and return a report dict (see module docstring)."""
    out = Path(out_dir)
    blockers: list[str] = []
    warnings: list[str] = []

    # 1. Config typo / unknown-key check (silent-drop guard).
    unknown_keys: list[str] = []
    if config_path and Path(config_path).exists():
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        unknown_keys = find_unknown_keys(raw)
        for k in unknown_keys:
            blockers.append(f"unknown config key (silently ignored): {k}")

    # 2. Models exist and are non-empty.
    model_report = []
    for m in models:
        p = Path(m)
        ok = p.is_file() and p.stat().st_size > 0
        model_report.append({"path": str(p), "ok": ok,
                             "size_mb": round(p.stat().st_size / 1e6, 1) if p.is_file() else 0})
        if not ok:
            blockers.append(f"model missing or empty: {p}")
    if not models:
        blockers.append("no models provided")

    # 3. Videos: readability + resolution/duration for the ETA, plus metadata coverage.
    videos = find_videos(videos_root, cfg.input)
    if not videos:
        blockers.append(f"no videos found under {videos_root}")
    unreadable, no_species = [], []
    total_seconds = 0.0
    total_frames = 0.0
    est_output_bytes = 0.0
    by_res: dict[str, int] = {}
    for vp in videos:
        info = probe_video(vp)
        if not info.width or not info.height or not info.duration_s:
            unreadable.append(vp.name)
            continue
        mp = info.width * info.height / 1e6
        bucket = ("8K" if mp > 28 else "5.7K" if mp > 20 else "5.3K" if mp > 12
                  else "4K" if mp > 6 else "≤1080p")
        by_res[bucket] = by_res.get(bucket, 0) + 1
        total_seconds += runtime.estimate_video_seconds(
            info.width, info.height, info.frame_count, info.fps, info.duration_s, cfg)
        total_frames += runtime.processed_frames(info.frame_count, info.fps, info.duration_s, cfg)
        est_output_bytes += runtime.estimate_video_output_bytes(
            info.frame_count, info.fps, info.duration_s, cfg)
        if not parse_filename(vp.name).species:
            no_species.append(vp.name)
    est_output_bytes *= len(models)
    for u in unreadable:
        blockers.append(f"unreadable video: {u}")
    if no_species:
        warnings.append(f"{len(no_species)} video(s) have no parseable species "
                        f"(per-species grouping will be incomplete)")

    # 4. Disk: estimate output footprint vs free space on the out volume.
    #    Prefer a measured mean from any finished bundles in out_dir; otherwise
    #    fall back to the calibration-derived bytes-per-frame model above.
    out.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(out).free
    mean_bundle = _mean_bundle_size_bytes(out)
    if mean_bundle:
        est_output_bytes = mean_bundle * len(videos) * len(models)
    if est_output_bytes > free_bytes:
        blockers.append(f"insufficient disk: need ~{est_output_bytes/1e9:.0f} GB, "
                        f"free {free_bytes/1e9:.0f} GB")
    elif est_output_bytes > 0.8 * free_bytes:
        warnings.append(f"disk tight: est ~{est_output_bytes/1e9:.0f} GB output vs "
                        f"{free_bytes/1e9:.0f} GB free (>80%)")

    gpu = _gpu_status(cfg)
    if gpu.get("blocker"):
        blockers.append(f"config requests device '{cfg.model.device}' but CUDA is unavailable")

    return {
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "n_videos": len(videos),
        "n_models": len(models),
        "n_runs": len(videos) * len(models),
        "resolution_mix": by_res,
        "unreadable_videos": unreadable,
        "videos_without_species": no_species,
        "unknown_config_keys": unknown_keys,
        "models": model_report,
        "gpu": gpu,
        "estimated_processed_frames": int(total_frames * len(models)),
        "estimated_runtime_hours": round(total_seconds * len(models) / 3600, 1),
        "disk_free_gb": round(free_bytes / 1e9, 1),
        "estimated_output_gb": round(est_output_bytes / 1e9, 1) if est_output_bytes else None,
        "tracker": cfg.tracker.type,
    }
