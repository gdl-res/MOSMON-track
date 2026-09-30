"""Runtime estimation from the measured 3-video calibration.

These constants come from a calibration run on this machine (RTX 4060 8 GB,
mosmon_yolov11x, BoT-SORT, stride 2, imgsz 1920) — see the project memory note
``full-eval-runtime-calibration``. They let ``preflight`` and ``batch-status``
project wall-clock from the per-video resolution + frame budget.
"""

from __future__ import annotations

import numpy as np

from .config import Config

# (megapixels, seconds per processed frame) measured under BoT-SORT.
CALIB_POINTS_BOTSORT = [(8.29, 0.302), (15.87, 0.409), (33.18, 0.754)]
# ByteTrack measured ~2x faster than BoT-SORT in the tuning sweep.
BYTETRACK_SPEEDUP = 2.0

# Output-size model from the same calibration bundles, broken into per-frame
# components so the estimate can follow which outputs a config actually writes.
# Mean across the 4K/5.3K/8K samples (90-180 larvae/frame) was ~62 KB/frame.
# Heatmaps/plots are ~fixed per bundle.
BYTES_PER_FRAME_CORE = 42_000          # detections/tracks parquet (always written)
BYTES_PER_FRAME_MOT = 6_000            # tracks_mot.txt (reports.export_mot)
BYTES_PER_FRAME_GEOJSON = 12_000       # trajectories.geojson (export_trajectories_geojson)
BYTES_PER_FRAME_TIMESERIES = 2_000     # population_timeseries (export_population_timeseries)
FIXED_BYTES_PER_BUNDLE = 6_500_000     # heatmaps + plots + reports + ~fixed debug clip

_MP = [p[0] for p in CALIB_POINTS_BOTSORT]
_SPF = [p[1] for p in CALIB_POINTS_BOTSORT]


def seconds_per_frame(megapixels: float, tracker: str = "botsort") -> float:
    """Seconds per processed frame at a given source resolution (interpolated)."""
    spf = float(np.interp(megapixels, _MP, _SPF))  # clamped to measured range
    if str(tracker).lower().startswith("byte"):
        spf /= BYTETRACK_SPEEDUP
    return spf


def processed_frames(frame_count: int | None, fps: float | None,
                     duration_s: float | None, cfg: Config) -> float:
    """How many frames the pipeline will run for a video under ``cfg``."""
    n = frame_count if frame_count else ((fps or 0) * (duration_s or 0))
    stride = max(1, cfg.video.frame_stride)
    n = n / stride
    if cfg.video.max_frames:
        n = min(n, cfg.video.max_frames)
    return float(n)


def estimate_video_seconds(width: int, height: int, frame_count: int | None,
                           fps: float | None, duration_s: float | None,
                           cfg: Config) -> float:
    """Estimated wall-clock seconds to process one video under ``cfg``."""
    mp = (width * height) / 1e6 if width and height else 16.0
    n = processed_frames(frame_count, fps, duration_s, cfg)
    return n * seconds_per_frame(mp, cfg.tracker.type)


def estimate_video_output_bytes(frame_count: int | None, fps: float | None,
                                duration_s: float | None, cfg: Config) -> float:
    """Estimated output-bundle size in bytes for one video under ``cfg``.

    Follows the config's output toggles so a lean profile (MOT/GeoJSON off)
    estimates smaller, matching what it will actually write.
    """
    n = processed_frames(frame_count, fps, duration_s, cfg)
    per_frame = BYTES_PER_FRAME_CORE
    if cfg.reports.export_mot:
        per_frame += BYTES_PER_FRAME_MOT
    if cfg.reports.export_trajectories_geojson:
        per_frame += BYTES_PER_FRAME_GEOJSON
    if cfg.features.export_population_timeseries:
        per_frame += BYTES_PER_FRAME_TIMESERIES
    return n * per_frame + FIXED_BYTES_PER_BUNDLE
