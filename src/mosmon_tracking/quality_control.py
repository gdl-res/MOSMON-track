"""Detection/tracking quality metrics.

Behaviour analysis is only as good as the tracks, so we always compute a QC
report independent of the biology. Returned as a plain dict so it serialises
straight to JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _num(x) -> float:
    return float(x) if np.isfinite(x) else None  # type: ignore[return-value]


def compute_qc(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    cfg: Config,
    n_video_frames: int | None = None,
    dropped: pd.DataFrame | None = None,
) -> dict:
    """Compute a QC dict from the raw and cleaned track tables."""
    qc: dict = {"warnings": []}
    if raw.empty:
        qc["warnings"].append("no detections produced for this video/model")
        qc["n_detections"] = 0
        return qc

    det_per_frame = raw.groupby("frame_idx").size()
    frames_with_det = det_per_frame.index.nunique()
    qc["n_detections"] = int(len(raw))
    qc["n_frames_with_detections"] = int(frames_with_det)
    if n_video_frames:
        qc["pct_frames_with_detections"] = _num(100.0 * frames_with_det / n_video_frames)
    qc["detections_per_frame"] = {
        "mean": _num(det_per_frame.mean()),
        "median": _num(det_per_frame.median()),
        "max": int(det_per_frame.max()),
        "histogram": np.histogram(det_per_frame, bins=min(20, max(1, det_per_frame.max())))[0].tolist(),
    }

    if "track_id" in raw.columns:
        qc["n_unique_track_ids_raw"] = int(raw["track_id"].nunique())
    if not clean.empty and "track_id" in clean.columns:
        lengths = clean.groupby("track_id").size()
        qc["n_unique_track_ids_clean"] = int(clean["track_id"].nunique())
        qc["median_track_length"] = _num(lengths.median())
        qc["fraction_very_short_tracks"] = _num(
            (lengths < cfg.tracker.min_track_length_frames * 2).mean()
        )
        qc["track_length_hist"] = np.histogram(lengths, bins=20)[0].tolist()
        if "is_interpolated" in clean.columns:
            qc["interpolation_fraction"] = _num(clean["is_interpolated"].mean())
        if "jump_flag" in clean.columns:
            qc["large_jump_fraction"] = _num(clean["jump_flag"].mean())
        if "confidence" in clean.columns:
            conf = clean["confidence"].dropna()
            qc["confidence"] = {
                "mean": _num(conf.mean()),
                "median": _num(conf.median()),
                "p05": _num(conf.quantile(0.05)) if len(conf) else None,
                "p95": _num(conf.quantile(0.95)) if len(conf) else None,
            }

    # Gap frequency: mean gaps per track from raw frame indices.
    if "track_id" in raw.columns:
        gap_counts = []
        for _, g in raw.groupby("track_id"):
            fr = np.sort(g["frame_idx"].to_numpy())
            if fr.size >= 2:
                d = np.diff(fr)
                step = d.min()
                gap_counts.append(int(np.sum(d > step)))
        if gap_counts:
            qc["mean_gaps_per_track"] = _num(np.mean(gap_counts))

    # Duplicate-track proxy. Two variants, reported side by side: the published
    # box-geometry one that gates the species cohort, and the identity-aware one
    # that actually answers "did association duplicate an individual?".
    qc["duplicate_track_proxy"] = _num(_duplicate_proxy(raw))
    qc["duplicate_id_proxy"] = _num(_duplicate_id_proxy(raw))

    if dropped is not None and not dropped.empty:
        qc["n_tracks_dropped"] = int(len(dropped))
        qc["drop_reasons"] = dropped["drop_reason"].value_counts().to_dict()

    # Warnings
    if qc.get("pct_frames_with_detections", 100) is not None and qc.get("pct_frames_with_detections", 100) < 50:
        qc["warnings"].append("fewer than 50% of frames have detections")
    if qc.get("fraction_very_short_tracks", 0) and qc["fraction_very_short_tracks"] > 0.5:
        qc["warnings"].append("more than half of tracks are very short (possible fragmentation)")
    if qc.get("interpolation_fraction", 0) and qc["interpolation_fraction"] > 0.25:
        qc["warnings"].append("more than 25% of clean rows are interpolated")
    return qc


def _duplicate_id_proxy(raw: pd.DataFrame, iou_thresh: float = 0.7,
                        sample_frames: int = 200) -> float:
    """Fraction of sampled frames with a high-IoU box pair carrying DIFFERENT ids.

    This is what :func:`_duplicate_proxy` has always claimed to measure. That
    function tests box geometry alone and counts a frame whenever any two boxes
    overlap, whether or not they belong to different tracks -- so it also fires
    on a single crowded larva pair that the tracker handled correctly.

    Both are reported. ``_duplicate_proxy`` remains the published QC gate so the
    45-video species cohort is unchanged; this one is the metric to use when
    asking whether *association* duplicated an identity.
    """
    if not {"x1", "y1", "x2", "y2", "track_id"}.issubset(raw.columns):
        return np.nan
    frames = _sample_frames(raw, sample_frames)
    hits = 0
    for f in frames:
        g = raw[raw["frame_idx"] == f]
        if len(g) < 2:
            continue
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        ids = g["track_id"].to_numpy()
        if _has_overlap_pair(boxes, iou_thresh, ids=ids):
            hits += 1
    return hits / len(frames) if len(frames) else np.nan


def _sample_frames(raw: pd.DataFrame, sample_frames: int) -> np.ndarray:
    """The same seeded frame sample both proxies use, so they stay comparable."""
    frames = raw["frame_idx"].unique()
    if len(frames) > sample_frames:
        rng = np.random.default_rng(0)
        frames = rng.choice(frames, size=sample_frames, replace=False)
    return frames


def _duplicate_proxy(raw: pd.DataFrame, iou_thresh: float = 0.7, sample_frames: int = 200) -> float:
    """Fraction of sampled frames containing >=1 high-IoU box pair.

    NOTE: despite the historical docstring, this does NOT require the two boxes
    to carry different track ids -- it is a pure box-geometry statistic. It is
    kept unchanged because it is the published QC gate; see
    :func:`_duplicate_id_proxy` for the identity-aware version.
    """
    if not {"x1", "y1", "x2", "y2", "track_id"}.issubset(raw.columns):
        return np.nan
    frames = _sample_frames(raw, sample_frames)
    hits = 0
    for f in frames:
        g = raw[raw["frame_idx"] == f]
        if len(g) < 2:
            continue
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        if _has_overlap_pair(boxes, iou_thresh):
            hits += 1
    return hits / len(frames) if len(frames) else np.nan


def _has_overlap_pair(boxes: np.ndarray, thresh: float,
                      ids: np.ndarray | None = None) -> bool:
    """Any pair of boxes overlapping at ``thresh``; optionally requiring different ids."""
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if ids is not None and ids[i] == ids[j]:
                continue
            if _iou(boxes[i], boxes[j]) >= thresh:
                return True
    return False


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
