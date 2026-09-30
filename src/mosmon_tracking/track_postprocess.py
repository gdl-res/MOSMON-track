"""Track cleaning, gap handling, smoothing, and per-frame kinematics.

This module turns the raw per-frame detection table into a cleaned table with
quality flags and motion descriptors. Correctness rules we follow:

* Time deltas come from ``time_s`` and are validated (> 0) before any division;
  speeds are never computed from frame indices alone, because frame stride and
  variable FPS make that wrong.
* Gaps are filled by *interpolation only when short*; long gaps are dropped, not
  fabricated. Filled rows are flagged ``is_interpolated``.
* Implausible jumps are *flagged*, never silently deleted.
* Heading differences are wrapped to (-pi, pi] so turn angles are correct across
  the +/-pi discontinuity.

Cross-track quantities (nearest-neighbour distance, local density) are added
later by :mod:`mosmon_tracking.features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, RegionsConfig

GROUP_KEYS = ["video_name", "model_name", "tracker", "track_id"]


def _present_group_keys(df: pd.DataFrame) -> list[str]:
    keys = [k for k in GROUP_KEYS if k in df.columns]
    if "track_id" not in keys:
        raise ValueError("raw tracks must contain a 'track_id' column")
    return keys


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def classify_region(
    cx: np.ndarray,
    cy: np.ndarray,
    width: float,
    height: float,
    regions: RegionsConfig,
) -> np.ndarray:
    """Label each point as 'border', 'center', or 'intermediate'.

    A point is 'border' if within ``border_margin_fraction`` of any edge, 'center'
    if inside the central ``center_roi_fraction`` box, else 'intermediate'.
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    margin_x = regions.border_margin_fraction * width
    margin_y = regions.border_margin_fraction * height
    is_border = (
        (cx < margin_x) | (cx > width - margin_x) | (cy < margin_y) | (cy > height - margin_y)
    )
    half = regions.center_roi_fraction / 2.0
    cxlo, cxhi = (0.5 - half) * width, (0.5 + half) * width
    cylo, cyhi = (0.5 - half) * height, (0.5 + half) * height
    is_center = (cx >= cxlo) & (cx <= cxhi) & (cy >= cylo) & (cy <= cyhi)
    out = np.full(cx.shape, "intermediate", dtype=object)
    out[is_center] = "center"
    out[is_border] = "border"  # border takes precedence over center
    return out


def classify_compartment(cx: np.ndarray, width: float,
                         split_x_fraction: float | None) -> np.ndarray:
    """Label each point 'left' / 'right' for a dual-container (SX/DX) video.

    ``split_x_fraction`` is the divider position as a fraction of frame width.
    Returns all-'unassigned' when no split is configured, so the column exists
    with an honest value rather than a guessed one. The divider must be
    verified against a representative frame per video: the two boxes are not
    reliably centred (measured 0.484-0.508 across the four MOSMON dual videos).
    """
    cx = np.asarray(cx, dtype=float)
    out = np.full(cx.shape, "unassigned", dtype=object)
    if split_x_fraction is None or not np.isfinite(width) or width <= 0:
        return out
    split = float(split_x_fraction) * float(width)
    out[cx < split] = "left"
    out[cx >= split] = "right"
    return out


def add_compartment_species(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign species from the spatial compartment in dual-container videos.

    This is what makes the report's standing claim — "species assignment follows
    spatial-compartment metadata, not visual inference" — true. Without a
    configured ``regions.dual_container_split_x_fraction`` nothing is assigned
    and ``species_compartment`` stays None, because the detector's per-box class
    is not a validated species label for these videos.
    """
    from .metadata import parse_filename

    if df.empty or "cx" not in df.columns:
        return df
    df = df.copy()
    df["compartment"] = "unassigned"
    df["species_compartment"] = None
    if cfg.regions.dual_container_split_x_fraction is None:
        return df
    name_col = "video_name" if "video_name" in df.columns else None
    for name, g in (df.groupby(name_col, sort=False) if name_col else [(None, df)]):
        width = float(g["frame_width"].iloc[0]) if "frame_width" in g.columns else np.nan
        comp = classify_compartment(g["cx"].to_numpy(), width,
                                    cfg.regions.dual_container_split_x_fraction)
        df.loc[g.index, "compartment"] = comp
        if name is None:
            continue
        meta = parse_filename(str(name))
        if not meta.is_dual_container:
            continue
        mapping = {"left": meta.species_left, "right": meta.species_right}
        df.loc[g.index, "species_compartment"] = [mapping.get(c) for c in comp]
    return df


def distance_to_border(cx: np.ndarray, cy: np.ndarray, width: float, height: float) -> np.ndarray:
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    return np.minimum.reduce([cx, width - cx, cy, height - cy])


def _infer_step(frames: np.ndarray) -> int:
    """Infer the sampling step (in original frame indices) for one track."""
    if frames.size < 2:
        return 1
    diffs = np.diff(np.sort(frames))
    diffs = diffs[diffs > 0]
    return int(diffs.min()) if diffs.size else 1


def _reindex_and_interpolate(
    g: pd.DataFrame, max_gap_samples: int, do_interp: bool
) -> pd.DataFrame:
    """Reindex one track onto its full frame grid and interpolate short gaps."""
    g = g.sort_values("frame_idx")
    frames = g["frame_idx"].to_numpy()
    step = _infer_step(frames)
    full = np.arange(frames.min(), frames.max() + 1, step)
    g = g.set_index("frame_idx").reindex(full)
    g.index.name = "frame_idx"

    # Gap-aware interpolation: a gap is filled only if the *entire* gap is short
    # enough (<= max_gap_samples). Partial filling of long gaps is avoided so we
    # never fabricate trajectory through a long blackout.
    missing = g["cx"].isna()
    run = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run).transform("size")
    fillable = missing & (run_len <= max_gap_samples)
    g["is_interpolated"] = fillable.fillna(False).to_numpy()

    interp_cols = [c for c in ("x1", "y1", "x2", "y2", "cx", "cy", "time_s") if c in g.columns]
    if do_interp and max_gap_samples > 0:
        filled = g[interp_cols].interpolate(method="linear", limit_area="inside")
        # Apply interpolated values only on fillable rows; leave long gaps as NaN.
        g.loc[fillable, interp_cols] = filled.loc[fillable, interp_cols]
    # Forward/back-fill identity columns that are constant within a track.
    const_cols = [
        c for c in (
            "video_path", "video_name", "model_path", "model_name", "tracker",
            "track_id", "class_id", "class_name", "frame_width", "frame_height",
        ) if c in g.columns
    ]
    g[const_cols] = g[const_cols].ffill().bfill()
    # Rows that are still NaN in cx after interpolation are long gaps -> drop.
    g = g[g["cx"].notna()].copy()
    g["is_interpolated"] = g["is_interpolated"].fillna(False).astype(bool)
    return g.reset_index()


def _kinematics(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute per-frame motion descriptors for a single, time-sorted track."""
    g = g.sort_values("frame_idx").reset_index(drop=True)
    cx = g["cx"].to_numpy(dtype=float)
    cy = g["cy"].to_numpy(dtype=float)

    # Smoothing (Savitzky-Golay) on the center path.
    g["is_smoothed"] = False
    if cfg.postprocess.smoothing == "savgol" and len(g) >= cfg.postprocess.smoothing_window_frames:
        from scipy.signal import savgol_filter

        win = cfg.postprocess.smoothing_window_frames
        if win % 2 == 0:
            win += 1
        win = min(win, len(g) if len(g) % 2 == 1 else len(g) - 1)
        poly = min(cfg.postprocess.smoothing_polyorder, win - 1)
        if win >= 3 and poly >= 1:
            cx = savgol_filter(cx, win, poly)
            cy = savgol_filter(cy, win, poly)
            g["cx"] = cx
            g["cy"] = cy
            g["is_smoothed"] = True

    # Time deltas: validated, strictly positive.
    if "time_s" in g.columns and g["time_s"].notna().all():
        t = g["time_s"].to_numpy(dtype=float)
        dt = np.diff(t, prepend=t[0])
    else:
        dt = np.full(len(g), np.nan)
    dt[dt <= 0] = np.nan  # guard against zero/negative deltas
    g["dt"] = dt

    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    dx[0] = np.nan
    dy[0] = np.nan
    g["dx"] = dx
    g["dy"] = dy
    disp = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = disp / dt
    g["speed_px_s"] = speed

    accel = np.diff(speed, prepend=speed[0]) / dt
    accel[0] = np.nan
    g["acceleration_px_s2"] = accel

    heading = np.arctan2(dy, dx)
    g["heading_rad"] = heading
    turn = wrap_angle(np.diff(heading, prepend=heading[0]))
    turn[0] = np.nan
    g["turn_angle_rad"] = turn

    if {"w", "h"}.issubset(g.columns):
        w = g["w"].to_numpy(dtype=float)
        h = g["h"].to_numpy(dtype=float)
        g["bbox_area_px"] = w * h
        with np.errstate(invalid="ignore", divide="ignore"):
            g["bbox_aspect_ratio"] = np.where(h > 0, w / h, np.nan)

    # Region / border
    if {"frame_width", "frame_height"}.issubset(g.columns):
        W = float(g["frame_width"].iloc[0])
        H = float(g["frame_height"].iloc[0])
        g["roi_label"] = classify_region(cx, cy, W, H, cfg.regions)
        g["dist_to_border_px"] = distance_to_border(cx, cy, W, H)
        margin = cfg.regions.border_margin_fraction * min(W, H)
        g["border_flag"] = g["dist_to_border_px"] < margin

    # Jump flagging: configured absolute threshold OR per-track robust outlier.
    sp = g["speed_px_s"].to_numpy(dtype=float)
    if cfg.postprocess.max_speed_px_s is not None:
        jump = sp > cfg.postprocess.max_speed_px_s
    else:
        finite = sp[np.isfinite(sp)]
        if finite.size >= 4:
            med = np.median(finite)
            mad = np.median(np.abs(finite - med)) or np.nan
            thr = med + 8.0 * 1.4826 * mad if np.isfinite(mad) else np.inf
            jump = sp > thr
        else:
            jump = np.zeros_like(sp, dtype=bool)
    g["jump_flag"] = np.where(np.isfinite(sp), jump, False).astype(bool)

    # Low confidence per frame
    if "confidence" in g.columns:
        g["low_confidence_flag"] = (
            g["confidence"] < cfg.tracker.min_mean_confidence
        ).fillna(False)

    # Movement state (burst is refined later against the global distribution).
    fr = cfg.features.freezing_speed_threshold_px_s
    ac = cfg.features.activity_speed_threshold_px_s
    state = np.full(len(g), "slow", dtype=object)
    state[sp < fr] = "freezing"
    state[sp >= ac] = "active"
    state[~np.isfinite(sp)] = "unknown"
    g["movement_state"] = state

    return g


def filter_tracks(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop tracks that are too short or too low-confidence.

    Returns (kept_df, dropped_summary) where dropped_summary records why each
    removed track id was removed (for the QC report).
    """
    keys = _present_group_keys(df)
    records = []
    keep_mask = pd.Series(True, index=df.index)
    for key, g in df.groupby(keys, sort=False):
        n = len(g)
        mean_conf = g["confidence"].mean() if "confidence" in g.columns else np.nan
        reason = None
        if n < cfg.tracker.min_track_length_frames:
            reason = f"short_track(n={n})"
        elif np.isfinite(mean_conf) and mean_conf < cfg.tracker.min_mean_confidence:
            reason = f"low_mean_confidence({mean_conf:.3f})"
        if reason:
            keep_mask.loc[g.index] = False
            rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            rec["n_frames"] = n
            rec["mean_confidence"] = mean_conf
            rec["drop_reason"] = reason
            records.append(rec)
    dropped = pd.DataFrame(records)
    return df[keep_mask].copy(), dropped


def clean_tracks(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full cleaning pipeline: sort -> filter -> interpolate -> smooth -> kinematics.

    Returns (clean_df, dropped_summary).
    """
    if df.empty:
        return df.copy(), pd.DataFrame()
    keys = _present_group_keys(df)
    df = df.sort_values([*keys, "frame_idx"]).reset_index(drop=True)

    kept, dropped = filter_tracks(df, cfg)
    if kept.empty:
        return kept, dropped

    out_groups = []
    for _, g in kept.groupby(keys, sort=False):
        g = _reindex_and_interpolate(
            g,
            max_gap_samples=cfg.postprocess.interpolation_max_gap_frames,
            do_interp=cfg.postprocess.interpolate_gaps,
        )
        # Recompute normalized centers (interpolation may have added rows).
        if {"frame_width", "frame_height"}.issubset(g.columns):
            g["cx_norm"] = g["cx"] / g["frame_width"]
            g["cy_norm"] = g["cy"] / g["frame_height"]
        g = _kinematics(g, cfg)
        out_groups.append(g)

    clean = pd.concat(out_groups, ignore_index=True)
    clean = clean.sort_values([*keys, "frame_idx"]).reset_index(drop=True)
    return clean, dropped
