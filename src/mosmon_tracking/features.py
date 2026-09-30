"""Behaviour features at three levels: frame, track, and video.

Frame-level cross-track quantities (nearest-neighbour distance, local density)
are added to the cleaned table; track- and video-level summaries are returned as
their own tables/dicts. All distances are in pixels unless a :class:`Calibrator`
in physical mode is supplied, in which case calibrated columns are added with an
explicit ``_mm`` suffix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import Calibrator
from .config import Config
from .track_postprocess import _present_group_keys

EPS = 1e-9


def _safe_nanmean(x) -> float:
    """np.nanmean without the 'Mean of empty slice' warning for all-NaN input."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


# --------------------------------------------------------------------------- #
# Frame-level cross-track features
# --------------------------------------------------------------------------- #
def add_cross_track_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add nearest-neighbour distance and local density per detection.

    Computed within each (video, model, tracker, frame) group using a KD-tree.
    The density radius defaults to 5% of the frame diagonal (documented, not a
    biological constant).
    """
    if df.empty:
        df["nearest_neighbor_distance_px"] = []
        df["local_density"] = []
        return df
    from scipy.spatial import cKDTree

    keys = [k for k in _present_group_keys(df) if k != "track_id"]
    nn = np.full(len(df), np.nan)
    dens = np.full(len(df), np.nan)
    k = max(1, cfg.features.nearest_neighbor_k)

    frame_keys = [*keys, "frame_idx"] if keys else ["frame_idx"]
    for _, g in df.groupby(frame_keys, sort=False):
        idx = g.index.to_numpy()
        pts = g[["cx", "cy"]].to_numpy(dtype=float)
        if len(pts) < 2:
            continue
        W = float(g["frame_width"].iloc[0]) if "frame_width" in g.columns else pts[:, 0].max()
        H = float(g["frame_height"].iloc[0]) if "frame_height" in g.columns else pts[:, 1].max()
        radius = 0.05 * float(np.hypot(W, H))
        tree = cKDTree(pts)
        kq = min(k + 1, len(pts))
        dists, _ = tree.query(pts, k=kq)
        dists = np.atleast_2d(dists)
        nn[idx] = dists[:, -1]  # k-th neighbour (excludes self at column 0)
        counts = tree.query_ball_point(pts, r=radius, return_length=True)
        dens[idx] = np.asarray(counts, dtype=float) - 1.0  # exclude self
    df = df.copy()
    df["nearest_neighbor_distance_px"] = nn
    df["local_density"] = dens
    return df


def refine_movement_state(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mark 'burst' frames using the global speed-distribution quantile."""
    if df.empty or "speed_px_s" not in df.columns:
        return df
    sp = df["speed_px_s"].to_numpy(dtype=float)
    finite = sp[np.isfinite(sp)]
    if finite.size == 0:
        return df
    thr = float(np.quantile(finite, cfg.features.burst_speed_quantile))
    df = df.copy()
    burst = np.isfinite(sp) & (sp >= thr) & (sp >= cfg.features.activity_speed_threshold_px_s)
    state = df["movement_state"].to_numpy(dtype=object)
    state[burst] = "burst"
    df["movement_state"] = state
    df.attrs["burst_speed_threshold_px_s"] = thr
    return df


# --------------------------------------------------------------------------- #
# Track-level helpers
# --------------------------------------------------------------------------- #
def path_length(cx: np.ndarray, cy: np.ndarray) -> float:
    d = np.hypot(np.diff(cx), np.diff(cy))
    return float(np.nansum(d))


def net_displacement(cx: np.ndarray, cy: np.ndarray) -> float:
    if len(cx) < 2:
        return 0.0
    return float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))


def tortuosity(cx: np.ndarray, cy: np.ndarray) -> float:
    nd = net_displacement(cx, cy)
    return float(path_length(cx, cy) / nd) if nd > EPS else np.nan


def straightness(cx: np.ndarray, cy: np.ndarray) -> float:
    pl = path_length(cx, cy)
    return float(net_displacement(cx, cy) / pl) if pl > EPS else np.nan


def radius_of_gyration(cx: np.ndarray, cy: np.ndarray) -> float:
    if len(cx) == 0:
        return np.nan
    mx, my = np.mean(cx), np.mean(cy)
    return float(np.sqrt(np.mean((cx - mx) ** 2 + (cy - my) ** 2)))


def spatial_entropy(cx_norm: np.ndarray, cy_norm: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy (nats) of occupancy over a bins x bins normalized grid."""
    if len(cx_norm) == 0:
        return np.nan
    H, _, _ = np.histogram2d(cx_norm, cy_norm, bins=bins, range=[[0, 1], [0, 1]])
    p = H.ravel()
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def mean_squared_displacement(cx: np.ndarray, cy: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """MSD(tau) for tau = 1..max_lag, assuming uniform sampling.

    Returns an array indexed by lag-1 (so result[0] is MSD at lag 1).
    """
    n = len(cx)
    if n < 2:
        return np.array([])
    max_lag = min(max_lag or (n - 1), n - 1)
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        dx = cx[lag:] - cx[:-lag]
        dy = cy[lag:] - cy[:-lag]
        out[lag - 1] = np.mean(dx ** 2 + dy ** 2)
    return out


def anomalous_diffusion_exponent(msd: np.ndarray) -> float:
    """Fit MSD ~ tau^alpha in log-log space; return alpha (1=normal diffusion)."""
    if msd.size < 2:
        return np.nan
    lags = np.arange(1, msd.size + 1)
    mask = msd > 0
    if mask.sum() < 2:
        return np.nan
    alpha, _ = np.polyfit(np.log(lags[mask]), np.log(msd[mask]), 1)
    return float(alpha)


def _quality_score(n_frames: int, mean_conf: float, interp_frac: float,
                   jump_frac: float, min_len: int) -> float:
    """Interpretable 0-1 QC score (NOT a biological metric)."""
    dur_score = min(n_frames / (3.0 * max(min_len, 1)), 1.0)
    conf_score = float(np.clip(mean_conf, 0.0, 1.0)) if np.isfinite(mean_conf) else 0.0
    interp_score = 1.0 - float(np.clip(interp_frac, 0.0, 1.0))
    jump_score = 1.0 - float(np.clip(jump_frac, 0.0, 1.0))
    score = 0.35 * dur_score + 0.30 * conf_score + 0.20 * interp_score + 0.15 * jump_score
    return float(np.clip(score, 0.0, 1.0))


def compute_track_summary(df: pd.DataFrame, cfg: Config,
                          calibrator: Calibrator | None = None) -> pd.DataFrame:
    """One row per track with the track-summary schema."""
    if df.empty:
        return pd.DataFrame()
    keys = _present_group_keys(df)
    rows = []
    for key, g in df.groupby(keys, sort=False):
        g = g.sort_values("frame_idx")
        cx = g["cx"].to_numpy(dtype=float)
        cy = g["cy"].to_numpy(dtype=float)
        sp = g["speed_px_s"].to_numpy(dtype=float)
        sp_f = sp[np.isfinite(sp)]
        acc = g["acceleration_px_s2"].to_numpy(dtype=float)
        turn = np.abs(g["turn_angle_rad"].to_numpy(dtype=float))
        n = len(g)
        interp_frac = float(g["is_interpolated"].mean()) if "is_interpolated" in g else 0.0
        jump_frac = float(g["jump_flag"].mean()) if "jump_flag" in g else 0.0
        mean_conf = float(g["confidence"].mean()) if "confidence" in g else np.nan
        if "time_s" in g and g["time_s"].notna().any():
            duration = float(np.nanmax(g["time_s"]) - np.nanmin(g["time_s"]))
        else:
            duration = np.nan

        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update(
            class_id=g["class_id"].iloc[0] if "class_id" in g else None,
            class_name=g["class_name"].iloc[0] if "class_name" in g else None,
            n_frames=n,
            start_frame=int(g["frame_idx"].iloc[0]),
            end_frame=int(g["frame_idx"].iloc[-1]),
            duration_s=duration,
            mean_confidence=mean_conf,
            interpolated_fraction=interp_frac,
            path_length_px=path_length(cx, cy),
            net_displacement_px=net_displacement(cx, cy),
            mean_speed_px_s=float(np.mean(sp_f)) if sp_f.size else np.nan,
            median_speed_px_s=float(np.median(sp_f)) if sp_f.size else np.nan,
            max_speed_px_s=float(np.max(sp_f)) if sp_f.size else np.nan,
            mean_acceleration_px_s2=float(np.nanmean(np.abs(acc))) if n > 1 else np.nan,
            mean_abs_turn_angle_rad=float(np.nanmean(turn)) if n > 1 else np.nan,
            tortuosity=tortuosity(cx, cy),
            straightness=straightness(cx, cy),
            radius_of_gyration_px=radius_of_gyration(cx, cy),
            spatial_entropy=spatial_entropy(
                g["cx_norm"].to_numpy() if "cx_norm" in g else cx / max(cx.max(), 1),
                g["cy_norm"].to_numpy() if "cy_norm" in g else cy / max(cy.max(), 1),
            ),
            border_fraction=float((g["roi_label"] == "border").mean()) if "roi_label" in g else np.nan,
            center_fraction=float((g["roi_label"] == "center").mean()) if "roi_label" in g else np.nan,
            freezing_fraction=float((g["movement_state"] == "freezing").mean()) if "movement_state" in g else np.nan,
            burst_fraction=float((g["movement_state"] == "burst").mean()) if "movement_state" in g else np.nan,
            mean_nearest_neighbor_distance_px=_safe_nanmean(g["nearest_neighbor_distance_px"])
            if "nearest_neighbor_distance_px" in g else np.nan,
            quality_score=_quality_score(n, mean_conf, interp_frac, jump_frac,
                                         cfg.tracker.min_track_length_frames),
        )
        if calibrator is not None and calibrator.mode == "scalar":
            rec["mean_speed_mm_s"] = calibrator.distance_mm(rec["mean_speed_px_s"])
            rec["path_length_mm"] = calibrator.distance_mm(rec["path_length_px"])
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_video_summary(clean: pd.DataFrame, track_summary: pd.DataFrame,
                          cfg: Config) -> list[dict]:
    """One dict per (video, model, tracker) with population-level metrics."""
    if clean.empty:
        return []
    keys = [k for k in ("video_name", "model_name", "tracker") if k in clean.columns]
    out = []
    grouper = clean.groupby(keys, sort=False) if keys else [((), clean)]
    for key, g in grouper:
        kv = key if isinstance(key, tuple) else (key,)
        n_frames_seen = g["frame_idx"].nunique()
        per_frame_counts = g.groupby("frame_idx")["track_id"].nunique()
        ts = track_summary
        if keys:
            mask = np.ones(len(ts), dtype=bool)
            for col, val in zip(keys, kv):
                mask &= (ts[col] == val).to_numpy()
            ts = ts[mask]
        n_tracks = len(ts)
        durations = ts["duration_s"].dropna()
        short_frac = float((ts["n_frames"] < cfg.tracker.min_track_length_frames * 2).mean()) if n_tracks else np.nan
        rec = dict(zip(keys, kv))
        rec.update(
            total_tracks=n_tracks,
            n_frames_with_detections=int(n_frames_seen),
            mean_active_tracks_per_frame=float(per_frame_counts.mean()) if len(per_frame_counts) else 0.0,
            max_tracks_per_frame=int(per_frame_counts.max()) if len(per_frame_counts) else 0,
            mean_track_duration_s=float(durations.mean()) if len(durations) else np.nan,
            median_track_duration_s=float(durations.median()) if len(durations) else np.nan,
            median_track_length_frames=float(ts["n_frames"].median()) if n_tracks else np.nan,
            fraction_short_tracks=short_frac,
            global_mean_speed_px_s=_safe_nanmean(g["speed_px_s"]) if "speed_px_s" in g else np.nan,
            population_activity_index=float((g["movement_state"].isin(["active", "burst"])).mean())
            if "movement_state" in g else np.nan,
            border_occupancy_fraction=float((g["roi_label"] == "border").mean()) if "roi_label" in g else np.nan,
            center_occupancy_fraction=float((g["roi_label"] == "center").mean()) if "roi_label" in g else np.nan,
            mean_nearest_neighbor_distance_px=_safe_nanmean(g["nearest_neighbor_distance_px"])
            if "nearest_neighbor_distance_px" in g else np.nan,
            interpolation_fraction=float(g["is_interpolated"].mean()) if "is_interpolated" in g else np.nan,
            large_jump_fraction=float(g["jump_flag"].mean()) if "jump_flag" in g else np.nan,
            mean_confidence=float(g["confidence"].mean()) if "confidence" in g else np.nan,
        )
        # Aggregation index: higher when larvae cluster (smaller NN distance vs frame scale).
        if "frame_width" in g.columns and np.isfinite(rec["mean_nearest_neighbor_distance_px"]):
            diag = float(np.hypot(g["frame_width"].iloc[0], g["frame_height"].iloc[0]))
            rec["aggregation_index"] = float(1.0 - rec["mean_nearest_neighbor_distance_px"] / (diag + EPS))
        out.append(rec)
    return out


def compute_population_timeseries(clean: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (video, model, tracker, frame) of population-level metrics.

    This is the per-frame data behind the "activity over time" / "speed over
    time" plots, persisted so it can be re-plotted or correlated with metadata
    conditions (lighting, turbidity, species, ...) without re-running anything.

    Speeds use only finite values (interpolated/edge frames can produce NaN/inf).
    """
    if clean is None or clean.empty or "frame_idx" not in clean.columns:
        return pd.DataFrame()

    keys = [k for k in ("video_name", "model_name", "tracker") if k in clean.columns]
    group_cols = keys + ["frame_idx"]

    # Assemble a working frame with the source columns and derived state flags,
    # so the whole table can be produced with a single vectorised groupby.
    work = pd.DataFrame({"frame_idx": clean["frame_idx"].to_numpy()}, index=clean.index)
    for k in keys:
        work[k] = clean[k].to_numpy()
    work["track_id"] = clean["track_id"].to_numpy()
    if "time_s" in clean.columns:
        work["time_s"] = clean["time_s"].to_numpy()
    if "speed_px_s" in clean.columns:
        work["speed_px_s"] = clean["speed_px_s"].replace([np.inf, -np.inf], np.nan).to_numpy()
    for col in ("acceleration_px_s2", "nearest_neighbor_distance_px", "local_density", "confidence"):
        if col in clean.columns:
            work[col] = clean[col].replace([np.inf, -np.inf], np.nan).to_numpy()
    if "movement_state" in clean.columns:
        ms = clean["movement_state"]
        work["frac_active"] = ms.isin(["active", "burst"]).to_numpy(dtype=float)
        work["frac_freezing"] = (ms == "freezing").to_numpy(dtype=float)
        work["frac_burst"] = (ms == "burst").to_numpy(dtype=float)
    if "roi_label" in clean.columns:
        work["frac_border"] = (clean["roi_label"] == "border").to_numpy(dtype=float)
        work["frac_center"] = (clean["roi_label"] == "center").to_numpy(dtype=float)
    if "is_interpolated" in clean.columns:
        work["interpolated_fraction"] = clean["is_interpolated"].fillna(False).to_numpy(dtype=float)

    spec: dict[str, tuple[str, object]] = {"n_larvae": ("track_id", "nunique")}
    if "time_s" in work.columns:
        spec["time_s"] = ("time_s", "first")
    if "speed_px_s" in work.columns:
        spec["mean_speed_px_s"] = ("speed_px_s", "mean")
        spec["median_speed_px_s"] = ("speed_px_s", "median")
        spec["max_speed_px_s"] = ("speed_px_s", "max")
    if "acceleration_px_s2" in work.columns:
        spec["mean_acceleration_px_s2"] = ("acceleration_px_s2", "mean")
    if "nearest_neighbor_distance_px" in work.columns:
        spec["mean_nearest_neighbor_distance_px"] = ("nearest_neighbor_distance_px", "mean")
    if "local_density" in work.columns:
        spec["mean_local_density"] = ("local_density", "mean")
    for frac in ("frac_active", "frac_freezing", "frac_burst", "frac_border", "frac_center"):
        if frac in work.columns:
            spec[frac] = (frac, "mean")
    if "interpolated_fraction" in work.columns:
        spec["interpolated_fraction"] = ("interpolated_fraction", "mean")
    if "confidence" in work.columns:
        spec["mean_confidence"] = ("confidence", "mean")

    ts = work.groupby(group_cols, sort=True).agg(**spec).reset_index()
    # Order columns: keys, frame, time, then metrics — and put time right after frame.
    front = keys + ["frame_idx"] + (["time_s"] if "time_s" in ts.columns else [])
    rest = [c for c in ts.columns if c not in front]
    return ts[front + rest]
