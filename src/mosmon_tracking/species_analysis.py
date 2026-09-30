"""Species discrimination from tracking metrics.

Answers one question over a finished batch: *do any tracking-derived metrics
distinguish the four MOSMON-Larvae species?* — and is built so that the answer
"no, not separably" is a first-class outcome rather than a failure mode.

Three things make a naive per-species table (``aggregate_by_species.parquet``)
unusable as evidence, and each has an explicit counter-measure here:

1. **Duplicate and mislabelled inputs.** The full eval contains three sha256
   duplicate source videos, two pairs of which carry contradictory species or
   lighting labels. :func:`build_cohort` drops them and records why.
2. **Pixel units across a 16x resolution range.** Speeds in px/s are not
   comparable between a 1080p and an 8K recording. Every kinematic feature here
   is either dimensionless or divided by the animal's own body length
   (``body_length_px``, the median bbox major axis of that track).
3. **Density confounds everything.** Larval density spans 200x and is almost
   perfectly confounded with species. Every comparison is repeated
   density-adjusted and on a density-matched subset, and every classifier score
   is reported beside a density-only baseline.

Ground truth for species is the *filename metadata* (``meta_species``). The
detector's own class prediction is not used as a label: on this batch it agrees
with the filename label in only 45/66 videos.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, SpeciesAnalysisConfig

EPS = 1e-9

#: Track-level features that are dimensionless or body-length normalised, i.e.
#: comparable across videos of different resolution. The classifier and the
#: univariate tests both draw from this list.
SCALE_FREE_FEATURES = [
    "mean_speed_bl_s", "median_speed_bl_s", "p90_speed_bl_s", "max_speed_bl_s",
    "speed_cv", "speed_burstiness", "speed_ac1", "speed_acorr_time_s",
    "path_length_bl", "net_displacement_bl", "radius_of_gyration_bl",
    "nn_distance_bl", "straightness", "log_tortuosity", "msd_alpha", "msd_r2",
    "turn_abs_mean_rad", "turn_circular_variance", "reversal_fraction", "turn_ac1",
    "pause_fraction", "move_bout_mean_s", "pause_bout_mean_s", "bout_rate_hz",
    "aspect_ratio_cv", "aspect_ratio_ac1",
    "aspect_peak_freq_hz", "aspect_peak_power_fraction",
    "spatial_entropy_track",
]

#: Reported separately and kept out of the classifier: these describe the animal's
#: apparent size and shape rather than what it does. ``aspect_ratio_mean`` is the
#: bbox w/h level, which tracks body elongation and camera orientation;
#: ``aspect_ratio_cv``/``_ac1`` (how the ratio *changes*) stay behavioural.
MORPHOMETRIC_FEATURES = ["body_length_px", "bbox_area_med_px", "aspect_ratio_mean"]

#: Scale-free but not individual behaviour: nearest-neighbour distance is set by
#: how many larvae are in the container, so it re-imports the density confound.
#: Reported, but the classifier is also scored with these dropped.
DENSITY_LINKED_FEATURES = ["nn_distance_bl"]

#: Kept only as a labelled sensitivity block — NOT comparable across resolutions.
PIXEL_FEATURES = ["mean_speed_px_s", "median_speed_px_s", "radius_of_gyration_px",
                  "mean_nearest_neighbor_distance_px"]

_CLEAN_COLUMNS = [
    "track_id", "frame_idx", "time_s", "cx", "cy", "cx_norm", "cy_norm",
    "w", "h", "speed_px_s", "turn_angle_rad", "nearest_neighbor_distance_px",
    "compartment", "species_compartment",
]


# --------------------------------------------------------------------------- #
# 1. Cohort construction
# --------------------------------------------------------------------------- #
def _source_sha(run_dir: Path) -> str | None:
    """sha256 of the source video from a run's provenance.json, if recorded."""
    prov = run_dir / "provenance.json"
    if not prov.exists():
        return None
    try:
        data = json.loads(prov.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for item in data.get("inputs", []) or []:
        if item.get("role") == "source-video":
            return item.get("sha256")
    return None


def _coverage_pct(qc_full: dict, qc: dict, vi: dict, ms: dict) -> float | None:
    """Percentage of processed frames carrying at least one detection.

    ``quality_control`` only writes ``pct_frames_with_detections`` when it is
    given the processed-frame count, and the benchmark's replay path never
    passes one -- so every tracker-benchmark run lacks the key and would be
    dropped by the cohort's coverage gate as "QC metrics missing".

    The value is recoverable exactly from fields the run *does* store, using the
    same definition: frames-with-detections over ``ceil(frame_count / stride)``.
    Verified to reproduce the stored value to the last digit on the published
    full-eval runs, so a backfilled cohort and a natively-scored one agree.
    """
    stored = qc_full.get("pct_frames_with_detections",
                         qc.get("pct_frames_with_detections"))
    if stored is not None:
        return stored
    n_with = qc_full.get("n_frames_with_detections", qc.get("n_frames_with_detections"))
    frame_count = vi.get("frame_count")
    stride = ms.get("frame_stride") or 1
    if not n_with or not frame_count or stride < 1:
        return None
    processed = math.ceil(float(frame_count) / float(stride))
    return 100.0 * float(n_with) / processed if processed else None


def _run_record(summary_path: Path) -> dict | None:
    """Flatten one run's video_summary.json + qc_report.json + provenance.json."""
    run_dir = summary_path.parent
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    vi = data.get("video_info", {}) or {}
    ms = data.get("model_settings", {}) or {}
    beh = data.get("behaviour", {}) or {}
    qc = data.get("qc", {}) or {}

    # qc_report.json is the superset of warnings; video_summary.json is written
    # earlier in the run and misses anything raised afterwards (e.g. PDF skips).
    qc_full = {}
    qc_path = run_dir / "qc_report.json"
    if qc_path.exists():
        try:
            qc_full = json.loads(qc_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            qc_full = {}

    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "video": vi.get("filename") or run_dir.name,
        "sha256": _source_sha(run_dir),
        "frame_stride": ms.get("frame_stride"),
        "tracker": beh.get("tracker") or ms.get("tracker"),
        "model": beh.get("model_name"),
        "width": vi.get("width"),
        "height": vi.get("height"),
        "fps": vi.get("fps"),
        "duration_s": vi.get("duration_s"),
        "coverage_pct": _coverage_pct(qc_full, qc, vi, ms),
        "duplicate_track_proxy": qc_full.get("duplicate_track_proxy",
                                             qc.get("duplicate_track_proxy")),
        "interpolation_fraction": qc_full.get("interpolation_fraction"),
        "n_qc_warnings": len(qc_full.get("warnings", qc.get("warnings", [])) or []),
        "total_tracks": beh.get("total_tracks"),
        "density": beh.get("mean_active_tracks_per_frame"),
        "median_track_duration_s": beh.get("median_track_duration_s"),
        "mean_nn_distance_px_video": beh.get("mean_nearest_neighbor_distance_px"),
        "aggregation_index": beh.get("aggregation_index"),
        "heatmap_entropy_occupancy": beh.get("heatmap_entropy_occupancy"),
        "border_occupancy_fraction": beh.get("border_occupancy_fraction"),
        "center_occupancy_fraction": beh.get("center_occupancy_fraction"),
        "population_activity_index": beh.get("population_activity_index"),
    }
    row.update({k: v for k, v in vi.items() if k.startswith("meta_")})
    return row


def _duplicate_verdicts(runs: pd.DataFrame, opts: SpeciesAnalysisConfig) -> pd.Series:
    """Per-run exclusion reason for sha256-identical source videos ('' = keep).

    Within a duplicate group, a *conflict* is a disagreement on any label that
    the analysis groups by (species, lighting, stage, container). Conflicting
    groups are dropped whole: at least one label is wrong and we cannot tell
    which. Consistent groups keep their first run and drop the rest.
    """
    reason = pd.Series("", index=runs.index, dtype=object)
    if not opts.drop_duplicate_inputs or "sha256" not in runs.columns:
        return reason
    label_cols = [c for c in ("meta_species", "meta_lighting", "meta_stage",
                              "meta_container_type") if c in runs.columns]
    for sha, grp in runs[runs["sha256"].notna()].groupby("sha256"):
        if len(grp) < 2:
            continue
        conflicts = [c for c in label_cols if grp[c].astype(str).nunique() > 1]
        if conflicts and opts.drop_conflicting_duplicates:
            reason.loc[grp.index] = (
                f"duplicate_input_conflicting_labels({','.join(conflicts)}; sha={sha[:12]})"
            )
        else:
            keep = grp.sort_values("run").index[0]
            drop = [i for i in grp.index if i != keep]
            reason.loc[drop] = f"duplicate_input(sha={sha[:12]})"
    return reason


def build_cohort(batch_dir: str | Path, cfg: Config | None = None) -> pd.DataFrame:
    """Per-run table with an auditable ``excluded_reason`` ('' = in the cohort).

    Applies, in order: duplicate-input removal, dual-container separation, and
    the QC gate (``duplicate_track_proxy`` and detection coverage). Nothing is
    silently dropped — every excluded run keeps its row and its reason.
    """
    cfg = cfg or Config()
    opts = cfg.species_analysis
    batch_dir = Path(batch_dir)

    rows = [r for p in sorted(batch_dir.glob("*/video_summary.json"))
            if (r := _run_record(p)) is not None]
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs

    reason = _duplicate_verdicts(runs, opts)

    species = runs.get("meta_species", pd.Series("", index=runs.index)).fillna("")
    is_dual = runs.get("meta_is_dual_container", pd.Series(False, index=runs.index)).fillna(False)
    is_dual = is_dual.astype(bool) | species.astype(str).str.contains(r"\|")
    runs["is_dual_container"] = is_dual
    if opts.exclude_dual_container:
        reason = reason.mask((reason == "") & is_dual, "dual_container(analysed separately)")
    reason = reason.mask((reason == "") & (species.astype(str).str.strip() == ""),
                         "no species in filename metadata")

    if opts.apply_qc_gate:
        dup = pd.to_numeric(runs.get("duplicate_track_proxy"), errors="coerce")
        cov = pd.to_numeric(runs.get("coverage_pct"), errors="coerce")
        reason = reason.mask((reason == "") & (dup > opts.max_duplicate_track_proxy),
                            dup.map(lambda v: f"duplicate_track_proxy={v:.3f} > "
                                              f"{opts.max_duplicate_track_proxy}"))
        reason = reason.mask((reason == "") & (cov < opts.min_coverage_pct),
                            cov.map(lambda v: f"coverage={v:.1f}% < {opts.min_coverage_pct}%"))
        reason = reason.mask((reason == "") & (dup.isna() | cov.isna()),
                             "QC metrics missing")

    runs["excluded_reason"] = reason
    runs["in_cohort"] = reason == ""
    return runs


# --------------------------------------------------------------------------- #
# 2. Track-level features (scale-free / body-length normalised)
# --------------------------------------------------------------------------- #
def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation for lags 1..max_lag (NaN when undefined)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    max_lag = int(min(max_lag, n - 2))
    if n < 4 or max_lag < 1:
        return np.full(max(int(max_lag), 0), np.nan)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= EPS:
        return np.full(max_lag, np.nan)
    return np.array([float(np.dot(xc[lag:], xc[:-lag]) / denom)
                     for lag in range(1, max_lag + 1)])


def autocorrelation_time_s(acf: np.ndarray, dt_s: float) -> float:
    """First lag (in seconds) at which the autocorrelation drops below 1/e.

    Linearly interpolated between lags; NaN if it never drops within max_lag,
    which is itself informative (persistent motion).
    """
    if acf.size == 0 or not np.isfinite(dt_s) or dt_s <= 0:
        return np.nan
    thr = 1.0 / np.e
    prev = 1.0
    for i, val in enumerate(acf, start=1):
        if not np.isfinite(val):
            return np.nan
        if val < thr:
            frac = (prev - thr) / (prev - val) if (prev - val) > EPS else 0.0
            return float((i - 1 + frac) * dt_s)
        prev = val
    return np.nan


def _bout_stats(is_pause: np.ndarray, dt_s: float) -> tuple[float, float, float]:
    """Mean move-bout duration, mean pause-bout duration (s), and bout rate (Hz)."""
    n = is_pause.size
    if n < 2 or not np.isfinite(dt_s) or dt_s <= 0:
        return np.nan, np.nan, np.nan
    change = np.flatnonzero(np.diff(is_pause.astype(int)) != 0) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [n]))) * dt_s
    states = is_pause[starts]
    move = lengths[~states]
    pause = lengths[states]
    rate = float((len(starts) - 1) / (n * dt_s)) if n * dt_s > 0 else np.nan
    return (float(move.mean()) if move.size else np.nan,
            float(pause.mean()) if pause.size else np.nan,
            rate)


def dominant_frequency(x: np.ndarray, fs: float,
                       min_hz: float = 0.3) -> tuple[float, float]:
    """Peak frequency (Hz) of a detrended signal and its share of total power.

    Used on the bbox aspect-ratio series as a *proxy* for body undulation. With
    30 fps at stride 2 the Nyquist limit is 7.5 Hz, so genuine wriggle frequency
    is partially aliased — treat the value as a texture descriptor, not a rate.
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 16 or not np.isfinite(fs) or fs <= 0:
        return np.nan, np.nan
    x = x[ok]
    n = x.size
    x = x - np.polyval(np.polyfit(np.arange(n), x, 1), np.arange(n))  # linear detrend
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = freqs >= min_hz
    total = float(spec[1:].sum())
    if not band.any() or total <= EPS:
        return np.nan, np.nan
    idx = int(np.argmax(np.where(band, spec, -np.inf)))
    return float(freqs[idx]), float(spec[idx] / total)


def track_shape_features(g: pd.DataFrame, opts: SpeciesAnalysisConfig) -> dict:
    """Scale-free + body-length-normalised descriptors for one track.

    ``g`` must be one track's rows from ``tracks_clean``, sorted by frame.
    Body length is the median bbox major axis of the track itself, which makes
    every derived speed/distance comparable across resolutions.
    """
    from . import features as feat

    cx = g["cx"].to_numpy(dtype=float)
    cy = g["cy"].to_numpy(dtype=float)
    w = g["w"].to_numpy(dtype=float)
    h = g["h"].to_numpy(dtype=float)
    sp = g["speed_px_s"].to_numpy(dtype=float)
    turn = g["turn_angle_rad"].to_numpy(dtype=float)
    t = g["time_s"].to_numpy(dtype=float)
    n = len(g)

    bl = float(np.nanmedian(np.maximum(w, h))) if n else np.nan
    if not np.isfinite(bl) or bl <= EPS:
        bl = np.nan
    dts = np.diff(t)
    dt_s = float(np.nanmedian(dts)) if dts.size else np.nan
    fs = 1.0 / dt_s if np.isfinite(dt_s) and dt_s > 0 else np.nan

    sp_f = sp[np.isfinite(sp)]
    sp_bl = sp_f / bl if np.isfinite(bl) else np.array([])
    turn_f = turn[np.isfinite(turn)]
    ar = np.where(h > 0, w / h, np.nan)

    out: dict[str, float] = {
        "n_frames_used": n,
        "dt_s": dt_s,
        "body_length_px": bl,
        "bbox_area_med_px": float(np.nanmedian(w * h)) if n else np.nan,
    }

    # --- speed (body lengths per second) ---
    if sp_bl.size:
        mean_sp, std_sp = float(sp_bl.mean()), float(sp_bl.std())
        out.update(
            mean_speed_bl_s=mean_sp,
            median_speed_bl_s=float(np.median(sp_bl)),
            p90_speed_bl_s=float(np.quantile(sp_bl, 0.90)),
            max_speed_bl_s=float(sp_bl.max()),
            speed_cv=std_sp / mean_sp if mean_sp > EPS else np.nan,
            speed_burstiness=((std_sp - mean_sp) / (std_sp + mean_sp)
                              if (std_sp + mean_sp) > EPS else np.nan),
        )
    acf_sp = autocorrelation(sp, opts.autocorr_max_lag)
    out["speed_ac1"] = float(acf_sp[0]) if acf_sp.size else np.nan
    out["speed_acorr_time_s"] = autocorrelation_time_s(acf_sp, dt_s)

    # --- geometry (body lengths) ---
    path = feat.path_length(cx, cy)
    net = feat.net_displacement(cx, cy)
    if np.isfinite(bl):
        out.update(path_length_bl=path / bl, net_displacement_bl=net / bl,
                   radius_of_gyration_bl=feat.radius_of_gyration(cx, cy) / bl)
        if "nearest_neighbor_distance_px" in g.columns:
            nn_vals = g["nearest_neighbor_distance_px"].to_numpy(dtype=float)
            nn_vals = nn_vals[np.isfinite(nn_vals)]
            if nn_vals.size:
                out["nn_distance_bl"] = float(np.median(nn_vals)) / bl
    out["straightness"] = feat.straightness(cx, cy)
    # Tortuosity divides by net displacement and blows up on near-stationary
    # tracks (max 50,721 in this batch). Guard on one body length, then log.
    out["log_tortuosity"] = (float(np.log(path / net))
                             if np.isfinite(bl) and net > bl and path > EPS else np.nan)

    # --- diffusion ---
    msd = feat.mean_squared_displacement(cx, cy, max_lag=min(opts.msd_max_lag, max(n // 4, 2)))
    out["msd_alpha"] = feat.anomalous_diffusion_exponent(msd)
    if msd.size >= 2:
        lags, m = np.arange(1, msd.size + 1), msd
        ok = m > 0
        if ok.sum() >= 2:
            lx, ly = np.log(lags[ok]), np.log(m[ok])
            pred = np.polyval(np.polyfit(lx, ly, 1), lx)
            ss_tot = float(((ly - ly.mean()) ** 2).sum())
            out["msd_r2"] = float(1 - ((ly - pred) ** 2).sum() / ss_tot) if ss_tot > EPS else np.nan

    # --- turning ---
    if turn_f.size:
        out.update(
            turn_abs_mean_rad=float(np.abs(turn_f).mean()),
            turn_circular_variance=float(1.0 - np.abs(np.exp(1j * turn_f).mean())),
            reversal_fraction=float((np.abs(turn_f) > 2.5).mean()),
        )
    acf_turn = autocorrelation(turn, opts.autocorr_max_lag)
    out["turn_ac1"] = float(acf_turn[0]) if acf_turn.size else np.nan

    # --- intermittency (threshold in body lengths/s, so resolution-free) ---
    if sp_bl.size and np.isfinite(dt_s):
        is_pause = sp_bl < opts.pause_speed_bl_s
        out["pause_fraction"] = float(is_pause.mean())
        move_s, pause_s, rate = _bout_stats(is_pause, dt_s)
        out.update(move_bout_mean_s=move_s, pause_bout_mean_s=pause_s, bout_rate_hz=rate)

    # --- posture proxy from the bbox aspect ratio ---
    ar_f = ar[np.isfinite(ar)]
    if ar_f.size:
        mean_ar = float(ar_f.mean())
        out["aspect_ratio_mean"] = mean_ar
        out["aspect_ratio_cv"] = float(ar_f.std() / mean_ar) if mean_ar > EPS else np.nan
    acf_ar = autocorrelation(ar, opts.autocorr_max_lag)
    out["aspect_ratio_ac1"] = float(acf_ar[0]) if acf_ar.size else np.nan
    peak_hz, peak_frac = dominant_frequency(ar, fs)
    out["aspect_peak_freq_hz"] = peak_hz
    out["aspect_peak_power_fraction"] = peak_frac

    if {"cx_norm", "cy_norm"}.issubset(g.columns):
        out["spatial_entropy_track"] = feat.spatial_entropy(
            g["cx_norm"].to_numpy(dtype=float), g["cy_norm"].to_numpy(dtype=float))
    return out


def _run_track_features(run_dir: Path, opts: SpeciesAnalysisConfig,
                        seed: int = 42) -> pd.DataFrame:
    """Track features for one run folder, merged onto its track_summary rows."""
    from .video_io import load_table

    summ_path = next((run_dir / n for n in ("track_summary.parquet", "track_summary.csv")
                      if (run_dir / n).exists()), None)
    clean_path = next((run_dir / n for n in ("tracks_clean.parquet", "tracks_clean.csv")
                       if (run_dir / n).exists()), None)
    if summ_path is None or clean_path is None:
        return pd.DataFrame()

    summ = load_table(summ_path)
    if summ.empty:
        return pd.DataFrame()
    keep = summ[summ["duration_s"] >= opts.min_track_duration_s].copy()
    if keep.empty:
        return pd.DataFrame()
    if opts.max_tracks_per_video and len(keep) > opts.max_tracks_per_video:
        keep = keep.sample(n=opts.max_tracks_per_video,
                           random_state=seed).sort_values("track_id")
    wanted = set(keep["track_id"].tolist())

    if clean_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(clean_path).schema_arrow.names)
        clean = pq.read_table(clean_path,
                              columns=[c for c in _CLEAN_COLUMNS if c in available]
                              ).to_pandas()
    else:
        clean = load_table(clean_path)
    clean = clean[clean["track_id"].isin(wanted)]
    if clean.empty:
        return pd.DataFrame()

    rows = []
    for tid, g in clean.sort_values(["track_id", "frame_idx"]).groupby("track_id", sort=False):
        rec = {"track_id": tid}
        rec.update(track_shape_features(g, opts))
        if "compartment" in g.columns:
            counts = g["compartment"].value_counts()
            rec["compartment"] = str(counts.index[0])
            # The two boxes are physically separate: a track that changes side
            # is an ID error, not a larva, so purity is a usable filter.
            rec["compartment_purity"] = float(counts.iloc[0] / len(g))
        if "species_compartment" in g.columns:
            # Take the species of the *majority* compartment, not of the first
            # row, so a few jittered frames cannot flip a track's label.
            side = g[g["compartment"] == rec.get("compartment")] if "compartment" in g else g
            sc = side["species_compartment"].dropna()
            rec["species_compartment"] = str(sc.iloc[0]) if len(sc) else None
        rows.append(rec)
    shape = pd.DataFrame(rows)
    # Recomputed columns win over the ones already in track_summary (identical
    # definitions), so the merge cannot produce _x/_y duplicates.
    dupe_cols = [c for c in shape.columns if c != "track_id" and c in keep.columns]
    merged = keep.drop(columns=dupe_cols).merge(shape, on="track_id", how="inner")
    merged.insert(0, "run", run_dir.name)
    return merged


def extract_track_features(cohort: pd.DataFrame, cfg: Config | None = None,
                           cache_path: str | Path | None = None,
                           runs: list[str] | None = None) -> pd.DataFrame:
    """Track-level feature table for every run in ``cohort`` (cached to Parquet).

    The cache is keyed on the run list plus the options that change the numbers;
    a stale cache is recomputed rather than silently reused.
    """
    from .video_io import load_table, save_table

    cfg = cfg or Config()
    opts = cfg.species_analysis
    sel = cohort if runs is None else cohort[cohort["run"].isin(runs)]
    if runs is None and "in_cohort" in sel.columns:
        sel = sel[sel["in_cohort"].astype(bool)]
    key = json.dumps({"runs": sorted(sel["run"].tolist()),
                      "min_dur": opts.min_track_duration_s,
                      "max_tracks": opts.max_tracks_per_video,
                      "pause": opts.pause_speed_bl_s,
                      "msd_lag": opts.msd_max_lag,
                      "acf_lag": opts.autocorr_max_lag}, sort_keys=True)

    cache_path = Path(cache_path) if cache_path else None
    meta_path = cache_path.with_suffix(".key.json") if cache_path else None
    if cache_path and cache_path.exists() and meta_path and meta_path.exists():
        if meta_path.read_text(encoding="utf-8") == key:
            return load_table(cache_path)

    frames = []
    for _, row in sel.iterrows():
        run_dir = Path(row["run_dir"])
        df = _run_track_features(run_dir, opts, seed=cfg.project.random_seed)
        if df.empty:
            warnings.warn(f"no usable tracks for run {row['run']}", stacklevel=2)
            continue
        frames.append(df)
    tracks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not tracks.empty:
        meta_cols = [c for c in cohort.columns
                     if c.startswith("meta_") or c in ("video", "density", "sha256",
                                                       "width", "height", "fps")]
        tracks = tracks.merge(cohort[["run", *meta_cols]], on="run", how="left")
    if cache_path is not None and not tracks.empty:
        save_table(tracks, cache_path)
        meta_path.write_text(key, encoding="utf-8")
    return tracks


# --------------------------------------------------------------------------- #
# 3. Video-level aggregation
# --------------------------------------------------------------------------- #
def as_labels(x: pd.Series, missing: str = "unknown") -> pd.Series:
    """String labels with missing values made explicit (pandas keeps NA through astype)."""
    return x.astype(object).where(x.notna(), missing).astype(str)


def _iqr(x: pd.Series) -> float:
    v = x.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.subtract(*np.percentile(v, [75, 25]))) if v.size >= 4 else np.nan


def video_feature_table(tracks: pd.DataFrame, cohort: pd.DataFrame,
                        feature_cols: list[str] | None = None) -> pd.DataFrame:
    """One row per video: median and IQR of each track feature + population metrics.

    The video is the unit of analysis for every statistical claim — tracks
    within a video are not independent observations of a species.
    """
    if tracks.empty:
        return pd.DataFrame()
    feature_cols = feature_cols or [c for c in SCALE_FREE_FEATURES + MORPHOMETRIC_FEATURES
                                    if c in tracks.columns]
    grouped = tracks.groupby("run", sort=False)
    med = grouped[feature_cols].median(numeric_only=True).add_suffix("_med")
    iqr = grouped[feature_cols].agg(_iqr).add_suffix("_iqr")
    out = pd.concat([med, iqr], axis=1).reset_index()
    out.insert(1, "n_tracks_used", grouped.size().to_numpy())

    pop_cols = [c for c in ("video", "density", "total_tracks", "median_track_duration_s",
                            "aggregation_index", "heatmap_entropy_occupancy",
                            "border_occupancy_fraction", "center_occupancy_fraction",
                            "population_activity_index", "coverage_pct",
                            "duplicate_track_proxy", "width", "height", "fps", "sha256")
                if c in cohort.columns]
    meta_cols = [c for c in cohort.columns if c.startswith("meta_")]
    return out.merge(cohort[["run", *pop_cols, *meta_cols]], on="run", how="left")


def video_feature_columns(video_table: pd.DataFrame,
                          max_missing_fraction: float = 0.2) -> list[str]:
    """Aggregated feature columns usable for statistics (drop mostly-empty ones)."""
    cand = [f"{f}{sfx}" for f in SCALE_FREE_FEATURES for sfx in ("_med", "_iqr")]
    cand += ["aggregation_index", "heatmap_entropy_occupancy",
             "border_occupancy_fraction", "center_occupancy_fraction",
             "population_activity_index"]
    out = []
    for c in cand:
        if c not in video_table.columns:
            continue
        col = pd.to_numeric(video_table[c], errors="coerce")
        if col.notna().mean() >= (1 - max_missing_fraction) and col.nunique(dropna=True) > 1:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# 4. Univariate statistics
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (same order as the input)."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    n = vals.size
    adj = vals[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def residualize(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Residuals of ``values`` after an OLS fit on ``covariate`` (NaN-safe)."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(covariate, dtype=float)
    ok = np.isfinite(v) & np.isfinite(c)
    out = np.full(v.shape, np.nan)
    if ok.sum() < 3 or np.ptp(c[ok]) <= EPS:
        return out
    slope, intercept = np.polyfit(c[ok], v[ok], 1)
    out[ok] = v[ok] - (slope * c[ok] + intercept)
    return out


def univariate_tests(video_table: pd.DataFrame, feature_cols: list[str],
                     group_col: str = "meta_species",
                     covariate_col: str = "density",
                     opts: SpeciesAnalysisConfig | None = None) -> pd.DataFrame:
    """Kruskal-Wallis per feature across species, in three confound variants.

    ``raw`` uses the video-level values; ``density_adjusted`` uses residuals
    after regressing out log10(density); ``density_matched`` restricts to the
    density band where all species are represented. A feature that only
    separates species in the ``raw`` variant is describing crowding.
    """
    from scipy import stats

    opts = opts or SpeciesAnalysisConfig()
    if video_table.empty:
        return pd.DataFrame()
    dens = pd.to_numeric(video_table.get(covariate_col), errors="coerce").to_numpy(dtype=float)
    log_dens = np.log10(np.where(dens > 0, dens, np.nan))
    matched = (dens >= opts.density_match_min) & (dens <= opts.density_match_max)
    groups = as_labels(video_table[group_col]).to_numpy()

    rows = []
    for variant in ("raw", "density_adjusted", "density_matched"):
        for feat_col in feature_cols:
            vals = pd.to_numeric(video_table[feat_col], errors="coerce").to_numpy(dtype=float)
            mask = np.ones(vals.shape, dtype=bool)
            if variant == "density_adjusted":
                vals = residualize(vals, log_dens)
            elif variant == "density_matched":
                mask = matched
            samples, labels = [], []
            for sp in sorted(set(groups[mask])):
                s = vals[mask & (groups == sp)]
                s = s[np.isfinite(s)]
                if s.size >= 3:
                    samples.append(s)
                    labels.append(sp)
            if len(samples) < 2:
                continue
            H, p = stats.kruskal(*samples)
            grand = np.concatenate(samples)
            rows.append({
                "feature": feat_col, "variant": variant, "n_groups": len(samples),
                "n_videos": int(sum(s.size for s in samples)),
                "kruskal_H": float(H), "p": float(p),
                # epsilon-squared: KW effect size, 0-1, comparable across features
                "epsilon_sq": float((H - len(samples) + 1) / (grand.size - len(samples)))
                if grand.size > len(samples) else np.nan,
                "groups": "|".join(labels),
                **{f"median[{lab}]": float(np.median(s)) for lab, s in zip(labels, samples, strict=True)},
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = np.concatenate([
        benjamini_hochberg(g["p"].to_numpy()) for _, g in out.groupby("variant", sort=False)
    ])
    out["significant"] = out["p_fdr"] < (opts.fdr_alpha if opts else 0.05)
    return out.sort_values(["variant", "p"]).reset_index(drop=True)


def pairwise_tests(video_table: pd.DataFrame, feature_cols: list[str],
                   group_col: str = "meta_species",
                   opts: SpeciesAnalysisConfig | None = None) -> pd.DataFrame:
    """Mann-Whitney U per species pair with rank-biserial effect sizes."""
    from itertools import combinations

    from scipy import stats

    opts = opts or SpeciesAnalysisConfig()
    groups = as_labels(video_table[group_col])
    rows = []
    for feat_col in feature_cols:
        vals = pd.to_numeric(video_table[feat_col], errors="coerce")
        for a, b in combinations(sorted(groups.unique()), 2):
            xa = vals[groups == a].dropna().to_numpy()
            xb = vals[groups == b].dropna().to_numpy()
            if xa.size < 3 or xb.size < 3:
                continue
            U, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            rows.append({"feature": feat_col, "species_a": a, "species_b": b,
                         "n_a": xa.size, "n_b": xb.size, "U": float(U), "p": float(p),
                         # rank-biserial: -1..1, sign says which species is higher
                         "rank_biserial": float(2 * U / (xa.size * xb.size) - 1),
                         "median_a": float(np.median(xa)), "median_b": float(np.median(xb))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = benjamini_hochberg(out["p"].to_numpy())
    out["significant"] = out["p_fdr"] < opts.fdr_alpha
    return out.sort_values("p").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Multivariate classification (video-disjoint)
# --------------------------------------------------------------------------- #
def _design_matrix(X: np.ndarray, medians: np.ndarray, mean: np.ndarray,
                   std: np.ndarray) -> np.ndarray:
    """Impute NaNs with training medians, standardise, prepend an intercept."""
    Z = np.where(np.isfinite(X), X, medians)
    Z = (Z - mean) / np.where(std > EPS, std, 1.0)
    return np.hstack([np.ones((Z.shape[0], 1)), Z])


def fit_multinomial_logistic(X: np.ndarray, Y: np.ndarray, l2: float = 1.0,
                             max_iter: int = 200) -> np.ndarray:
    """L2-regularised multinomial logistic regression (numpy + scipy L-BFGS-B).

    Deliberately dependency-free: scikit-learn is not part of this project's
    environment and the project avoids heavy dependencies. ``X`` must already
    include the intercept column; the intercept is not penalised.
    """
    from scipy.optimize import minimize

    n, d = X.shape
    k = Y.shape[1]

    def obj(theta: np.ndarray):
        W = theta.reshape(d, k)
        Z = X @ W
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        denom = expZ.sum(axis=1, keepdims=True)
        nll = float(-(np.sum(Z * Y) - np.log(denom).sum()) / n)
        P = expZ / denom
        grad = X.T @ (P - Y) / n
        pen = W.copy()
        pen[0] = 0.0
        return nll + 0.5 * l2 * float((pen ** 2).sum()), (grad + l2 * pen).ravel()

    res = minimize(obj, np.zeros(d * k), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    return res.x.reshape(d, k)


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, l2: float = 1.0,
               max_iter: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-disjoint cross-validated predictions (labels, probabilities, classes).

    One fold per unique group — leave-one-video-out or leave-one-camera-out.
    No row from a held-out group ever appears in its own training fold, which is
    the whole point: tracks from one video are not independent samples.
    """
    classes = np.array(sorted(set(y)))
    idx = {c: i for i, c in enumerate(classes)}
    proba = np.full((len(y), classes.size), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() < 2 or len(set(y[tr])) < 2:
            continue
        Xtr_raw, Xte_raw = X[tr], X[te]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = np.nanmedian(Xtr_raw, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(Xtr_raw), Xtr_raw, medians)
        mean, std = filled.mean(axis=0), filled.std(axis=0)
        Xtr = _design_matrix(Xtr_raw, medians, mean, std)
        Xte = _design_matrix(Xte_raw, medians, mean, std)
        ytr = y[tr]
        Y = np.zeros((ytr.size, classes.size))
        for i, lab in enumerate(ytr):
            Y[i, idx[lab]] = 1.0
        W = fit_multinomial_logistic(Xtr, Y, l2=l2, max_iter=max_iter)
        Z = Xte @ W
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        proba[te] = expZ / expZ.sum(axis=1, keepdims=True)
    pred = np.where(np.isfinite(proba[:, 0]), classes[np.nanargmax(np.nan_to_num(proba, nan=-1), axis=1)], None)
    return np.array(pred, dtype=object), proba, classes


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall — the honest score under 22-vs-6 class imbalance."""
    ok = np.array([p is not None for p in y_pred])
    if not ok.any():
        return np.nan
    recalls = []
    for c in sorted(set(y_true[ok])):
        m = (y_true == c) & ok
        if m.any():
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.crosstab(pd.Series(y_true, name="true"),
                       pd.Series(y_pred, name="predicted"), dropna=False)


def session_crossed_species(video_table: pd.DataFrame,
                            group_col: str = "meta_species",
                            session_col: str = "meta_recording_date") -> list[str]:
    """Species that share at least one recording session with another species.

    A species filmed only on days when nothing else was filmed cannot be told
    apart from its session: any classifier separating it may be recognising the
    day, the water batch or the rig setup rather than the animal.
    """
    if session_col not in video_table.columns:
        return sorted(as_labels(video_table[group_col]).unique())
    tab = pd.crosstab(as_labels(video_table[group_col]), as_labels(video_table[session_col]))
    shared = tab.columns[(tab > 0).sum(axis=0) > 1]
    return sorted(tab.index[(tab[shared] > 0).any(axis=1)]) if len(shared) else []


def _feature_matrix(table: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return table[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def scored_against_null(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        opts: SpeciesAnalysisConfig, rng: np.random.Generator,
                        n_permutations: int | None = None) -> dict:
    """Group-disjoint balanced accuracy with its own label-permutation null.

    Each cross-validation scheme needs its own null: leave-one-group-out with
    grouped labels is *negatively* biased (holding out a group also removes it
    from training), and the size of that bias depends on the number and balance
    of the groups. Comparing a leave-one-session-out score against 1/k, or
    against the leave-one-video-out null, would be wrong.
    """
    pred, _, _ = cv_predict(X, y, groups, opts.logistic_l2, opts.logistic_max_iter)
    obs = balanced_accuracy(y, pred)
    out = {"balanced_accuracy": obs, "n_groups": int(len(set(groups)))}
    n_perm = opts.n_permutations if n_permutations is None else n_permutations
    if n_perm and np.isfinite(obs):
        null = []
        for _ in range(int(n_perm)):
            yp = rng.permutation(y)
            p_, _, _ = cv_predict(X, yp, groups, opts.logistic_l2, opts.logistic_max_iter)
            null.append(balanced_accuracy(yp, p_))
        arr = np.array([v for v in null if np.isfinite(v)])
        if arr.size:
            out.update(null_mean=float(arr.mean()),
                       null_p95=float(np.quantile(arr, 0.95)),
                       p_value=float((np.sum(arr >= obs) + 1) / (arr.size + 1)),
                       n_permutations=int(arr.size))
    return out


def classify_species(video_table: pd.DataFrame, feature_cols: list[str],
                     tracks: pd.DataFrame | None = None,
                     cfg: Config | None = None) -> dict:
    """Video-disjoint species classification with its nulls and controls.

    Returns a dict with the video-level and (optional) track-level scores, plus
    the three baselines that decide whether a score means anything:
    label permutation, a density-only model, and the same features asked to
    predict *camera model* instead of species.
    """
    cfg = cfg or Config()
    opts = cfg.species_analysis
    rng = np.random.default_rng(cfg.project.random_seed)

    y = as_labels(video_table["meta_species"]).to_numpy()
    runs = video_table["run"].to_numpy()
    X = _feature_matrix(video_table, feature_cols)
    classes, counts = np.unique(y, return_counts=True)
    majority = float(counts.max() / counts.sum())
    chance = 1.0 / classes.size

    pred, _, _ = cv_predict(X, y, runs, opts.logistic_l2, opts.logistic_max_iter)
    video_bacc = balanced_accuracy(y, pred)

    result = {
        "n_videos": int(len(y)),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "classes": classes.tolist(),
        "class_counts": {c: int(n) for c, n in zip(classes, counts, strict=True)},
        "majority_class_accuracy": majority,
        "chance_balanced_accuracy": chance,
        "video_level": {
            "balanced_accuracy_lovo": video_bacc,
            "accuracy_lovo": float(np.mean([p == t for p, t in zip(pred, y, strict=True) if p is not None])),
            "confusion": confusion_table(y, pred).to_dict(),
        },
    }

    # Leave-one-camera-out: a harsher split — the model must generalise to a rig
    # it has never seen, which is where "species signal" usually turns out to be
    # camera signal.
    if "meta_camera_model" in video_table.columns:
        cams = as_labels(video_table["meta_camera_model"]).to_numpy()
        loco = scored_against_null(X, y, cams, opts, rng)
        result["video_level"]["balanced_accuracy_loco"] = loco["balanced_accuracy"]
        result["leave_one_camera_out"] = loco
        # Control: same features, camera as the target.
        if len(set(cams)) > 1:
            pred_rig, _, _ = cv_predict(X, cams, runs, opts.logistic_l2, opts.logistic_max_iter)
            result["camera_control"] = {
                "balanced_accuracy_lovo": balanced_accuracy(cams, pred_rig),
                "n_cameras": int(len(set(cams))),
                "chance_balanced_accuracy": 1.0 / len(set(cams)),
            }

    # Session-disjoint validation. Videos of one species were often shot in one
    # session, so a species-looking signal can be a session fingerprint (water
    # batch, turbidity, rig setup). Holding out whole recording days tests that.
    if "meta_recording_date" in video_table.columns:
        dates = as_labels(video_table["meta_recording_date"]).to_numpy()
        if len(set(dates)) > 1:
            lodo = scored_against_null(X, y, dates, opts, rng)
            result["video_level"]["balanced_accuracy_lodo"] = lodo["balanced_accuracy"]
            result["leave_one_session_out"] = lodo
        crossed = session_crossed_species(video_table)
        excluded = sorted(set(y) - set(crossed))
        if excluded and len(crossed) >= 2:
            m = np.isin(y, list(crossed))
            pred_c, _, _ = cv_predict(X[m], y[m], runs[m], opts.logistic_l2,
                                      opts.logistic_max_iter)
            result["session_crossed_subset"] = {
                "species_kept": sorted(crossed),
                "species_dropped_single_session": excluded,
                "n_videos": int(m.sum()),
                "balanced_accuracy_lovo": balanced_accuracy(y[m], pred_c),
                "chance_balanced_accuracy": 1.0 / len(crossed),
            }

    # Density-only baseline: if log density alone matches the full model, the
    # "behavioural" separation is crowding.
    dens = pd.to_numeric(video_table.get("density"), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(dens).sum() >= 5:
        Xd = np.log10(np.where(dens > 0, dens, np.nan)).reshape(-1, 1)
        pred_d, _, _ = cv_predict(Xd, y, runs, opts.logistic_l2, opts.logistic_max_iter)
        result["density_only_baseline"] = {"balanced_accuracy_lovo": balanced_accuracy(y, pred_d)}

    # Regularisation sensitivity: no headline number should hinge on one
    # arbitrary lambda, so report the LOVO score across the grid.
    result["l2_sensitivity"] = {
        str(l2): balanced_accuracy(
            y, cv_predict(X, y, runs, l2, opts.logistic_max_iter)[0])
        for l2 in (opts.logistic_l2_grid or [opts.logistic_l2])
    }

    # Individual-behaviour-only: drop the features that are really density in
    # disguise (nearest-neighbour distance) rather than something a larva does.
    solo = [c for c in feature_cols
            if not any(c.startswith(f) for f in DENSITY_LINKED_FEATURES)]
    if solo and len(solo) < len(feature_cols):
        Xs = _feature_matrix(video_table, solo)
        pred_s, _, _ = cv_predict(Xs, y, runs, opts.logistic_l2, opts.logistic_max_iter)
        result["individual_behaviour_only"] = {
            "n_features": len(solo),
            "dropped": [c for c in feature_cols if c not in solo],
            "balanced_accuracy_lovo": balanced_accuracy(y, pred_s),
        }

    # Density-residualised features: the sharpest test of "is this behaviour or
    # is this crowding?". Every feature is regressed on log density first, so the
    # classifier only sees what density could not explain. (The residualisation
    # uses all videos, so the covariate — not the label — is shared across folds.)
    if np.isfinite(dens).sum() >= 5:
        log_dens = np.log10(np.where(dens > 0, dens, np.nan))
        Xr = np.column_stack([residualize(X[:, j], log_dens) for j in range(X.shape[1])])
        pred_r, _, _ = cv_predict(Xr, y, runs, opts.logistic_l2, opts.logistic_max_iter)
        result["density_residualised"] = {"balanced_accuracy_lovo": balanced_accuracy(y, pred_r)}

    # Permutation null: shuffle species across videos, keep everything else.
    if opts.n_permutations > 0 and np.isfinite(video_bacc):
        null = []
        for _ in range(int(opts.n_permutations)):
            yp = rng.permutation(y)
            p, _, _ = cv_predict(X, yp, runs, opts.logistic_l2, opts.logistic_max_iter)
            null.append(balanced_accuracy(yp, p))
        null_arr = np.array([v for v in null if np.isfinite(v)])
        result["permutation_null"] = {
            "n": int(null_arr.size),
            "mean": float(null_arr.mean()) if null_arr.size else np.nan,
            "p95": float(np.quantile(null_arr, 0.95)) if null_arr.size else np.nan,
            # (#null >= observed + 1) / (n + 1): the standard conservative estimate
            "p_value": float((np.sum(null_arr >= video_bacc) + 1) / (null_arr.size + 1))
            if null_arr.size else np.nan,
        }

    if tracks is not None and not tracks.empty:
        result["track_level"] = _track_level_classification(tracks, cfg)
    return result


def _track_level_classification(tracks: pd.DataFrame, cfg: Config) -> dict:
    """Per-track classification, scored per track and by per-video vote."""
    opts = cfg.species_analysis
    cols = [c for c in SCALE_FREE_FEATURES if c in tracks.columns]
    df = tracks[["run", "meta_species", *cols]].copy()
    cap = opts.classifier_max_tracks_per_video
    if cap:
        # Cap rows per video so no single dense video dominates the fit.
        picked = [g.sample(n=min(len(g), cap), random_state=cfg.project.random_seed)
                  for _, g in df.groupby("run", sort=False)]
        df = pd.concat(picked, ignore_index=True)
    y = as_labels(df["meta_species"]).to_numpy()
    groups = as_labels(df["run"]).to_numpy()
    X = _feature_matrix(df, cols)
    pred, proba, classes = cv_predict(X, y, groups, opts.logistic_l2, opts.logistic_max_iter)

    vote = pd.DataFrame({"run": groups, "true": y})
    for i, c in enumerate(classes):
        vote[c] = proba[:, i]
    per_video = vote.groupby("run").agg({**{c: "mean" for c in classes}, "true": "first"})
    voted = per_video[list(classes)].to_numpy().argmax(axis=1)
    video_pred = classes[voted]
    video_true = per_video["true"].to_numpy()
    return {
        "n_tracks": int(len(df)),
        "balanced_accuracy_per_track_lovo": balanced_accuracy(y, pred),
        "balanced_accuracy_per_video_vote": balanced_accuracy(video_true, video_pred),
        "confusion_per_video_vote": confusion_table(video_true, video_pred).to_dict(),
    }


# --------------------------------------------------------------------------- #
# 6. Dual-container within-video contrast
# --------------------------------------------------------------------------- #
def estimate_split_x_fraction(run_dir: str | Path,
                              search_band: tuple[float, float] = (0.30, 0.70)) -> float | None:
    """Divider position (fraction of frame width) from the occupancy heatmap.

    The gap between two physically separate boxes shows up as a column of near-
    zero occupancy. This is a *proposal* to be checked against a representative
    frame, not an automatic calibration — see ``classify_compartment``.
    """
    occ_path = Path(run_dir) / "heatmaps" / "occupancy.npy"
    if not occ_path.exists():
        return None
    occ = np.load(occ_path)
    col = occ.sum(axis=0)
    n = col.size
    lo, hi = int(search_band[0] * n), int(search_band[1] * n)
    if hi - lo < 3:
        return None
    band = col[lo:hi]
    i = int(np.argmin(band))
    # The wall is a *band* of empty columns, not one. Taking the argmin alone
    # lands on its left edge and misassigns larvae hugging the near face, so
    # expand across the contiguous low-occupancy run and take its midpoint.
    thr = max(float(band[i]), 0.1 * float(np.median(band)))
    left = i
    while left > 0 and band[left - 1] <= thr:
        left -= 1
    right = i
    while right < band.size - 1 and band[right + 1] <= thr:
        right += 1
    return float((lo + (left + right) / 2.0 + 0.5) / n)


def dual_container_contrast(cohort: pd.DataFrame, cfg: Config | None = None,
                            run_dirs: dict[str, str] | None = None) -> pd.DataFrame:
    """Within-video species contrast for dual-container (SX/DX) videos.

    The only design in this dataset where two species share a recording, so
    camera, lighting, session and observer are held constant by construction.
    Requires runs re-analysed with ``regions.dual_container_split_x_fraction``
    set, i.e. with ``species_compartment`` present in ``tracks_clean``.

    Per video and feature: Mann-Whitney between compartments with a rank-biserial
    effect size, plus a within-compartment split control that shows what the same
    test yields when there is no species difference at all.
    """
    from scipy import stats

    cfg = cfg or Config()
    opts = cfg.species_analysis
    rng = np.random.default_rng(cfg.project.random_seed)
    duals = cohort[cohort.get("is_dual_container", False).fillna(False).astype(bool)]
    rows = []
    for _, row in duals.iterrows():
        run_dir = Path((run_dirs or {}).get(row["run"], row["run_dir"]))
        tracks = _run_track_features(run_dir, opts, seed=cfg.project.random_seed)
        if tracks.empty or "species_compartment" not in tracks.columns:
            continue
        tracks = tracks[tracks.get("compartment_purity", 1.0) >= 0.95]
        tracks = tracks[tracks["species_compartment"].notna()]
        species = sorted(as_labels(tracks["species_compartment"]).unique())
        if len(species) != 2:
            continue
        a, b = species
        for feat_col in [c for c in SCALE_FREE_FEATURES if c in tracks.columns]:
            xa = pd.to_numeric(tracks.loc[tracks["species_compartment"] == a, feat_col],
                               errors="coerce").dropna().to_numpy()
            xb = pd.to_numeric(tracks.loc[tracks["species_compartment"] == b, feat_col],
                               errors="coerce").dropna().to_numpy()
            if xa.size < 20 or xb.size < 20:
                continue
            U, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            # Control: same test between two random halves of one compartment.
            half = rng.permutation(xa.size)
            c1, c2 = xa[half[: xa.size // 2]], xa[half[xa.size // 2:]]
            Uc, pc = stats.mannwhitneyu(c1, c2, alternative="two-sided")
            rows.append({
                "run": row["run"], "video": row.get("video"),
                "species_a": a, "species_b": b, "feature": feat_col,
                "n_a": xa.size, "n_b": xb.size,
                "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
                "rank_biserial": float(2 * U / (xa.size * xb.size) - 1),
                "p": float(p),
                "control_rank_biserial": float(2 * Uc / (c1.size * c2.size) - 1),
                "control_p": float(pc),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = benjamini_hochberg(out["p"].to_numpy())
    out["significant"] = out["p_fdr"] < opts.fdr_alpha
    return out.sort_values(["feature", "run"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Confound audit
# --------------------------------------------------------------------------- #
def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Cramer's V and chi-square p for two categorical series."""
    from scipy import stats

    table = pd.crosstab(as_labels(a), as_labels(b))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    denom = n * (min(table.shape) - 1)
    return (float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan), float(p)


def confound_audit(video_table: pd.DataFrame,
                   group_col: str = "meta_species") -> pd.DataFrame:
    """How strongly each nuisance variable is tied to species in this cohort.

    A confound with a large association is not something the statistics can undo
    — it bounds what any per-species claim can mean. Stage is the extreme case
    here: *Anopheles stephensi* is the only stage-3 group.
    """
    from scipy import stats

    rows = []
    groups = as_labels(video_table[group_col])
    numeric = [c for c in ("density", "total_tracks", "coverage_pct",
                           "duplicate_track_proxy", "width", "height", "fps",
                           "median_track_duration_s") if c in video_table.columns]
    for col in numeric:
        vals = pd.to_numeric(video_table[col], errors="coerce")
        samples = [vals[groups == g].dropna().to_numpy() for g in sorted(groups.unique())]
        samples = [s for s in samples if s.size >= 3]
        if len(samples) < 2:
            continue
        H, p = stats.kruskal(*samples)
        n = sum(s.size for s in samples)
        rows.append({"variable": col, "kind": "numeric", "statistic": "kruskal_H",
                     "value": float(H), "p": float(p),
                     "effect": float((H - len(samples) + 1) / (n - len(samples))),
                     "effect_name": "epsilon_sq",
                     "detail": "; ".join(
                         f"{g}: {pd.to_numeric(video_table.loc[groups == g, col], errors='coerce').median():.3g}"
                         for g in sorted(groups.unique()))})
    categorical = [c for c in ("meta_recording_date", "meta_camera_model",
                               "meta_camera_family", "meta_lighting",
                               "meta_container_type", "meta_stage", "meta_resolution",
                               "meta_camera_position", "meta_water_depth_mm")
                   if c in video_table.columns]
    for col in categorical:
        v, p = cramers_v(groups, as_labels(video_table[col]))
        if not np.isfinite(v):
            continue
        rows.append({"variable": col, "kind": "categorical", "statistic": "chi2",
                     "value": np.nan, "p": p, "effect": v, "effect_name": "cramers_v",
                     "detail": "; ".join(
                         f"{g}: {'/'.join(sorted(set(as_labels(video_table.loc[groups == g, col]))))}"
                         for g in sorted(groups.unique()))})
    out = pd.DataFrame(rows)
    return out.sort_values("effect", ascending=False).reset_index(drop=True) if not out.empty else out


# --------------------------------------------------------------------------- #
# 8. Plots
# --------------------------------------------------------------------------- #
def _species_boxplot(video_table: pd.DataFrame, feature: str, out_path: Path,
                     group_col: str = "meta_species") -> Path:
    from .visualization import _save, plt

    groups = as_labels(video_table[group_col])
    labels = sorted(groups.unique())
    data = [pd.to_numeric(video_table.loc[groups == g, feature],
                          errors="coerce").dropna().to_numpy() for g in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=[f"{lab}\n(n={d.size})" for lab, d in zip(labels, data, strict=True)],
               showfliers=False)
    for i, d in enumerate(data, start=1):
        if d.size:
            ax.plot(np.full(d.size, i) + np.random.default_rng(0).normal(0, 0.05, d.size),
                    d, "o", ms=4, alpha=0.6)
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by species (one point = one video)")
    ax.tick_params(axis="x", labelsize=8)
    return _save(fig, out_path)


def _density_confound_plot(video_table: pd.DataFrame, out_path: Path) -> Path:
    from .visualization import _save, plt

    groups = as_labels(video_table["meta_species"])
    labels = sorted(groups.unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lab in enumerate(labels):
        d = pd.to_numeric(video_table.loc[groups == lab, "density"], errors="coerce").dropna()
        ax.plot(np.full(len(d), i) + np.random.default_rng(1).normal(0, 0.06, len(d)),
                d, "o", ms=6, alpha=0.7, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean active larvae per frame (log)")
    ax.set_title("Species is confounded with density — read every metric through this")
    return _save(fig, out_path)


def _permutation_plot(clf: dict, out_path: Path) -> Path | None:
    from .visualization import _save, plt

    null = clf.get("permutation_null") or {}
    obs = clf.get("video_level", {}).get("balanced_accuracy_lovo")
    if not null or obs is None or not np.isfinite(obs):
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axvline(null.get("mean", np.nan), color="grey", ls=":", label="null mean")
    ax.axvline(null.get("p95", np.nan), color="grey", ls="--", label="null 95th pct")
    ax.axvline(obs, color="crimson", lw=2, label=f"observed = {obs:.3f}")
    ax.set_xlabel("leave-one-video-out balanced accuracy")
    ax.set_yticks([])
    ax.set_title(f"Species classification vs label-permutation null "
                 f"(p = {null.get('p_value', float('nan')):.3f})")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def _confusion_plot(clf: dict, out_path: Path) -> Path | None:
    from .visualization import _save, plt

    conf = clf.get("video_level", {}).get("confusion")
    if not conf:
        return None
    table = pd.DataFrame(conf).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(table.to_numpy(dtype=float), cmap="Blues")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index, fontsize=7)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, int(table.iat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Leave-one-video-out confusion (videos)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 9. Orchestration
# --------------------------------------------------------------------------- #
SPECIES_CAVEATS = [
    "Species labels come from filename metadata, not from the detector: on this batch the "
    "detector's per-video majority class agrees with the filename label in only 45/66 videos.",
    "Larval density spans ~200x across videos and is confounded with species; every test is "
    "reported raw, density-adjusted, and on a density-matched subset. Prefer the latter two.",
    "Speeds and distances are body-length-normalised or dimensionless because source "
    "resolutions differ by 16x; raw pixel metrics are not comparable across videos.",
    "Body length is a morphometric, not a behaviour, and is resolution-dependent without "
    "calibration — it is reported separately and must not be read as a behavioural result.",
    "Leave-one-video-out with grouped labels biases accuracy *below* 1/k under the null; "
    "compare the observed score with the permutation null, not with chance.",
    "The aspect-ratio peak frequency is a texture proxy: at 30 fps with stride 2 the Nyquist "
    "limit is 7.5 Hz, so true undulation frequency is partially aliased.",
    "Median track duration in this batch is ~2.7 s; no long-horizon behaviour can be measured "
    "without ID stitching, which this pipeline deliberately does not perform.",
]


def _headline(uni: pd.DataFrame, clf: dict, n_videos: int) -> str:
    """One-paragraph verdict, phrased so that a null result reads as a result."""
    obs = clf.get("video_level", {}).get("balanced_accuracy_lovo", float("nan"))
    null = clf.get("permutation_null", {})
    p = null.get("p_value", float("nan"))
    dens = clf.get("density_only_baseline", {}).get("balanced_accuracy_lovo", float("nan"))
    matched = uni[(uni["variant"] == "density_matched") & uni["significant"]] if not uni.empty \
        else pd.DataFrame()
    adjusted = uni[(uni["variant"] == "density_adjusted") & uni["significant"]] if not uni.empty \
        else pd.DataFrame()
    loco = clf.get("video_level", {}).get("balanced_accuracy_loco", float("nan"))
    lodo = clf.get("video_level", {}).get("balanced_accuracy_lodo", float("nan"))
    lodo_p = (clf.get("leave_one_session_out") or {}).get("p_value", float("nan"))
    cam = clf.get("camera_control", {}).get("balanced_accuracy_lovo", float("nan"))
    parts = [
        f"Cohort: {n_videos} videos. Leave-one-video-out balanced accuracy "
        f"**{obs:.3f}** (permutation null p = {p:.3f}); leave-one-camera-out {loco:.3f}; "
        f"leave-one-session-out {lodo:.3f} (p = {lodo_p:.3f}); "
        f"density-only baseline {dens:.3f}; "
        f"same features predicting the camera {cam:.3f}.",
        f"{len(adjusted)} features survive FDR after adjusting for density and "
        f"{len(matched)} in the density-matched band.",
    ]
    dropped = clf.get("session_crossed_subset", {}).get("species_dropped_single_session") or []
    if dropped:
        sub = clf["session_crossed_subset"]
        parts.append(
            f"{', '.join(dropped)} was filmed only in sessions where nothing else was "
            f"filmed, so its separation cannot be told apart from the session; on the "
            f"remaining {sub['n_videos']} videos of {len(sub['species_kept'])} species the "
            f"score is {sub['balanced_accuracy_lovo']:.3f} (chance "
            f"{sub['chance_balanced_accuracy']:.3f}).")
    if np.isfinite(p) and p >= 0.05 and matched.empty:
        parts.append("**Verdict: no tracking metric separates the species once density is "
                     "controlled.** The apparent per-species differences in the aggregate "
                     "report are crowding, camera and stage effects.")
    elif np.isfinite(p) and p < 0.05 and np.isfinite(lodo_p) and lodo_p >= 0.05:
        parts.append("**Verdict: the separation does not survive session-disjoint "
                     "validation.** It is reproducible across videos and cameras but not "
                     "across recording sessions, so it cannot be separated from what changed "
                     "between sessions (larval batch, water, container setup, time of day). "
                     "Treat the per-feature differences as hypotheses, not species traits.")
    elif np.isfinite(p) and p < 0.05:
        verdict = ("**Verdict: species-linked signal is present above the permutation null "
                   "and stays above it when whole cameras and whole sessions are held out.**")
        # "Above its null" is not "unchanged": a large drop from the video-level
        # score means most of what separates the species is session-specific.
        if np.isfinite(lodo) and np.isfinite(obs) and lodo < 0.7 * obs:
            verdict += (f" But it degrades sharply across sessions ({obs:.3f} -> {lodo:.3f}), "
                        f"so most of the separation is session-specific and cannot be "
                        f"attributed to the animals alone.")
        parts.append(verdict + " It is an observational contrast, not a controlled "
                     "experiment — read it with the confound audit beside it.")
    else:
        parts.append("**Verdict: weak or partial separation** — see the per-feature tables "
                     "before making any claim.")
    return " ".join(parts)


def species_analysis(batch_dir: str | Path, out_dir: str | Path,
                     cfg: Config | None = None,
                     dual_run_dirs: dict[str, str] | None = None) -> dict:
    """Full species-discrimination analysis over a finished batch.

    Writes cohort/feature/statistics tables, plots and a markdown+HTML report,
    and returns the result dictionary.
    """
    from . import reports as rep
    from .video_io import save_table

    cfg = cfg or Config()
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    cohort = build_cohort(batch_dir, cfg)
    if cohort.empty:
        raise ValueError(f"no runs with video_summary.json under {batch_dir}")
    save_table(cohort, out / "cohort.parquet")

    tracks = extract_track_features(cohort, cfg, cache_path=out / "track_features.parquet")
    if tracks.empty:
        raise ValueError("no tracks passed the duration filter in the cohort")

    video_table = video_feature_table(tracks, cohort)
    video_path = save_table(video_table, out / "video_features.parquet")

    feature_cols = video_feature_columns(video_table)
    uni = univariate_tests(video_table, feature_cols, opts=cfg.species_analysis)
    save_table(uni, out / "univariate_tests.parquet")
    pair = pairwise_tests(video_table, feature_cols, opts=cfg.species_analysis)
    save_table(pair, out / "pairwise_tests.parquet")
    audit = confound_audit(video_table)
    save_table(audit, out / "confound_audit.parquet")

    clf = classify_species(video_table, feature_cols, tracks=tracks, cfg=cfg)
    (out / "classification_results.json").write_text(
        json.dumps(clf, indent=2, default=rep._json_default), encoding="utf-8")

    dual = dual_container_contrast(cohort, cfg, run_dirs=dual_run_dirs)
    if not dual.empty:
        save_table(dual, out / "dual_container_contrast.parquet")

    # --- plots ---
    plot_paths: list[str] = []
    top_feats = (uni[uni["variant"] == "density_adjusted"].head(3)["feature"].tolist()
                 if not uni.empty else feature_cols[:3])
    for maker in (
        lambda: _density_confound_plot(video_table, out / "plots" / "species_vs_density.png"),
        lambda: _permutation_plot(clf, out / "plots" / "permutation_null.png"),
        lambda: _confusion_plot(clf, out / "plots" / "confusion_lovo.png"),
        *[(lambda f=f: _species_boxplot(video_table, f, out / "plots" / f"{f}_by_species.png"))
          for f in top_feats],
    ):
        try:
            path = maker()
            if path is not None:
                plot_paths.append(str(Path(path).relative_to(out)))
        except Exception as exc:  # a single bad chart must not kill the report
            warnings.warn(f"plot failed: {exc}", stacklevel=2)

    # --- report ---
    counts = (video_table.groupby("meta_species")
              .agg(n_videos=("run", "size"), n_tracks=("n_tracks_used", "sum"),
                   median_density=("density", "median")).reset_index())
    excluded = cohort.loc[~cohort["in_cohort"], ["video", "excluded_reason"]]
    headline = _headline(uni, clf, len(video_table))
    clf_rows = pd.DataFrame([
        {"model": "species from track features (video level, LOVO)",
         "balanced_accuracy": clf["video_level"]["balanced_accuracy_lovo"]},
        {"model": "species from track features (video level, leave-one-camera-out)",
         "balanced_accuracy": clf["video_level"].get("balanced_accuracy_loco")},
        {"model": "species from track features (video level, leave-one-session-out)",
         "balanced_accuracy": clf["video_level"].get("balanced_accuracy_lodo")},
        {"model": "species, single-session species removed (video level, LOVO)",
         "balanced_accuracy": clf.get("session_crossed_subset", {}).get("balanced_accuracy_lovo")},
        {"model": "species per track (LOVO, per-track score)",
         "balanced_accuracy": clf.get("track_level", {}).get("balanced_accuracy_per_track_lovo")},
        {"model": "species per track aggregated to a per-video vote",
         "balanced_accuracy": clf.get("track_level", {}).get("balanced_accuracy_per_video_vote")},
        {"model": "species from individual-behaviour features only "
                  "(neighbour-distance features dropped, LOVO)",
         "balanced_accuracy":
             clf.get("individual_behaviour_only", {}).get("balanced_accuracy_lovo")},
        {"model": "species from density-residualised features (video level, LOVO)",
         "balanced_accuracy": clf.get("density_residualised", {}).get("balanced_accuracy_lovo")},
        {"model": "BASELINE: density only", "balanced_accuracy":
            clf.get("density_only_baseline", {}).get("balanced_accuracy_lovo")},
        {"model": "NULL: permuted species labels (mean)", "balanced_accuracy":
            clf.get("permutation_null", {}).get("mean")},
        {"model": "CONTROL: same features predicting camera model", "balanced_accuracy":
            clf.get("camera_control", {}).get("balanced_accuracy_lovo")},
    ])
    morph_cols = [f"{f}_med" for f in MORPHOMETRIC_FEATURES
                  if f"{f}_med" in video_table.columns]
    morph = univariate_tests(video_table, morph_cols, opts=cfg.species_analysis) \
        if morph_cols else pd.DataFrame()
    if not morph.empty:
        save_table(morph, out / "morphometric_tests.parquet")

    def _scheme_row(label, obs_key, null_key):
        null = clf.get(null_key) or {}
        return {"cross-validation scheme": label,
                "balanced_accuracy": clf["video_level"].get(obs_key),
                "null_mean": null.get("null_mean", null.get("mean")),
                "null_p95": null.get("null_p95", null.get("p95")),
                "p_value": null.get("p_value"),
                "n_held_out_groups": null.get("n_groups", clf.get("n_videos"))}

    schemes = pd.DataFrame([
        _scheme_row("leave one video out", "balanced_accuracy_lovo", "permutation_null"),
        _scheme_row("leave one camera model out", "balanced_accuracy_loco",
                    "leave_one_camera_out"),
        _scheme_row("leave one recording session out", "balanced_accuracy_lodo",
                    "leave_one_session_out"),
    ])
    chance = pd.DataFrame([
        {"reference": "chance (1/n classes)", "value": clf["chance_balanced_accuracy"]},
        {"reference": "majority-class share", "value": clf["majority_class_accuracy"]},
    ])
    sections: list[tuple[str, object]] = [
        ("Verdict", pd.DataFrame([{"summary": headline}])),
        ("Cohort", counts),
        ("Excluded runs and why", excluded),
        ("Confound audit (what species is entangled with)", audit),
        ("Classification, baselines and controls", clf_rows),
        ("Each cross-validation scheme against its own permutation null", schemes),
        ("Reference points", chance),
        ("Regularisation sensitivity (LOVO balanced accuracy per L2)",
         pd.DataFrame([{"l2": k, "balanced_accuracy": v}
                       for k, v in clf.get("l2_sensitivity", {}).items()])),
        ("Univariate tests — density-adjusted (top 25)",
         uni[uni["variant"] == "density_adjusted"].head(25)),
        ("Univariate tests — density-matched band (top 25)",
         uni[uni["variant"] == "density_matched"].head(25)),
        ("Univariate tests — raw, i.e. confounded (top 15)",
         uni[uni["variant"] == "raw"].head(15)),
        ("Pairwise species contrasts (top 25)", pair.head(25)),
        ("Morphometrics — apparent size/shape, NOT behaviour", morph),
    ]
    if not dual.empty:
        sections.append(("Dual-container within-video contrast "
                         "(same camera, lighting and session)", dual.head(40)))

    title = f"{len(video_table)} videos from {Path(batch_dir).name}"
    caveats_before = rep.CAVEATS
    try:
        rep.CAVEATS = SPECIES_CAVEATS + list(caveats_before)
        if cfg.reports.make_markdown:
            rep.write_markdown(rep.build_aggregate_markdown(title, sections, plot_paths),
                               out / "species_report.md")
        if cfg.reports.make_html:
            rep.write_html(rep.build_aggregate_html(title, sections, plot_paths),
                           out / "species_report.html")
    finally:
        rep.CAVEATS = caveats_before

    warnings_out: list[str] = []
    if cfg.reports.make_pdf and (out / "species_report.html").exists():
        try:
            rep.export_report_pdf(out / "species_report.html", out / "species_report.pdf")
        except Exception as exc:
            # Surfaced in the returned dict *and* the report folder, not only in a log.
            warnings_out.append(f"PDF report skipped: {exc}")
            (out / "warnings.json").write_text(json.dumps(warnings_out, indent=2),
                                               encoding="utf-8")

    if cfg.fair.enabled:
        from . import fair
        fair.finalize_fair_file(video_path, cfg,
                                inputs=[{"role": "batch-dir", "path": str(batch_dir)}])

    return {"cohort": cohort, "tracks": tracks, "video_table": video_table,
            "univariate": uni, "pairwise": pair, "confound_audit": audit,
            "classification": clf, "dual_container": dual,
            "headline": headline, "warnings": warnings_out, "out_dir": str(out)}
