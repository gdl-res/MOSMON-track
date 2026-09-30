"""Per-arm track quality, and how it moves with crowding (§12, §13, §14).

Without ground truth there is no AssA, so nothing here is a measure of
correctness. What these metrics do have is a property the GT-based ones would
not: because every arm consumed an identical detection pool, the density axis is
tracker-independent and every difference between arms is association.

Density is therefore taken from the **shared detections**, never from an arm's
own track count -- an arm that fragments more would otherwise appear to be
looking at a denser video, and the crowding trend would be partly self-inflicted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_utils import find_arm_dirs

#: Metrics for which a larger value means worse association.
LOWER_IS_BETTER = {
    "fragments_per_object", "track_birth_rate_per_object_per_min",
    "duplicate_track_proxy", "duplicate_id_proxy", "interpolation_fraction",
    "large_jump_fraction", "fraction_short_tracks",
}


def shared_density(detections_meta: str | Path) -> dict:
    """Tracker-independent density for one video, from the detection cache."""
    meta = json.loads(Path(detections_meta).read_text())
    n_det = meta.get("n_detections") or 0
    n_frames = meta.get("n_frames") or 0
    vid = meta.get("video", {})
    fps = vid.get("fps") or np.nan
    stride = meta.get("detector", {}).get("frame_stride", 1) or 1
    dt = stride / fps if fps and np.isfinite(fps) else np.nan
    return {
        "video_id": meta.get("video_id"),
        "detections_per_frame": (n_det / n_frames) if n_frames else np.nan,
        "n_detections": int(n_det),
        "n_frames": int(n_frames),
        "processed_fps": (1.0 / dt) if dt and np.isfinite(dt) else np.nan,
        "duration_min": (n_frames * dt / 60.0) if np.isfinite(dt) else np.nan,
        "width": vid.get("width"),
        "height": vid.get("height"),
    }


def arm_metrics(arm_dir: str | Path, density: dict) -> dict:
    """Track-quality metrics for one (arm, video) run folder."""
    arm_dir = Path(arm_dir)
    vs = _load_json(arm_dir / "video_summary.json")
    qc = _load_json(arm_dir / "qc_report.json")
    meta = _load_json(arm_dir / "arm_metadata.json")
    beh = vs.get("behaviour", {}) if vs else {}
    qcd = (vs.get("qc") if vs else {}) or qc or {}

    n_raw = qcd.get("n_unique_track_ids_raw")
    n_clean = qcd.get("n_unique_track_ids_clean")
    dpf = density.get("detections_per_frame", np.nan)
    dur_min = density.get("duration_min", np.nan)

    out = {
        "arm": meta.get("arm") or arm_dir.parent.name,
        "video_id": density.get("video_id") or arm_dir.name,
        # --- the tracker-independent crowding axis ---
        "detections_per_frame": dpf,
        "n_detections": density.get("n_detections"),
        "n_frames": density.get("n_frames"),
        "duration_min": dur_min,
        # --- what the arm produced ---
        "n_tracks_raw": n_raw,
        "n_tracks_clean": n_clean,
        "median_track_duration_s": beh.get("median_track_duration_s"),
        "mean_track_duration_s": beh.get("mean_track_duration_s"),
        "median_track_length_frames": qcd.get("median_track_length"),
        "fraction_short_tracks": qcd.get("fraction_very_short_tracks"),
        "interpolation_fraction": qcd.get("interpolation_fraction"),
        "large_jump_fraction": qcd.get("large_jump_fraction"),
        "mean_gaps_per_track": qcd.get("mean_gaps_per_track"),
        "duplicate_track_proxy": qcd.get("duplicate_track_proxy"),
        "duplicate_id_proxy": qcd.get("duplicate_id_proxy"),
        "mean_active_tracks_per_frame": beh.get("mean_active_tracks_per_frame"),
    }

    # --- normalised association-failure proxies -------------------------- #
    # Raw counts cannot be compared across clips of different length and
    # density (§12), so everything below is expressed per individual and per
    # unit time, using the shared detection density as the population estimate.
    if n_raw and np.isfinite(dpf) and dpf > 0:
        out["fragments_per_object"] = n_raw / dpf
        if np.isfinite(dur_min) and dur_min > 0:
            out["track_birth_rate_per_object_per_min"] = n_raw / dpf / dur_min
    p95 = _p95_duration(arm_dir)
    out["p95_track_duration_s"] = p95.get("p95")
    out["n_tracks_ge_3s"] = p95.get("n_ge_3s")

    timing = (meta.get("timing") or {}) if meta else {}
    out["association_seconds"] = timing.get("association_seconds")
    out["association_fps"] = timing.get("association_fps")
    out["postprocess_seconds"] = timing.get("postprocess_seconds")
    return out


def _p95_duration(arm_dir: Path) -> dict:
    from ..video_io import load_table

    ts = load_table(arm_dir / "track_summary.parquet")
    if ts is None or ts.empty or "duration_s" not in ts:
        return {}
    d = pd.to_numeric(ts["duration_s"], errors="coerce").dropna()
    if d.empty:
        return {}
    return {"p95": float(np.percentile(d, 95)), "n_ge_3s": int((d >= 3.0).sum())}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def collect(benchmark_dir: str | Path, detections_dir: str | Path) -> pd.DataFrame:
    """One row per (arm, video) across the whole benchmark."""
    benchmark_dir, detections_dir = Path(benchmark_dir), Path(detections_dir)
    densities = {}
    for m in sorted(detections_dir.glob("*.meta.json")):
        d = shared_density(m)
        densities[d["video_id"]] = d

    rows = []
    for arm_dir in find_arm_dirs(benchmark_dir):
        for run in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            dens = densities.get(run.name, {"video_id": run.name})
            rows.append(arm_metrics(run, dens))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# density bins and paired statistics
# --------------------------------------------------------------------------- #
def density_bins(table: pd.DataFrame, n_bins: int = 4,
                 labels: list[str] | None = None) -> pd.DataFrame:
    """Assign each video a density bin from quantiles of the SHARED density.

    Bin edges come from the detection pool, which is identical for every arm, so
    a video lands in the same bin whichever arm is being scored. Deriving bins
    from tracker output instead would let the arm under test choose its own
    difficulty stratification (§8: bins must not be defined after inspecting
    tracker performance).
    """
    labels = labels or ["Low", "Medium", "High", "Very high"][:n_bins]
    per_video = (table[["video_id", "detections_per_frame"]]
                 .drop_duplicates("video_id").dropna())
    if per_video.empty:
        return table.assign(density_bin=pd.NA)
    n_bins = min(n_bins, per_video["video_id"].nunique())
    binned = pd.qcut(per_video["detections_per_frame"], q=n_bins,
                     labels=labels[:n_bins], duplicates="drop")
    mapping = dict(zip(per_video["video_id"], binned, strict=True))
    return table.assign(density_bin=table["video_id"].map(mapping))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over videos (the analysis unit)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    if v.size == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def summarise_by_arm(table: pd.DataFrame, metrics: list[str],
                     by_bin: bool = False, n_boot: int = 2000,
                     seed: int = 42) -> pd.DataFrame:
    """Mean +- bootstrap CI per arm, optionally split by density bin."""
    keys = ["arm", "density_bin"] if by_bin else ["arm"]
    rows = []
    for key, g in table.groupby(keys, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(keys, key, strict=False))
        base["n_videos"] = int(g["video_id"].nunique())
        for m in metrics:
            if m not in g:
                continue
            mean, lo, hi = bootstrap_ci(g[m].to_numpy(), n_boot=n_boot, seed=seed)
            base[f"{m}_mean"] = mean
            base[f"{m}_lo"] = lo
            base[f"{m}_hi"] = hi
            base[f"{m}_median"] = float(pd.to_numeric(g[m], errors="coerce").median())
        rows.append(base)
    return pd.DataFrame(rows)


def paired_arm_comparison(table: pd.DataFrame, metric: str,
                          reference: str | None = None,
                          n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Paired differences between arms, video by video (§14).

    Every arm sees every video, so the comparison is paired and the video is the
    replicate. Frames are never treated as independent.
    """
    from scipy import stats

    wide = table.pivot_table(index="video_id", columns="arm", values=metric,
                             aggfunc="first")
    arms = sorted(wide.columns)
    pairs = ([(reference, b) for b in arms if b != reference]
             if reference else
             [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]])
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        d = (wide[b] - wide[a]).dropna()
        if d.empty:
            continue
        draws = rng.choice(d.to_numpy(), size=(n_boot, d.size), replace=True).mean(axis=1)
        try:
            _stat, p = stats.wilcoxon(d) if d.size >= 6 and d.nunique() > 1 else (np.nan, np.nan)
        except ValueError:
            p = np.nan
        # Matched-pairs rank-biserial: the effect size, which is what §14 asks
        # to emphasise over the p-value.
        pos = float((d > 0).sum())
        neg = float((d < 0).sum())
        rows.append({
            "metric": metric, "arm_a": a, "arm_b": b, "n_videos": int(d.size),
            "mean_diff": float(d.mean()), "median_diff": float(d.median()),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
            "rank_biserial": float((pos - neg) / (pos + neg)) if (pos + neg) else np.nan,
            "wilcoxon_p": float(p) if np.isfinite(p) else np.nan,
            "lower_is_better": metric in LOWER_IS_BETTER,
        })
    out = pd.DataFrame(rows)
    if not out.empty and out["wilcoxon_p"].notna().any():
        from ..species_analysis import benjamini_hochberg

        q = np.full(len(out), np.nan)
        ok = out["wilcoxon_p"].notna().to_numpy()
        q[ok] = benjamini_hochberg(out.loc[ok, "wilcoxon_p"].to_numpy())
        out["q_value"] = q
    return out


def density_trend(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-arm slope of a metric against log10 density, with Spearman rho.

    log10 because the corpus spans a 200-fold density range; a linear fit would
    be dominated by the few densest videos.
    """
    from scipy import stats

    rows = []
    for arm, g in table.groupby("arm"):
        x = pd.to_numeric(g["detections_per_frame"], errors="coerce")
        y = pd.to_numeric(g.get(metric), errors="coerce")
        m = x.notna() & y.notna() & (x > 0)
        if m.sum() < 3:
            continue
        lx = np.log10(x[m].to_numpy())
        yy = y[m].to_numpy()
        slope, intercept, r, p, se = stats.linregress(lx, yy)
        rho, rho_p = stats.spearmanr(lx, yy)
        rows.append({"arm": arm, "metric": metric, "n_videos": int(m.sum()),
                     "slope_per_log10_density": float(slope),
                     "intercept": float(intercept), "r": float(r),
                     "slope_p": float(p), "slope_se": float(se),
                     "spearman_rho": float(rho), "spearman_p": float(rho_p)})
    return pd.DataFrame(rows)
