"""Roll the benchmark up into tables and a report.

Everything reported here is derived from written artifacts, never from a value
carried in code, so a stale number cannot survive a rerun.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..video_io import save_table
from .analyze_density import (
    collect,
    density_bins,
    density_trend,
    paired_arm_comparison,
    summarise_by_arm,
)

#: Headline metrics for the primary table (§15), in report order.
PRIMARY_METRICS = [
    "median_track_duration_s",
    "p95_track_duration_s",
    "fragments_per_object",
    "track_birth_rate_per_object_per_min",
    "fraction_short_tracks",
    "interpolation_fraction",
    "large_jump_fraction",
    "duplicate_track_proxy",
    "duplicate_id_proxy",
]

#: Metrics whose density dependence is the crowding result (§12).
DENSITY_METRICS = [
    "fragments_per_object",
    "median_track_duration_s",
    "duplicate_id_proxy",
    "identity_agreement_mean",
]


def agreement_table(benchmark_dir: str | Path) -> pd.DataFrame:
    """Every per-video pairwise identity agreement, stacked."""
    from ..video_io import load_table

    d = Path(benchmark_dir) / "_agreement"
    frames = []
    for p in sorted(d.glob("*.parquet")):
        t = load_table(p)
        if t is None or t.empty:
            continue
        t = t.copy()
        t["video_id"] = p.stem
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_arm_agreement(agreement: pd.DataFrame) -> pd.DataFrame:
    """Each arm's mean agreement with all the others, per video.

    An arm that disagrees with everything is the one whose association decisions
    are most distinctive. That is not the same as being wrong -- with no ground
    truth, this ranks arms by how much they differ, not by quality.
    """
    if agreement.empty:
        return pd.DataFrame()
    long = pd.concat([
        agreement.rename(columns={"arm_a": "arm", "arm_b": "other"}),
        agreement.rename(columns={"arm_b": "arm", "arm_a": "other"}),
    ], ignore_index=True)
    return (long.groupby(["video_id", "arm"])
            .agg(identity_agreement_mean=("ari", "mean"),
                 identity_f1_mean=("identity_f1", "mean"),
                 n_pairs=("ari", "size"))
            .reset_index())


def evaluate(benchmark_dir: str | Path, detections_dir: str | Path,
             out_dir: str | Path, reference_arm: str = "botsort",
             n_boot: int = 2000, seed: int = 42) -> dict:
    """Build every benchmark table and return the summary dict."""
    out = Path(out_dir)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)

    metrics = collect(benchmark_dir, detections_dir)
    if metrics.empty:
        raise ValueError(f"no arm run folders under {benchmark_dir}")

    agree = agreement_table(benchmark_dir)
    if not agree.empty:
        metrics = metrics.merge(per_arm_agreement(agree), on=["video_id", "arm"], how="left")
    metrics = density_bins(metrics)
    save_table(metrics, out / "evaluation" / "metrics_per_video.parquet")
    save_table(agree, out / "evaluation" / "identity_agreement_per_video.parquet")

    available = [m for m in PRIMARY_METRICS + ["identity_agreement_mean"]
                 if m in metrics.columns]
    overall = summarise_by_arm(metrics, available, n_boot=n_boot, seed=seed)
    by_bin = summarise_by_arm(metrics, available, by_bin=True, n_boot=n_boot, seed=seed)
    save_table(overall, out / "evaluation" / "summary_by_arm.parquet")
    save_table(by_bin, out / "evaluation" / "summary_by_arm_and_density_bin.parquet")

    paired = pd.concat(
        [paired_arm_comparison(metrics, m, reference=reference_arm, n_boot=n_boot, seed=seed)
         for m in available if metrics[m].notna().any()],
        ignore_index=True) if available else pd.DataFrame()
    save_table(paired, out / "evaluation" / "paired_arm_comparison.parquet")

    trends = pd.concat(
        [density_trend(metrics, m) for m in DENSITY_METRICS if m in metrics.columns],
        ignore_index=True)
    save_table(trends, out / "evaluation" / "density_trends.parquet")

    proxy = proxy_coherence(metrics)
    save_table(proxy, out / "evaluation" / "proxy_coherence.parquet")

    summary = {
        "n_videos": int(metrics["video_id"].nunique()),
        "n_arms": int(metrics["arm"].nunique()),
        "arms": sorted(metrics["arm"].unique().tolist()),
        "reference_arm": reference_arm,
        "density_range_detections_per_frame": [
            float(metrics["detections_per_frame"].min()),
            float(metrics["detections_per_frame"].max()),
        ],
        "metrics_reported": available,
    }
    (out / "evaluation" / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "metrics": metrics, "overall": overall,
            "by_bin": by_bin, "paired": paired, "trends": trends,
            "agreement": agree, "proxy": proxy}


def proxy_coherence(metrics: pd.DataFrame) -> pd.DataFrame:
    """How the duplicate-track proxies relate to the other failure signals (§17).

    This is a coherence check, NOT the validation §17 asks for. Without ground
    truth there is no measured association error to correlate against, so a
    strong correlation here shows the proxies move with fragmentation and
    crowding -- not that either predicts a real identity failure. That question
    stays open until identity annotation exists.
    """
    from scipy import stats

    targets = ["fragments_per_object", "median_track_duration_s",
               "identity_agreement_mean", "detections_per_frame",
               "large_jump_fraction"]
    rows = []
    for proxy in ("duplicate_track_proxy", "duplicate_id_proxy"):
        if proxy not in metrics:
            continue
        for arm, g in metrics.groupby("arm"):
            for t in targets:
                if t not in g:
                    continue
                x = pd.to_numeric(g[proxy], errors="coerce")
                y = pd.to_numeric(g[t], errors="coerce")
                m = x.notna() & y.notna()
                if m.sum() < 4 or x[m].nunique() < 2 or y[m].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(x[m], y[m])
                rows.append({"arm": arm, "proxy": proxy, "target": t,
                             "n_videos": int(m.sum()),
                             "spearman_rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows)
