"""Figure 6 - fixed-detector multi-tracker benchmark.

    (a) per-arm track quality, mean over videos with bootstrap CI
    (b) cross-arm identity agreement vs crowding
    (c) fragmentation vs crowding
    (d) median track duration vs crowding

The crowding axis is the **shared detection density**: detections per frame from
the one detector pass every arm replays. It is therefore identical for all arms,
so an arm that fragments more cannot appear to be facing a denser video.

Nothing plotted here is a MOT metric. There is no identity ground truth, so
panels (b)-(d) show how arms *differ*, not which is correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs  # noqa: E402

FIG = "fig06_tracker_benchmark"

# The arrow states which direction is "more fragmented", because panel (a)
# normalises each metric onto a shared axis and a reader cannot otherwise tell that
# a tall bar for fragments means worse and a tall bar for duration means longer.
PANEL_A_METRICS = [
    ("median_track_duration_s", "Median track\nduration (s)"),
    ("fragments_per_object", "Fragments\nper object $\\downarrow$"),
    ("identity_agreement_mean", "Mean ARI vs\nother arms"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = fs.load_config(args.config)
    paths = fs.resolve_paths(cfg)
    opts = cfg.get("fig06") or {}
    fs.apply_style()

    sidecar = fs.Sidecar(figure=FIG,
                         description="Fixed-detector multi-tracker benchmark",
                         hash_max_bytes=cfg.get("hash_max_bytes"))

    eval_dir = paths.tracker_benchmark / "evaluation"
    metrics_path = sidecar.add_input(eval_dir / "metrics_per_video.parquet",
                                     "per-video benchmark metrics")
    summary_path = sidecar.add_input(eval_dir / "summary_by_arm.parquet",
                                     "per-arm bootstrap summary")
    metrics = pd.read_parquet(metrics_path)
    summary = pd.read_parquet(summary_path)

    arms = fs.order_trackers(metrics["arm"])
    n_videos = int(metrics["video_id"].nunique())
    min_trend = int(opts.get("min_videos_for_trend", 8))
    sidecar.params.update({"n_arms": len(arms), "n_videos": n_videos,
                           "min_videos_for_trend": min_trend,
                           "density_axis": "shared detections/frame (identical across arms)"})
    sidecar.notes.append(
        "No identity ground truth: no HOTA/DetA/AssA/IDF1 anywhere in this figure. "
        "Panels (b)-(d) show how arms differ, not which is correct.")

    panels = {
        "a": lambda ax: _panel_a(ax, summary, arms),
        "b": lambda ax: _scatter(ax, metrics, arms,
                                 opts.get("density_metric_b", "identity_agreement_mean"),
                                 "Mean ARI vs other arms", min_trend),
        "c": lambda ax: _scatter(ax, metrics, arms,
                                 opts.get("density_metric_c", "fragments_per_object"),
                                 "Fragments per object", min_trend),
        "d": lambda ax: _scatter(ax, metrics, arms,
                                 opts.get("density_metric_d", "median_track_duration_s"),
                                 "Median track duration (s)", min_trend),
    }

    fig, axes = fs.plt.subplots(2, 2, figsize=(7.2, 5.6))
    for key, ax in zip("abcd", axes.ravel(), strict=True):
        panels[key](ax)
        fs.panel_label(ax, key)
    _legend(fig, arms)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    table = _table(metrics, summary, arms)
    fs.save_figure(fig, FIG, sidecar, table, paths.output_dir, panels=panels)
    fs.report_checks(sidecar)
    return 0


# --------------------------------------------------------------------------- #
def _panel_a(ax, summary: pd.DataFrame, arms: list[str]) -> None:
    """Grouped point plot with bootstrap CIs, one column per metric."""
    present = [(m, lab) for m, lab in PANEL_A_METRICS if f"{m}_mean" in summary.columns]
    if not present:
        fs.hide_axes(ax)
        return
    idx = {a: i for i, a in enumerate(arms)}
    # Spacing between arms WITHIN a metric group. Dividing by the number of
    # metrics instead would let a group's markers spill past its own half-width
    # (+-0.5) and land under a neighbouring metric's label.
    width = 0.8 / max(len(arms), 1)
    for j, (metric, _label) in enumerate(present):
        for arm in arms:
            row = summary[summary["arm"] == arm]
            if row.empty or not np.isfinite(row.iloc[0].get(f"{metric}_mean", np.nan)):
                continue
            r = row.iloc[0]
            # Each metric is on its own scale, so plot the value relative to the
            # arm mean for that metric. A shared absolute axis would be a
            # dual-axis chart in disguise.
            scale = summary[f"{metric}_mean"].abs().max() or 1.0
            x = j + (idx[arm] - (len(arms) - 1) / 2) * width
            st = fs.tracker_style(arm)
            ax.plot([x, x], [r[f"{metric}_lo"] / scale, r[f"{metric}_hi"] / scale],
                    color=st["color"], lw=1.2, solid_capstyle="butt", zorder=2)
            ax.plot([x], [r[f"{metric}_mean"] / scale], linestyle="none",
                    markersize=4.5, zorder=3, **st)
    for j in range(1, len(present)):
        ax.axvline(j - 0.5, color=fs.NEUTRAL["grid"], lw=0.6, zorder=0)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([lab for _, lab in present], fontsize=7)
    ax.set_xlim(-0.5, len(present) - 0.5)
    ax.set_ylabel("Value / max across arms")
    ax.set_title("Track quality by arm", pad=6)
    ax.set_ylim(bottom=0)
    ax.axhline(0, color=fs.NEUTRAL["faint"], lw=0.6, zorder=0)
    # $\downarrow$ marks a metric where a larger value means worse association.


def _scatter(ax, metrics: pd.DataFrame, arms: list[str], metric: str,
             ylabel: str, min_trend: int) -> None:
    """One point per video per arm, against the shared detection density."""
    if metric not in metrics.columns:
        fs.hide_axes(ax)
        return
    n_videos = metrics["video_id"].nunique()
    for arm in arms:
        g = metrics[metrics["arm"] == arm]
        x = pd.to_numeric(g["detections_per_frame"], errors="coerce")
        y = pd.to_numeric(g[metric], errors="coerce")
        m = x.notna() & y.notna() & (x > 0)
        if not m.any():
            continue
        st = fs.tracker_style(arm)
        ax.plot(x[m], y[m], linestyle="none", markersize=4.5, alpha=0.85,
                zorder=3, **st)
        # A trend line on a handful of videos would assert a relationship the
        # data cannot support, so it is drawn only once there are enough.
        if m.sum() >= min_trend:
            lx = np.log10(x[m].to_numpy())
            order = np.argsort(lx)
            coef = np.polyfit(lx, y[m].to_numpy(), 1)
            ax.plot(10 ** lx[order], np.polyval(coef, lx[order]),
                    color=st["color"], lw=1.0, alpha=0.9, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Shared detections / frame")
    ax.set_ylabel(ylabel)
    if n_videos < min_trend:
        ax.set_title(f"{n_videos} video{'' if n_videos == 1 else 's'} — no trend fitted",
                     fontsize=7, color=fs.NEUTRAL["mid"], pad=4)


def _legend(fig, arms: list[str]) -> None:
    """Identity is carried by colour AND marker, and always named."""
    handles = [fs.plt.Line2D([], [], linestyle="none", markersize=5,
                             label=fs.tracker_label(a), **fs.tracker_style(a))
               for a in arms]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(arms), 5),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))


def _table(metrics: pd.DataFrame, summary: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Exactly the values plotted, long-form."""
    keep = ["arm", "video_id", "detections_per_frame", "identity_agreement_mean",
            "fragments_per_object", "median_track_duration_s"]
    long = metrics[[c for c in keep if c in metrics.columns]].copy()
    long["panel"] = "b,c,d"
    wide = summary.copy()
    wide["panel"] = "a"
    return pd.concat([long, wide], ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
