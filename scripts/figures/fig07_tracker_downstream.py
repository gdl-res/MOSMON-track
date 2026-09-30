"""Figure 7 - does the association algorithm reach the biology?

    (a) video-level descriptor stability across arms
    (b) species classification balanced accuracy by arm and validation scheme
    (c) track-level vs video-level accuracy by arm
    (d) cohort size: common vs tracker-specific

This is the figure the manuscript's claim rests on. Tracking metrics say the
arms differ; this says whether that difference survives into the population
descriptors and the species classification built on them.

Panel (a) uses BoT-SORT as the anchor only because it is the published
configuration. It is a reference point, **not** ground truth: a descriptor that
agrees across arms is reproducible, not necessarily correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs  # noqa: E402

FIG = "fig07_tracker_downstream"

# Pre-wrapped: these labels are long and hyphenated, so splitting on spaces
# leaves them overlapping under a four-group axis.
SCHEME_LABELS = {
    "video_lovo_ba": "Leave-one-\nvideo-out",
    "camera_out_ba": "Leave-one-\ncamera-out",
    "session_out_ba": "Leave-one-\nsession-out",
    "track_level_ba": "Individual\ntrack",
    "track_vote_ba": "Track vote\nper video",
    "density_only_ba": "Density-only\nbaseline",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = fs.load_config(args.config)
    paths = fs.resolve_paths(cfg)
    opts = cfg.get("fig07") or {}
    fs.apply_style()

    sidecar = fs.Sidecar(figure=FIG,
                         description="Downstream effect of tracker choice",
                         hash_max_bytes=cfg.get("hash_max_bytes"))
    down = paths.tracker_benchmark / "downstream"
    stability = pd.read_parquet(sidecar.add_input(
        down / "descriptor_stability.parquet", "descriptor stability"))
    clf = pd.read_parquet(sidecar.add_input(
        down / "classification_by_arm.parquet", "per-arm classification"))
    cohorts = pd.read_parquet(sidecar.add_input(
        down / "cohort_summary.parquet", "per-arm cohorts"))

    arms = fs.order_trackers(clf["arm"]) if not clf.empty else []
    schemes = [s for s in opts.get("schemes", list(SCHEME_LABELS))
               if s in clf.columns]
    top = int(opts.get("top_features", 12))
    sidecar.params.update({"n_arms": len(arms), "top_features": top,
                           "schemes": schemes})
    sidecar.notes.append(
        "BoT-SORT is the anchor because it is the published configuration, not "
        "because it is correct: with no identity ground truth, agreement across "
        "arms means reproducible, not right.")

    panels = {
        "a": lambda ax: _panel_stability(ax, stability, top),
        "b": lambda ax: _panel_schemes(ax, clf, arms, schemes),
        "c": lambda ax: _panel_levels(ax, clf, arms),
        "d": lambda ax: _panel_cohorts(ax, cohorts, arms),
    }
    fig, axes = fs.plt.subplots(2, 2, figsize=(7.2, 6.0))
    for key, ax in zip("abcd", axes.ravel(), strict=True):
        panels[key](ax)
        fs.panel_label(ax, key)
    _legend(fig, arms)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    table = pd.concat([
        stability.assign(panel="a"),
        clf.assign(panel="b,c"),
        cohorts.assign(panel="d"),
    ], ignore_index=True)
    fs.save_figure(fig, FIG, sidecar, table, paths.output_dir, panels=panels)
    fs.report_checks(sidecar)
    return 0


# --------------------------------------------------------------------------- #
def _panel_stability(ax, stability: pd.DataFrame, top: int) -> None:
    """Mean cross-arm Spearman per descriptor, most stable at the top."""
    if stability.empty:
        fs.hide_axes(ax)
        return
    d = stability.dropna(subset=["spearman_mean"]).head(top).iloc[::-1]
    y = np.arange(len(d))
    # One series, so no legend: neutral ink carries it and the axis names it.
    ax.barh(y, d["spearman_mean"], height=0.62, color=fs.NEUTRAL["accent"],
            edgecolor="none", zorder=2)
    ax.plot(d["spearman_min"], y, linestyle="none", marker="|", markersize=6,
            color=fs.NEUTRAL["ink"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([str(f)[:28] for f in d["feature"]], fontsize=6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Cross-arm Spearman (bar = mean, tick = worst pair)")
    ax.set_title("Descriptor stability across arms", pad=6)
    for cut, style in ((0.95, "-"), (0.80, ":")):
        ax.axvline(cut, color=fs.NEUTRAL["light"], lw=0.6, ls=style, zorder=1)


def _panel_schemes(ax, clf: pd.DataFrame, arms: list[str], schemes: list[str]) -> None:
    """Balanced accuracy per arm under each group-disjoint scheme."""
    if clf.empty or not schemes:
        fs.hide_axes(ax)
        return
    idx = {a: i for i, a in enumerate(arms)}
    width = 0.8 / max(len(arms), 1)
    for j, scheme in enumerate(schemes):
        for arm in arms:
            row = clf[clf["arm"] == arm]
            if row.empty or not np.isfinite(row.iloc[0].get(scheme, np.nan)):
                continue
            x = j + (idx[arm] - (len(arms) - 1) / 2) * width
            ax.plot([x], [row.iloc[0][scheme]], linestyle="none", markersize=4.5,
                    zorder=3, **fs.tracker_style(arm))
        if j:
            ax.axvline(j - 0.5, color=fs.NEUTRAL["grid"], lw=0.6, zorder=0)
    chance = pd.to_numeric(clf.get("chance"), errors="coerce").dropna()
    if not chance.empty:
        ax.axhline(float(chance.iloc[0]), color=fs.NEUTRAL["mid"], lw=0.8, ls="--",
                   zorder=1)
        ax.text(len(schemes) - 0.5, float(chance.iloc[0]), " chance", va="center",
                fontsize=6, color=fs.NEUTRAL["mid"])
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([SCHEME_LABELS.get(s, s) for s in schemes], fontsize=6)
    ax.set_xlim(-0.5, len(schemes) - 0.5)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Species classification by arm", pad=6)


def _panel_levels(ax, clf: pd.DataFrame, arms: list[str]) -> None:
    """Video-level vs track-level accuracy: is population inference the robust one?"""
    if clf.empty or "track_level_ba" not in clf or "video_lovo_ba" not in clf:
        fs.hide_axes(ax)
        return
    for arm in arms:
        row = clf[clf["arm"] == arm]
        if row.empty:
            continue
        x = row.iloc[0].get("track_level_ba")
        y = row.iloc[0].get("video_lovo_ba")
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.plot([x], [y], linestyle="none", markersize=5, zorder=3,
                **fs.tracker_style(arm))
    lims = [0, 1]
    ax.plot(lims, lims, color=fs.NEUTRAL["light"], lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Individual-track balanced accuracy")
    ax.set_ylabel("Video-level balanced accuracy")
    ax.set_title("Population vs individual inference", pad=6)
    ax.text(0.97, 0.05, "above the diagonal =\npopulation inference\nis more robust",
            transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            color=fs.NEUTRAL["mid"])


def _panel_cohorts(ax, cohorts: pd.DataFrame, arms: list[str]) -> None:
    """Cohort sizes: tracker choice can move a video across the QC line."""
    if cohorts.empty:
        fs.hide_axes(ax)
        return
    present = [a for a in arms if a in set(cohorts["arm"])]
    x = np.arange(len(present))
    own = [float(cohorts.loc[cohorts["arm"] == a, "n_in_cohort"].iloc[0]) for a in present]
    common = [float(cohorts.loc[cohorts["arm"] == a, "n_in_common_cohort"].iloc[0])
              for a in present]
    ax.bar(x, own, width=0.62, color=fs.NEUTRAL["faint"],
           edgecolor=fs.NEUTRAL["light"], linewidth=0.5,
           label="Tracker-specific cohort", zorder=2)
    ax.bar(x, common, width=0.62, color=fs.NEUTRAL["accent"], edgecolor="none",
           label="Common cohort", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([fs.tracker_label(a).replace(" ", "\n") for a in present],
                       fontsize=6)
    ax.set_ylabel("Videos passing QC")
    ax.set_title("Cohort size by arm", pad=6)
    # Headroom so the legend never sits on top of a bar.
    ax.set_ylim(0, max(own + [1]) * 1.35)
    ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2,
              handlelength=1.2, columnspacing=1.0)


def _legend(fig, arms: list[str]) -> None:
    if not arms:
        return
    handles = [fs.plt.Line2D([], [], linestyle="none", markersize=5,
                             label=fs.tracker_label(a), **fs.tracker_style(a))
               for a in arms]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(arms), 5),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
