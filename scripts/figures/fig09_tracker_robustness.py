#!/usr/bin/env python3
"""Tracker robustness: the arms associate differently, the population result does not move.

    (a) per-video cross-tracker identity agreement -- how much the three arms
        actually disagree about which detections belong to the same larva
    (b) species classification balanced accuracy per arm, own cohort and the
        common cohort, against chance and the density-only baseline

Panel (a) sets up panel (b): the arms are *not* interchangeable at the level of
individual temporal association, which is what makes the downstream agreement in
(b) worth reporting. Neither panel is an accuracy measurement -- there is no
identity-resolved MOT ground truth, so agreement means two trackers concur, not
that either is right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG = "mosmon_track_tracker_robustness"
ARMS = ["bytetrack", "ocsort", "botsort"]
LABEL = {"bytetrack": "ByteTrack", "ocsort": "OC-SORT", "botsort": "BoT-SORT"}

# Tracker hues follow figstyle's tracker palette, with ONE substitution: OC-SORT
# moves from #e87ba4 to #a01f5c. Checked in OKLab under Vienot dichromat
# simulation against a white surface, all pairs:
#   ByteTrack vs OC-SORT   #eda100/#e87ba4 -> normal 19.6, worst-CVD  4.9  FAIL
#   ByteTrack vs OC-SORT   #eda100/#a01f5c -> normal 35.7, worst-CVD 27.5  PASS
#   ByteTrack vs BoT-SORT                  -> normal 30.3, worst-CVD 15.6  PASS
#   OC-SORT   vs BoT-SORT  #a01f5c/#008300 -> normal 33.8, worst-CVD 11.0  PASS
# The original pink sits below even the 6.0 CVD floor against the orange, which
# secondary encoding does not excuse. The replacement keeps the same hue family
# and lifts contrast from 2.7:1 to 7.4:1. Marker shape still differs per arm, so
# identity is never colour alone.
COLOR = {"bytetrack": "#eda100", "ocsort": "#a01f5c", "botsort": "#008300"}
MARKER = {"bytetrack": "s", "ocsort": "^", "botsort": "D"}
MS = 5.0
EDGE, EDGE_W = fs.NEUTRAL["ink"], 0.5


def _load(bench: Path, results: Path):
    agree = pd.read_parquet(bench / "evaluation" / "identity_agreement_per_video.parquet")
    main = pd.read_csv(results / "tracker_robustness_results.csv")
    prov = json.loads((results / "provenance.json").read_text())
    stab = pd.read_csv(results / "tracker_representation_agreement.csv")
    return agree, main, prov, stab


def draw_panel_a(ax, agree: pd.DataFrame) -> None:
    """Per-video ARI for each arm pair: jittered points plus the median."""
    pairs = [("botsort", "bytetrack"), ("botsort", "ocsort"), ("bytetrack", "ocsort")]
    rng = np.random.default_rng(0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=fs.NEUTRAL["grid"], linewidth=0.5)

    for i, (a, b) in enumerate(pairs):
        v = agree[(agree.arm_a == a) & (agree.arm_b == b)]["ari"].to_numpy()
        x = i + rng.uniform(-0.13, 0.13, v.size)
        ax.plot(x, v, linestyle="none", marker="o", markersize=2.4,
                color=fs.NEUTRAL["light"], markeredgewidth=0, zorder=2)
        med = float(np.median(v))
        ax.plot([i - 0.28, i + 0.28], [med, med], color=fs.NEUTRAL["ink"],
                linewidth=1.4, zorder=3, solid_capstyle="round")
        ax.annotate(f"{med:.3f}", (i, med), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.0, color=fs.NEUTRAL["ink"])

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{LABEL[a]}\nvs {LABEL[b]}" for a, b in pairs], fontsize=6.5)
    ax.set_ylabel("Identity agreement, ARI")
    ax.set_ylim(0.55, 1.03)
    ax.set_xlim(-0.55, len(pairs) - 0.45)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.spines["left"].set_bounds(0.6, 1.0)
    ax.spines["bottom"].set_bounds(0, len(pairs) - 1)
    ax.annotate(f"{agree.video_id.nunique()} videos", (0.5, 0.02), xycoords="axes fraction",
                ha="center", fontsize=6.0, color=fs.NEUTRAL["mid"])


def draw_panel_b(ax, main: pd.DataFrame, prov: dict, stab: pd.DataFrame) -> None:
    """Balanced accuracy per arm: own cohort and common cohort."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=fs.NEUTRAL["grid"], linewidth=0.5)
    m = main.set_index("arm")

    chance = float(m["chance_ba"].iloc[0])
    dens = float(m["density_only_ba"].max())
    ax.axhline(chance, color=fs.NEUTRAL["mid"], linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(dens, color=fs.NEUTRAL["light"], linewidth=0.7, linestyle=":", zorder=1)
    # Threshold labels sit hard left; the descriptive annotation owns the middle.
    ax.annotate("chance", (-0.42, chance), fontsize=5.8,
                color=fs.NEUTRAL["mid"], va="bottom", ha="left")
    ax.annotate("density-only", (-0.42, dens), fontsize=5.8,
                color=fs.NEUTRAL["mid"], va="top", ha="left")

    for i, arm in enumerate(ARMS):
        own, com = m.loc[arm, "own_cohort_lovo_ba"], m.loc[arm, "common_cohort_lovo_ba"]
        ax.plot([i, i], [own, com], color=fs.NEUTRAL["light"], linewidth=1.0, zorder=2)
        ax.plot([i], [own], linestyle="none", marker=MARKER[arm], markersize=MS,
                markerfacecolor="white", markeredgecolor=COLOR[arm],
                markeredgewidth=1.1, zorder=3)
        ax.plot([i], [com], linestyle="none", marker=MARKER[arm], markersize=MS,
                color=COLOR[arm], markeredgecolor=EDGE, markeredgewidth=EDGE_W, zorder=4)
        # The two cohorts coincide for two of the three arms, so the value goes
        # to the right of the filled marker and the cohort size above the open
        # one -- side by side they overprint into an unreadable run of digits.
        ax.annotate(f"{com:.3f}", (i, com), textcoords="offset points", xytext=(10, 0),
                    ha="left", va="center", fontsize=6.0, color=fs.NEUTRAL["ink"])
        ax.annotate(f"n={int(m.loc[arm,'own_cohort_videos'])}", (i, own),
                    textcoords="offset points", xytext=(0, 8), ha="center", va="bottom",
                    fontsize=5.8, color=fs.NEUTRAL["mid"])

    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=6.5)
    ax.set_ylabel("Species balanced accuracy (LOVO)")
    ax.set_ylim(0.13, 1.05)
    ax.set_xlim(-0.5, len(ARMS) - 0.35)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.spines["left"].set_bounds(0.2, 1.0)
    ax.spines["bottom"].set_bounds(0, len(ARMS) - 1)

    pa = prov["prediction_agreement_common_cohort"]
    rho = stab["spearman_mean"].median()
    ax.annotate(f"identical prediction on {pa['all_three_identical']}/{pa['n_videos']} "
                f"common-cohort videos\nmedian descriptor agreement "
                f"$\\rho$ = {rho:.3f} over 63 descriptors",
                (0.5, 0.40), xycoords="axes fraction", ha="center", va="center",
                fontsize=6.0, color=fs.NEUTRAL["mid"])

    handles = [fs.plt.Line2D([], [], linestyle="none", marker="o", markersize=4.4,
                             markerfacecolor="white", markeredgecolor=fs.NEUTRAL["mid"],
                             markeredgewidth=1.1, label="own QC cohort"),
               fs.plt.Line2D([], [], linestyle="none", marker="o", markersize=4.4,
                             color=fs.NEUTRAL["mid"], markeredgecolor=EDGE,
                             markeredgewidth=EDGE_W, label="common cohort")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.2,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3,
              bbox_to_anchor=(0.0, 0.62))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--results", default=str(REPO_ROOT / "reports" / "tracker_robustness"))
    args = ap.parse_args(argv)

    fs.apply_style()
    bench, results = Path(args.benchmark), Path(args.results)
    agree, main_tbl, prov, stab = _load(bench, results)
    out_dir = fs.resolve_paths(fs.load_config()).output_dir

    sidecar = fs.Sidecar(figure=FIG, description=(
        "Cross-tracker identity agreement (a) and species classification balanced "
        "accuracy per tracker (b), for ByteTrack, OC-SORT and BoT-SORT."))
    for p, role in [(results / "tracker_robustness_results.csv", "per-arm results"),
                    (results / "tracker_representation_agreement.csv", "descriptor agreement"),
                    (bench / "evaluation" / "identity_agreement_per_video.parquet",
                     "per-video identity agreement")]:
        sidecar.add_input(p, role)
    sidecar.param("arms", ARMS)
    sidecar.param("palette", COLOR)
    sidecar.note("bytetrack_iou excluded: bit-identical to bytetrack (benchmark report §4.1).")
    sidecar.note("Agreement is between two trackers, not against ground truth: there is no "
                 "identity-resolved MOT ground truth, so these are not HOTA/AssA/IDF1 and "
                 "two arms can agree perfectly and both be wrong.")
    sidecar.note("OC-SORT hue moved from #e87ba4 to #a01f5c: the original fell below the "
                 "CVD separation floor against the ByteTrack orange (worst-CVD dE 4.9).")

    fig, axes = fs.plt.subplots(1, 2, figsize=(7.2, 3.1),
                                gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.32})
    draw_a = lambda ax: draw_panel_a(ax, agree)             # noqa: E731
    draw_b = lambda ax: draw_panel_b(ax, main_tbl, prov, stab)  # noqa: E731
    draw_a(axes[0]); draw_b(axes[1])
    fs.panel_label(axes[0], "a", dx=-0.155, dy=1.02)
    fs.panel_label(axes[1], "b", dx=-0.175, dy=1.02)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.93, bottom=0.16)

    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{FIG}.svg"
    fig.savefig(svg)
    sidecar.param("svg", str(svg))

    table = main_tbl.copy()
    table["panel"] = "b"
    ag = agree.groupby(["arm_a", "arm_b"])["ari"].median().reset_index()
    ag["panel"] = "a"
    outputs = fs.save_figure(fig, FIG, sidecar,
                             pd.concat([table, ag], ignore_index=True), out_dir,
                             panels={"a": draw_a, "b": draw_b},
                             panel_size={"a": (3.6, 3.0), "b": (3.6, 3.0)})
    print(f"  {'svg':14s} {svg}")
    fs.report_checks(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
