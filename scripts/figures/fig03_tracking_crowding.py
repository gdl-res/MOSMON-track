#!/usr/bin/env python3
"""Figure 3 - tracking quality across crowding.

Shows how tracker diagnostics vary across the 66-run evaluation as active-track
density changes, while frame-level detection coverage stays near saturation for
most recordings.

Nothing here is a ground-truth MOT measurement. ``duplicate_track_proxy`` is a
tracking *diagnostic*, frame detection coverage is the fraction of processed
frames holding at least one detection (not instance recall), and mean active
tracks/frame is tracker-derived, not a true larval count. The axis labels say so.

Run from the repository root:

    python scripts/figures/fig03_tracking_crowding.py
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

FIGURE = "fig03_tracking_crowding"

# Values stated in the manuscript, checked against the source data before plotting.
MANUSCRIPT = {
    "n_runs": 66,
    "n_unique_videos": 63,
    "density_min": 1.7,
    "density_median": 68.1,
    "density_max": 351.0,
    "coverage_median_pct": 99.79,
    "n_videos_above_99pct": 52,
    "r_dup_density": 0.601,
    "r_dup_coverage": -0.008,
    "n_below_97pct": 8,
    "quartiles": {
        # metric: (min, q1, median, q3, max)
        "dup": (0.000, 0.005, 0.025, 0.090, 0.390),
        "jump": (0.010, 0.016, 0.020, 0.026, 0.055),
        "interp": (0.004, 0.074, 0.135, 0.182, 0.305),
    },
}

QC_VARIABLES = [
    ("dup", "Duplicate-track\nproxy"),
    ("jump", "Large-jump\nfraction"),
    ("interp", "Interpolation\nfraction"),
    ("medlen", "Median track\nlength (frames)"),
]

DENSITY_AXIS = "Mean active tracks per frame\n(tracker-derived, not a true larval count)"


# --------------------------------------------------------------------------- #
# Data loading and validation
# --------------------------------------------------------------------------- #
def load_summary(paths: fs.FigurePaths, side: fs.Sidecar) -> pd.DataFrame:
    """Per-run QC table, cross-checked against each run's own video_summary.json."""
    side.add_input(paths.summary_table, "per-run QC summary table")
    df = pd.read_csv(paths.summary_table)

    if df["run"].duplicated().any():
        dupes = df.loc[df["run"].duplicated(), "run"].tolist()
        raise SystemExit(f"Duplicate run identifiers in the summary table: {dupes}")
    for col in ("dens", "dup", "cov", "jump", "interp", "medlen", "sha"):
        if col not in df.columns:
            raise SystemExit(f"Summary table is missing the required column '{col}'.")
    non_finite = {c: int((~np.isfinite(df[c])).sum())
                  for c in ("dens", "dup", "cov", "jump", "interp", "medlen")}
    if any(non_finite.values()):
        raise SystemExit(f"Non-finite values in the summary table: {non_finite}")

    # A run appearing twice under different names is the same footage: three
    # source videos were processed twice, so 66 runs cover 63 distinct inputs.
    df["is_duplicate_input"] = df["sha"].duplicated(keep=False)
    df["species_label"] = df["species"].map(fs.species_label)
    return df


def crosscheck_against_runs(df: pd.DataFrame, paths: fs.FigurePaths,
                            side: fs.Sidecar) -> None:
    """Re-derive every plotted QC value from the per-run JSON, independently.

    The summary CSV is a roll-up; this confirms it still matches the run
    directories it was rolled up from rather than trusting it on its own.
    """
    fields = {
        "cov": ("qc", "pct_frames_with_detections"),
        "dup": ("qc", "duplicate_track_proxy"),
        "jump": ("qc", "large_jump_fraction"),
        "interp": ("qc", "interpolation_fraction"),
        "medlen": ("qc", "median_track_length"),
        "dens": ("behaviour", "mean_active_tracks_per_frame"),
    }
    mismatches: list[str] = []
    checked = 0
    for _, row in df.iterrows():
        summary_path = Path(str(row["out_dir"])) / "video_summary.json"
        if not summary_path.exists():
            mismatches.append(f"{row['run']}: video_summary.json not found")
            continue
        with open(summary_path) as fh:
            summary = json.load(fh)
        checked += 1
        for col, (section, key) in fields.items():
            got = summary.get(section, {}).get(key)
            if got is None or not np.isclose(float(got), float(row[col]), rtol=0, atol=1e-6):
                mismatches.append(f"{row['run']}: {col} CSV={row[col]!r} JSON={got!r}")
    side.param("runs_crosschecked_against_video_summary_json", checked)
    if mismatches:
        raise SystemExit(
            "Summary table disagrees with the per-run video_summary.json files:\n  "
            + "\n  ".join(mismatches[:10])
        )
    print(f"  [ok  ] all {checked} runs match their own video_summary.json")


def validate(df: pd.DataFrame, side: fs.Sidecar) -> dict:
    """Compare every manuscript claim this figure rests on with the source data."""
    unique = df.drop_duplicates("sha")
    stats = {
        "n_runs": len(df),
        "n_unique_videos": int(df["sha"].nunique()),
        "density_min": float(df["dens"].min()),
        "density_median": float(df["dens"].median()),
        "density_max": float(df["dens"].max()),
        "coverage_median_pct": float(df["cov"].median()),
        "n_videos_above_99pct": int((unique["cov"] > 99).sum()),
        "r_dup_density": fs.pearson_r(df["dup"], df["dens"]),
        "r_dup_coverage": fs.pearson_r(df["dup"], df["cov"]),
        # Not a manuscript number: recomputed here so panel (c) can label it
        # separately, because it is easily confused with r_dup_coverage.
        "r_density_coverage": fs.pearson_r(df["dens"], df["cov"]),
        "r_dup_density_deduplicated": fs.pearson_r(unique["dup"], unique["dens"]),
    }

    side.check("n processed runs", stats["n_runs"], MANUSCRIPT["n_runs"], tol=0)
    side.check("n unique source videos", stats["n_unique_videos"],
               MANUSCRIPT["n_unique_videos"], tol=0)
    side.check("density min", stats["density_min"], MANUSCRIPT["density_min"], tol=0.05)
    side.check("density median", stats["density_median"],
               MANUSCRIPT["density_median"], tol=0.05)
    side.check("density max", stats["density_max"], MANUSCRIPT["density_max"], tol=0.05)
    side.check("median frame detection coverage (%)", round(stats["coverage_median_pct"], 2),
               MANUSCRIPT["coverage_median_pct"], tol=0.005, unit="%")
    side.check("unique videos above 99% coverage", stats["n_videos_above_99pct"],
               MANUSCRIPT["n_videos_above_99pct"], tol=0)
    side.check("r(duplicate proxy, density) over 66 runs", round(stats["r_dup_density"], 3),
               MANUSCRIPT["r_dup_density"], tol=0.0005)
    side.check("r(duplicate proxy, coverage) over 66 runs", round(stats["r_dup_coverage"], 3),
               MANUSCRIPT["r_dup_coverage"], tol=0.0005)

    for col, expected in MANUSCRIPT["quartiles"].items():
        got = (df[col].min(), df[col].quantile(0.25), df[col].median(),
               df[col].quantile(0.75), df[col].max())
        for label, g, e in zip(("min", "Q1", "median", "Q3", "max"), got, expected, strict=True):
            side.check(f"{col} {label}", round(float(g), 3), e, tol=0.0015)

    low = df[df["cov"] < MANUSCRIPT_LOW_COVERAGE].sort_values("cov")
    stats["n_below_97pct"] = len(low)
    agrees = side.check("recordings below 97% coverage", len(low),
                        MANUSCRIPT["n_below_97pct"], tol=0)
    if not agrees:
        side.note(
            f"{len(low)} recordings fall below 97% coverage, not the "
            f"{MANUSCRIPT['n_below_97pct']} stated in the manuscript. The additional one is "
            f"{low.iloc[-1]['cov']:.2f}% ({low.iloc[-1]['species_label']}, density "
            f"{low.iloc[-1]['dens']:.1f}, duplicate proxy {low.iloc[-1]['dup']:.3f}); it is not "
            "a duplicated input. All of them are plotted."
        )
    sparse = low[(low["dens"] >= 1.7) & (low["dens"] <= 4.3)]
    side.note(
        f"of the {len(low)} low-coverage recordings, {len(sparse)} sit in the "
        f"1.7-4.3 density band (near-empty containers) and "
        f"{int((low['dens'] > 50).sum())} are dense/disturbed."
    )

    side.note(
        "panel (b) uses all 66 processed runs, the universe that reproduces the "
        f"manuscript's r = {MANUSCRIPT['r_dup_density']}; deduplicating to "
        f"{stats['n_unique_videos']} unique source videos gives r = "
        f"{stats['r_dup_density_deduplicated']:.3f}. Duplicated inputs are drawn as open markers."
    )
    return stats


MANUSCRIPT_LOW_COVERAGE = 97.0


# --------------------------------------------------------------------------- #
# Panel (a) - representative density regimes
# --------------------------------------------------------------------------- #
def select_example_runs(df: pd.DataFrame, percentiles: list[float],
                        overrides: dict, side: fs.Sidecar) -> list[dict]:
    """Pick one run per target density percentile, deterministically.

    Absolute min/max are deliberately avoided: the targets are percentiles, and
    ties break on the run identifier so a rerun always picks the same video.
    """
    labels = ["low", "median", "high"]
    picks: list[dict] = []
    for label, pct in zip(labels, percentiles, strict=True):
        target = float(np.percentile(df["dens"], pct))
        if label in overrides and overrides[label].get("run"):
            row = df[df["run"] == overrides[label]["run"]]
            if row.empty:
                raise SystemExit(f"Override run for '{label}' not found: {overrides[label]['run']}")
            row = row.iloc[0]
            side.note(f"panel (a) '{label}' overridden from configs/figures.yaml to {row['run']}")
        else:
            ranked = df.assign(_d=(df["dens"] - target).abs()).sort_values(["_d", "run"])
            row = ranked.iloc[0]
        picks.append({
            "regime": label, "target_percentile": pct, "target_density": target,
            "run": str(row["run"]), "video": str(row["video"]),
            "out_dir": str(row["out_dir"]), "density": float(row["dens"]),
            "dup": float(row["dup"]), "cov": float(row["cov"]),
            "width": int(row["w"]), "height": int(row["h"]), "fps": float(row["fps"]),
        })
    return picks


def choose_frame(pick: dict, overrides: dict) -> dict:
    """Processed frame whose active-track count is closest to the video's median."""
    label = pick["regime"]
    chosen = fs.median_active_frame(pick["out_dir"])
    if label in overrides and overrides[label].get("frame_idx") is not None:
        frame_idx = int(overrides[label]["frame_idx"])
        ts = pd.read_parquet(Path(pick["out_dir"]) / "population_timeseries.parquet",
                             columns=["frame_idx", "n_larvae"])
        match = ts[ts["frame_idx"] == frame_idx]
        if match.empty:
            raise SystemExit(f"Override frame {frame_idx} was not processed in {pick['run']}")
        chosen = {**chosen, "frame_idx": frame_idx,
                  "n_active_in_frame": int(match["n_larvae"].iloc[0])}
    return {**pick, **chosen}


def load_frame_overlay(pick: dict, trail_frames: int, max_labels: int,
                       max_display_width: int = 1500) -> dict:
    """Read the chosen video frame plus the track geometry drawn on top of it."""
    tracks_path = Path(pick["out_dir"]) / "tracks_clean.parquet"
    if not tracks_path.exists():
        raise SystemExit(f"Missing required input [clean tracks]: {tracks_path}")

    stride = max(1, pick["frame_stride"])
    first = pick["frame_idx"] - stride * trail_frames
    cols = ["frame_idx", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "video_path"]
    window = pd.read_parquet(
        tracks_path, columns=cols,
        filters=[("frame_idx", ">=", first), ("frame_idx", "<=", pick["frame_idx"])],
    )
    if window.empty:
        raise SystemExit(f"No tracks found near frame {pick['frame_idx']} of {pick['run']}")

    video_path = Path(str(window["video_path"].iloc[0]))
    frame, scale = fs.read_video_frame(video_path, pick["frame_idx"], max_display_width)

    current = window[window["frame_idx"] == pick["frame_idx"]].copy()
    boxes = current[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) * scale
    ids = current["track_id"].to_numpy()
    trails = [
        g.sort_values("frame_idx")[["cx", "cy"]].to_numpy(dtype=float) * scale
        for tid, g in window.groupby("track_id") if len(g) > 1 and tid in set(ids)
    ]

    # Labelling every box is unreadable at 338 tracks and hides the very thing
    # the panel is about, so a bounded, evenly spread subset is labelled instead:
    # sort left-to-right and take equally spaced ranks (deterministic).
    n_label = min(len(boxes), max_labels)
    if n_label:
        order = np.lexsort((boxes[:, 1], boxes[:, 0]))
        take = order[np.linspace(0, len(order) - 1, n_label).round().astype(int)]
        label_boxes, label_ids = boxes[take], ids[take]
    else:
        label_boxes, label_ids = boxes[:0], ids[:0]

    return {
        **pick, "image": frame, "boxes": boxes, "track_ids": ids, "trails": trails,
        "label_boxes": label_boxes, "label_ids": label_ids,
        "display_scale": scale, "source_video": str(video_path),
        "n_boxes_drawn": int(len(boxes)),
        "n_labels_drawn": int(n_label),
        "max_labels": max_labels,
    }


REGIME_TITLE = {"low": "Low density", "median": "Median density", "high": "High density"}


def draw_frame_panel(ax, ov: dict) -> None:
    """One density-regime frame with boxes, IDs and short trajectory tails."""
    ax.imshow(ov["image"], interpolation="nearest")
    fs.hide_axes(ax)
    ax.set_anchor("N")
    for trail in ov["trails"]:
        ax.plot(trail[:, 0], trail[:, 1], "-", lw=0.55,
                color=fs.NEUTRAL["accent"], alpha=0.8, solid_capstyle="round")
    for (x1, y1, x2, y2) in ov["boxes"]:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   lw=0.6, edgecolor="#f2a93b"))
    for (x1, y1, _x2, _y2), tid in zip(ov["label_boxes"], ov["label_ids"], strict=True):
        ax.text(x1, y1 - 2.0, f"{int(tid)}", fontsize=4.2, color="#f2a93b",
                ha="left", va="bottom", clip_on=True)
    ax.set_title(REGIME_TITLE[ov["regime"]], pad=3)
    ax.text(0.015, 0.975,
            f"{ov['n_active_in_frame']} active tracks\n"
            f"{ov['density']:.1f} mean/frame  ·  {ov['width']}x{ov['height']}",
            transform=ax.transAxes, fontsize=5.6, va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))


# --------------------------------------------------------------------------- #
# Panels (b) - (d)
# --------------------------------------------------------------------------- #
def draw_density_vs_dup(ax, df: pd.DataFrame, stats: dict) -> None:
    dup_gate = 0.15
    uniq = ~df["is_duplicate_input"]
    ax.scatter(df.loc[uniq, "dens"], df.loc[uniq, "dup"], s=16,
               facecolor=fs.NEUTRAL["accent"], edgecolor="white", linewidth=0.3,
               alpha=0.85, zorder=3, label="processed run")
    ax.scatter(df.loc[~uniq, "dens"], df.loc[~uniq, "dup"], s=20, marker="o",
               facecolor="none", edgecolor=fs.NEUTRAL["accent"], linewidth=0.8,
               zorder=4, label="duplicated source video")

    grid, fit, lo, hi, _s, _i = fs.linfit_with_band(df["dens"], df["dup"])
    ax.fill_between(grid, lo, hi, color=fs.NEUTRAL["accent"], alpha=0.13, lw=0, zorder=1)
    ax.plot(grid, fit, "-", color=fs.NEUTRAL["ink"], lw=1.0, zorder=2)
    ax.axhline(dup_gate, color=fs.NEUTRAL["warn"], lw=0.8, ls="--", zorder=2)
    ax.text(0.02, dup_gate, "QC threshold 0.15", fontsize=6, color=fs.NEUTRAL["warn"],
            transform=ax.get_yaxis_transform(), va="bottom", ha="left")
    ax.text(0.035, 0.95, f"Pearson $r$ = {stats['r_dup_density']:.3f}\n$n$ = {len(df)} runs",
            transform=ax.transAxes, fontsize=7, va="top", ha="left")
    ax.set_xlabel(DENSITY_AXIS)
    ax.set_ylabel("Duplicate-track proxy\n(tracking diagnostic)")
    ax.legend(loc="lower right", fontsize=6)


def draw_density_vs_coverage(ax, df: pd.DataFrame, stats: dict, xlim) -> None:
    low = df[df["cov"] < MANUSCRIPT_LOW_COVERAGE]
    high = df[df["cov"] >= MANUSCRIPT_LOW_COVERAGE]
    for ref in (95.0, 97.0, 99.0):
        ax.axhline(ref, color=fs.NEUTRAL["faint"], lw=0.6, zorder=1)
        ax.text(0.985, ref, f"{ref:.0f}%", fontsize=5.5, color=fs.NEUTRAL["light"],
                transform=ax.get_yaxis_transform(), va="top", ha="right")
    ax.scatter(high["dens"], high["cov"], s=16, facecolor=fs.NEUTRAL["mid"],
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
    ax.scatter(low["dens"], low["cov"], s=26, marker="^", facecolor=fs.NEUTRAL["warn"],
               edgecolor="white", linewidth=0.3, zorder=4,
               label=f"below 97% ({len(low)} recordings)")
    ax.text(0.30, 0.30,
            f"median coverage {stats['coverage_median_pct']:.2f}%\n"
            f"{stats['n_videos_above_99pct']}/{stats['n_unique_videos']} unique videos > 99%\n"
            f"$r$(density, coverage) = {stats['r_density_coverage']:+.3f}",
            transform=ax.transAxes, fontsize=6.4, va="bottom", ha="left")
    ax.set_xlabel(DENSITY_AXIS)
    ax.set_ylabel("Frame detection coverage (%)\n(frames with $\\geq$1 detection)")
    ax.set_xlim(*xlim)
    ax.legend(loc="lower right", fontsize=6, bbox_to_anchor=(1.0, 0.02))


def draw_qc_distribution(ax, values: np.ndarray, label: str) -> None:
    """Violin + box for one QC variable, on its own scale."""
    parts = ax.violinplot([values], positions=[0], widths=0.75,
                          showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor(fs.NEUTRAL["accent"])
        body.set_alpha(0.22)
        body.set_edgecolor(fs.NEUTRAL["mid"])
        body.set_linewidth(0.5)
    ax.boxplot([values], positions=[0], widths=0.16, showfliers=False,
               medianprops=dict(color=fs.NEUTRAL["ink"], lw=1.2),
               boxprops=dict(color=fs.NEUTRAL["mid"], lw=0.7),
               whiskerprops=dict(color=fs.NEUTRAL["mid"], lw=0.7),
               capprops=dict(color=fs.NEUTRAL["mid"], lw=0.7))
    jitter = np.random.default_rng(42).uniform(-0.055, 0.055, values.size)
    ax.scatter(jitter, values, s=3.2, color=fs.NEUTRAL["ink"], alpha=0.4, zorder=3)
    ax.set_xticks([])
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylabel(label, fontsize=7)
    ax.set_title(f"median {np.median(values):.3g}", fontsize=6.2,
                 color=fs.NEUTRAL["mid"], pad=2)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_table(df: pd.DataFrame, overlays: list[dict], stats: dict) -> pd.DataFrame:
    """Exactly the values plotted, one tidy table."""
    per_run = df[["run", "video", "sha", "is_duplicate_input", "species_label",
                  "dens", "dup", "cov", "jump", "interp", "medlen"]].copy()
    per_run = per_run.rename(columns={
        "dens": "mean_active_tracks_per_frame", "dup": "duplicate_track_proxy",
        "cov": "frame_detection_coverage_pct", "jump": "large_jump_fraction",
        "interp": "interpolation_fraction", "medlen": "median_track_length_frames",
        "species_label": "species",
    })
    per_run.insert(0, "panel", "b,c,d")
    sel = pd.DataFrame([{
        "panel": "a", "run": o["run"], "video": o["video"], "sha": np.nan,
        "is_duplicate_input": np.nan, "species": np.nan,
        "mean_active_tracks_per_frame": o["density"],
        "duplicate_track_proxy": o["dup"], "frame_detection_coverage_pct": o["cov"],
        "large_jump_fraction": np.nan, "interpolation_fraction": np.nan,
        "median_track_length_frames": np.nan, "regime": o["regime"],
        "frame_idx": o["frame_idx"], "n_active_in_frame": o["n_active_in_frame"],
    } for o in overlays])
    out = pd.concat([sel, per_run], ignore_index=True)
    for key, value in stats.items():
        out[f"stat_{key}"] = value
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to configs/figures.yaml")
    args = ap.parse_args(argv)

    cfg = fs.load_config(args.config)
    paths = fs.resolve_paths(cfg)
    paths.require("summary_table", "full_eval")
    opts = cfg.get("fig03") or {}

    fs.apply_style()
    side = fs.Sidecar(
        figure=FIGURE,
        description=("Tracking diagnostics across the density range of the 66-run "
                     "evaluation. QC proxies only; no ground-truth MOT quantities."),
        hash_max_bytes=cfg.get("hash_max_bytes"),
    )

    print(f"[{FIGURE}] loading and validating source data")
    df = load_summary(paths, side)
    crosscheck_against_runs(df, paths, side)
    stats = validate(df, side)

    print(f"[{FIGURE}] selecting representative frames")
    overrides = opts.get("overrides") or {}
    trail = int(opts.get("trail_frames", 6))
    max_labels = int(opts.get("max_labels", 60))
    side.param("density_percentiles", opts.get("density_percentiles", [10, 50, 90]))
    side.param("trail_frames", trail)
    side.param("max_track_id_labels", max_labels)
    side.param("low_coverage_threshold_pct", MANUSCRIPT_LOW_COVERAGE)

    picks = select_example_runs(df, list(opts.get("density_percentiles", [10, 50, 90])),
                                overrides, side)
    overlays = []
    for pick in picks:
        pick = choose_frame(pick, overrides)
        ov = load_frame_overlay(pick, trail, max_labels)
        side.add_input(Path(ov["out_dir"]) / "population_timeseries.parquet",
                       f"population timeseries ({ov['regime']})")
        side.add_input(ov["source_video"], f"source video ({ov['regime']})",
                       sha256=fs.source_video_sha256(ov["out_dir"]))
        side.select(f"panel_a_{ov['regime']}", {
            "run": ov["run"], "video": ov["video"], "source_video": ov["source_video"],
            "frame_idx": ov["frame_idx"], "frame_stride": ov["frame_stride"],
            "n_active_in_frame": ov["n_active_in_frame"],
            "median_active_tracks": ov["median_active_tracks"],
            "mean_active_tracks_per_frame": ov["density"],
            "target_percentile": ov["target_percentile"],
            "native_resolution": f"{ov['width']}x{ov['height']}",
            "display_scale": ov["display_scale"],
            "n_boxes_drawn": ov["n_boxes_drawn"], "n_labels_drawn": ov["n_labels_drawn"],
        })
        overlays.append(ov)
        print(f"  {ov['regime']:>6}: frame {ov['frame_idx']} of {ov['video'][:60]}... "
              f"({ov['n_active_in_frame']} active)")

    print(f"[{FIGURE}] drawing")
    xlim = (-8, float(df["dens"].max()) * 1.06)

    # The frame row is sized from the real image aspect rather than a guessed
    # ratio: an imshow axes keeps its aspect, so a mismatched row height would
    # leave a band of white between panel (a) and panel (b).
    fig_w, left, right = 7.2, 0.085, 0.985
    wspace_a = 0.06
    frame_w_in = fig_w * (right - left) / (3 + 2 * wspace_a)
    frame_aspect = max(o["image"].shape[0] / o["image"].shape[1] for o in overlays)
    # The axes box must match the image exactly, otherwise `set_anchor("N")`
    # leaves the slack as a white band underneath. Title space goes in top_pad.
    h_a = frame_w_in * frame_aspect
    h_b, h_d = 2.85, 2.10                        # scatter row, distribution row
    top_pad, gap_ab, gap_bd, bottom_pad = 0.52, 0.42, 0.92, 0.46
    fig_h = top_pad + h_a + gap_ab + h_b + gap_bd + h_d + bottom_pad

    def band(y_top_in: float, height_in: float) -> dict:
        """Figure-fraction top/bottom for a row placed y_top_in below the top."""
        return {"top": 1.0 - y_top_in / fig_h, "bottom": 1.0 - (y_top_in + height_in) / fig_h}

    fig = plt.figure(figsize=(fig_w, fig_h))
    row_a = gridspec.GridSpec(1, 3, figure=fig, left=left, right=right,
                              wspace=wspace_a, **band(top_pad, h_a))
    row_b = gridspec.GridSpec(1, 2, figure=fig, left=left, right=right, wspace=0.34,
                              **band(top_pad + h_a + gap_ab, h_b))
    row_c = gridspec.GridSpec(1, 4, figure=fig, left=left, right=right, wspace=0.62,
                              **band(top_pad + h_a + gap_ab + h_b + gap_bd, h_d))

    axes_a = [fig.add_subplot(row_a[i]) for i in range(3)]
    for ax, ov in zip(axes_a, overlays, strict=True):
        draw_frame_panel(ax, ov)
    fs.panel_label(axes_a[0], "a", dx=-0.02, dy=1.16)

    ax_b = fig.add_subplot(row_b[0])
    draw_density_vs_dup(ax_b, df, stats)
    ax_b.set_xlim(*xlim)
    fs.panel_label(ax_b, "b")

    ax_c = fig.add_subplot(row_b[1])
    draw_density_vs_coverage(ax_c, df, stats, xlim)
    fs.panel_label(ax_c, "c")

    axes_d = [fig.add_subplot(row_c[i]) for i in range(4)]
    for ax, (col, label) in zip(axes_d, QC_VARIABLES, strict=True):
        draw_qc_distribution(ax, df[col].to_numpy(dtype=float), label)
    fs.panel_label(axes_d[0], "d", dx=-0.44, dy=1.13)
    fig.text(0.5, 0.10 / fig_h,
                          f"Tracking QC indicators across the {len(df)} processed runs "
                          "(separate scales)",
                          ha="center", fontsize=7, color=fs.NEUTRAL["mid"])

    panels = {
        **{f"a_{o['regime']}": (lambda ax, o=o: draw_frame_panel(ax, o)) for o in overlays},
        "b": lambda ax: (draw_density_vs_dup(ax, df, stats), ax.set_xlim(*xlim))[0],
        "c": lambda ax: draw_density_vs_coverage(ax, df, stats, xlim),
        **{f"d_{col}": (lambda ax, c=col, lb=lbl: draw_qc_distribution(
            ax, df[c].to_numpy(dtype=float), lb)) for col, lbl in QC_VARIABLES},
    }
    panel_size = {f"a_{o['regime']}": (3.4, 2.0) for o in overlays}
    panel_size.update({"b": (3.5, 2.8), "c": (3.5, 2.8)})
    panel_size.update({f"d_{c}": (1.7, 2.4) for c, _ in QC_VARIABLES})

    fs.save_figure(fig, FIGURE, side, build_table(df, overlays, stats),
                   paths.output_dir, panels=panels, panel_size=panel_size)
    fs.report_checks(side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
