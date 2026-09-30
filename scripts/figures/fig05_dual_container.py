#!/usr/bin/env python3
"""Figure 5 - controlled dual-container analysis.

Four recordings film two species side by side in a physically divided container,
so camera, lighting, session, water and observer are held constant by
construction and only the compartment differs. This is a controlled within-video
comparison, not a definitive biological replication study: each pair of videos
shares a date, camera position and container, so the two videos of a pair are
probably one setup filmed twice.

Species come from the filename's SX/DX compartment metadata and a divider that
was verified against a frame of each video - never from detector class
predictions. Movement variables are body-length normalised.

Run from the repository root:

    python scripts/figures/fig05_dual_container.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "src"))

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

from mosmon_tracking.metadata import parse_filename  # noqa: E402

FIGURE = "fig05_dual_container"

MANUSCRIPT = {
    "n_recordings": 4,
    "purity_threshold": 0.95,
    "split_fractions": [0.4961, 0.5000, 0.4961, 0.5117],
    "control_max_abs_r": 0.101,
    "control_n_significant": 5,
    "control_n_tests": 116,
    "albo_vs_culex": {
        "mean_speed_bl_s": +0.609, "p90_speed_bl_s": +0.602,
        "pause_fraction": -0.585, "median_speed_bl_s": +0.566,
        "pause_bout_mean_s": -0.525, "turn_abs_mean_rad": -0.506,
        "move_bout_mean_s": +0.453, "reversal_fraction": -0.440,
    },
    "aegypti_vs_albo": {
        "nn_distance_bl": +0.242, "turn_abs_mean_rad": +0.223,
        "turn_circular_variance": +0.207, "mean_speed_bl_s": -0.149,
        "straightness": -0.148,
    },
}

PAIR_MAIN = ("Ae. albopictus", "Cx. pipiens")
PAIR_SUPP = ("Ae. aegypti", "Ae. albopictus")

_FEATURE_LABEL = {
    "mean_speed_bl_s": "Mean speed (BL/s)",
    "median_speed_bl_s": "Median speed (BL/s)",
    "p90_speed_bl_s": "90th-percentile speed (BL/s)",
    "max_speed_bl_s": "Maximum speed (BL/s)",
    "speed_cv": "Speed variability",
    "speed_burstiness": "Speed burstiness",
    "speed_ac1": "Speed autocorrelation",
    "speed_acorr_time_s": "Speed autocorrelation time (s)",
    "pause_fraction": "Pause fraction",
    "pause_bout_mean_s": "Mean pause-bout duration (s)",
    "move_bout_mean_s": "Mean movement-bout duration (s)",
    "bout_rate_hz": "Bout rate (Hz)",
    "turn_abs_mean_rad": "Mean absolute turning angle",
    "turn_circular_variance": "Turning circular variance",
    "turn_ac1": "Turn autocorrelation",
    "reversal_fraction": "Reversal fraction",
    "path_length_bl": "Path length (BL)",
    "net_displacement_bl": "Net displacement (BL)",
    "radius_of_gyration_bl": "Radius of gyration (BL)",
    "nn_distance_bl": "Nearest-neighbour distance (BL)",
    "straightness": "Straightness",
    "log_tortuosity": "log Tortuosity",
    "msd_alpha": "MSD exponent",
    "msd_r2": "MSD fit $R^2$",
    "aspect_ratio_cv": "Aspect-ratio variability",
    "aspect_ratio_ac1": "Aspect-ratio autocorrelation",
    "aspect_peak_freq_hz": "Aspect-ratio peak frequency (Hz)",
    "aspect_peak_power_fraction": "Aspect-ratio peak power fraction",
    "spatial_entropy_track": "Spatial entropy of track",
}

SIDE_COLOUR = {"left": "#1f6fb4", "right": "#d1701c", "rejected": "#9e9e9e"}
# Colour alone must not carry the category: in grayscale these three converge.
SIDE_LINESTYLE = {"left": "-", "right": "-", "rejected": (0, (2.2, 1.2))}


def feature_label(name: str) -> str:
    return _FEATURE_LABEL.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dual_runs(paths: fs.FigurePaths, side: fs.Sidecar) -> list[dict]:
    """The four dual-container runs with their verified divider positions."""
    splits_path = side.add_input(paths.dual_reanalysis / "splits.json",
                                 "verified divider positions")
    with open(splits_path) as fh:
        splits = json.load(fh)

    runs = []
    for run, split in sorted(splits.items()):
        run_dir = paths.dual_reanalysis / run
        summary_path = run_dir / "video_summary.json"
        if not summary_path.exists():
            raise SystemExit(f"Missing required input [dual run summary]: {summary_path}")
        with open(summary_path) as fh:
            info = json.load(fh)["video_info"]
        meta = parse_filename(info["filename"])
        meta = dataclasses.asdict(meta) if dataclasses.is_dataclass(meta) else vars(meta)
        if not meta.get("is_dual_container"):
            raise SystemExit(f"{run} is not parsed as a dual-container recording.")
        if not (meta.get("species_left") and meta.get("species_right")):
            raise SystemExit(
                f"{run}: filename does not unambiguously map a side to a species, so no "
                "compartment species labels can be assigned. Nothing was plotted."
            )
        runs.append({
            "run": run, "run_dir": run_dir, "video": info["filename"],
            "split_x_fraction": float(split),
            "width": int(info["width"]), "height": int(info["height"]),
            "fps": float(info["fps"]),
            "species_left": fs.species_label(meta["species_left"]),
            "species_right": fs.species_label(meta["species_right"]),
        })
    return runs


def track_side_assignment(run: dict, purity_threshold: float) -> pd.DataFrame:
    """Per-track compartment purity and its accept/reject verdict.

    ``compartment`` is written per observation by the pipeline from the verified
    divider. Purity is the fraction of a track's observations on its majority
    side; a track that changes side is an identity error rather than a larva,
    because the two boxes are physically separate.
    """
    tracks_path = run["run_dir"] / "tracks_clean.parquet"
    if not tracks_path.exists():
        raise SystemExit(f"Missing required input [clean tracks]: {tracks_path}")
    df = pd.read_parquet(tracks_path, columns=["track_id", "frame_idx", "cx", "cy",
                                               "compartment"])
    if "compartment" not in df.columns:
        raise SystemExit(f"{run['run']}: tracks_clean.parquet has no 'compartment' column; "
                         "the run was not re-analysed with a configured divider.")
    sides = df[df["compartment"].isin(["left", "right"])]
    counts = (sides.groupby(["track_id", "compartment"]).size()
              .unstack(fill_value=0).reindex(columns=["left", "right"], fill_value=0))
    totals = df.groupby("track_id").size().reindex(counts.index)
    majority = counts.idxmax(axis=1)
    purity = counts.max(axis=1) / totals
    return pd.DataFrame({
        "track_id": counts.index, "side": majority.to_numpy(),
        "purity": purity.to_numpy(), "n_obs": totals.to_numpy(),
        "accepted": (purity >= purity_threshold).to_numpy(),
    }), df


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def draw_frame(ax, ctx: dict) -> None:
    ax.imshow(ctx["image"], interpolation="nearest")
    fs.hide_axes(ax)
    ax.set_anchor("N")
    x = ctx["split_x_fraction"] * ctx["image"].shape[1]
    ax.axvline(x, color="#f2a93b", lw=1.1, ls="--")
    h = ctx["image"].shape[0]
    for frac, label in ((0.25, ctx["species_left"]), (0.75, ctx["species_right"])):
        ax.text(frac * ctx["image"].shape[1], 0.055 * h, label, ha="center", va="top",
                fontsize=6.0, style="italic", color="white",
                bbox=dict(boxstyle="round,pad=0.22", fc="black", ec="none", alpha=0.55))
    ax.text(x, 0.985 * h, f" divider  x = {ctx['split_x_fraction']:.3f} W ", fontsize=5.6,
            color="#f2a93b", ha="center", va="bottom")
    ax.set_title("Dual-container recording", pad=3)


def draw_occupancy(ax, ctx: dict) -> None:
    hist = ctx["occupancy"]
    ax.imshow(hist.T, origin="upper", cmap="magma", aspect="equal",
              extent=(0, ctx["width"], ctx["height"], 0),
              norm="log" if hist.max() > 0 else None)
    ax.set_anchor("N")
    ax.axvline(ctx["split_x_fraction"] * ctx["width"], color="#59c3f2", lw=1.1, ls="--")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"Track occupancy  ·  {ctx['bins_x']}x{ctx['bins_y']} bins", pad=3)
    ax.text(0.5, -0.58, "low-occupancy band = the physical divider",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
            color=fs.NEUTRAL["mid"])


def draw_assignment(ax, ctx: dict) -> None:
    for category in ("left", "right", "rejected"):
        for xy in ctx["trajectories"][category]:
            ax.plot(xy[:, 0], xy[:, 1], lw=0.45 if category != "rejected" else 0.9,
                    alpha=0.8, color=SIDE_COLOUR[category],
                    ls=SIDE_LINESTYLE[category], solid_capstyle="round",
                    zorder=3 if category == "rejected" else 2)
    ax.axvline(ctx["split_x_fraction"] * ctx["width"], color=fs.NEUTRAL["ink"],
               lw=1.0, ls="--")
    ax.set_xlim(0, ctx["width"])
    ax.set_ylim(ctx["height"], 0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"Track side assignment  ·  $\\geq${ctx['purity_threshold']:.0%} same side",
                 pad=3)
    handles = [
        plt.Line2D([], [], color=SIDE_COLOUR["left"], lw=1.4,
                   label=f"left: {ctx['species_left']}  ($n$={ctx['n_left']})"),
        plt.Line2D([], [], color=SIDE_COLOUR["right"], lw=1.4,
                   label=f"right: {ctx['species_right']}  ($n$={ctx['n_right']})"),
        plt.Line2D([], [], color=SIDE_COLOUR["rejected"], lw=1.4,
                   ls=SIDE_LINESTYLE["rejected"],
                   label=f"rejected (crossing): $n$={ctx['n_rejected']}"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.46),
              fontsize=5.8, ncol=1, handlelength=1.1)


def draw_effects(ax, effects: pd.DataFrame, pair: tuple[str, str], n_top: int) -> None:
    top = effects.reindex(effects["rank_biserial"].abs().sort_values().index).tail(n_top)
    y = np.arange(len(top))
    values = top["rank_biserial"].to_numpy()
    colours = [SIDE_COLOUR["right"] if v < 0 else SIDE_COLOUR["left"] for v in values]
    ax.axvline(0, color=fs.NEUTRAL["ink"], lw=0.8)
    ax.hlines(y, 0, values, color=fs.NEUTRAL["faint"], lw=1.0, zorder=1)
    ax.scatter(values, y, s=32, c=colours, edgecolor="white", linewidth=0.5, zorder=3)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + (0.035 if v > 0 else -0.035), yi, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=6.2, color=fs.NEUTRAL["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(f) for f in top["feature"]], fontsize=6.4)
    lim = float(np.abs(values).max()) * 1.42
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Rank-biserial $r$ (within-video, mean of both recordings)")
    # Sides are labelled directly above each half and colour-matched to the dots,
    # so the sign is readable without a legend and without italic mathtext.
    ax.text(0.0, 1.03, f"\u2190 {pair[1]} higher", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.6, style="italic",
            color=SIDE_COLOUR["right"])
    ax.text(1.0, 1.03, f"{pair[0]} higher \u2192", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.6, style="italic",
            color=SIDE_COLOUR["left"])


def draw_control(ax, control_abs: np.ndarray, effects: pd.DataFrame,
                 n_tests: int, n_sig: int) -> None:
    x = np.sort(control_abs)
    ecdf = np.arange(1, x.size + 1) / x.size
    ax.step(np.concatenate([[0.0], x]), np.concatenate([[0.0], ecdf]), where="post",
            color=fs.NEUTRAL["mid"], lw=1.2)
    ax.axvline(x.max(), color=fs.NEUTRAL["warn"], lw=0.9, ls="--")
    ax.text(x.max(), 0.5, f" max |$r$| = {x.max():.3f}", fontsize=6.2,
            color=fs.NEUTRAL["warn"], va="center", ha="left")
    real = effects["rank_biserial"].abs().to_numpy()
    ax.plot(real, np.full(real.size, 0.97), marker="|", ls="", ms=7,
            color=SIDE_COLOUR["left"])
    ax.text(float(real.mean()), 0.93, "between-compartment effects (panel d)",
            fontsize=5.8, color=SIDE_COLOUR["left"], ha="center", va="top")
    ax.set_xlim(0, max(float(real.max()), float(x.max())) * 1.10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("|rank-biserial $r$|")
    ax.set_ylabel("ECDF of artificial-split tests")
    ax.set_title(f"Artificial-split control  ·  {n_sig}/{n_tests} at nominal $p$ < 0.05",
                 pad=4)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def pair_effects(dual: pd.DataFrame, pair: tuple[str, str]) -> pd.DataFrame:
    """Mean rank-biserial per feature over the two recordings of one species pair.

    Sign convention follows the manuscript: positive means higher for ``pair[0]``.
    The stored table orders the two species alphabetically per row, so any row
    whose stored ``species_a`` is not ``pair[0]`` has its sign flipped.
    """
    labels_a = dual["species_a"].map(fs.species_label)
    labels_b = dual["species_b"].map(fs.species_label)
    m = ((labels_a == pair[0]) & (labels_b == pair[1])) | \
        ((labels_a == pair[1]) & (labels_b == pair[0]))
    sub = dual[m].copy()
    if sub.empty:
        raise SystemExit(f"No stored contrast rows for {pair[0]} vs {pair[1]}.")
    sign = np.where(labels_a[m] == pair[0], 1.0, -1.0)
    sub["rank_biserial"] = sub["rank_biserial"].to_numpy() * sign
    out = (sub.groupby("feature")
           .agg(rank_biserial=("rank_biserial", "mean"),
                n_videos=("rank_biserial", "size"),
                min_r=("rank_biserial", "min"), max_r=("rank_biserial", "max"),
                n_a=("n_a", "sum"), n_b=("n_b", "sum"))
           .reset_index())
    return out


def build_table(ctx: dict, effects: pd.DataFrame, supp: pd.DataFrame,
                dual: pd.DataFrame, runs: list[dict], n_top: int) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        rows.append({"panel": "a,b,c", "item": run["run"], "quantity": "split_x_fraction",
                     "value": run["split_x_fraction"], "video": run["video"]})
    for key in ("n_left", "n_right", "n_rejected", "n_tracks_total"):
        rows.append({"panel": "c", "item": ctx["run"], "quantity": key, "value": ctx[key],
                     "video": ctx["video"]})
    for _i, row in effects.reindex(
            effects["rank_biserial"].abs().sort_values(ascending=False).index).iterrows():
        rows.append({"panel": "d", "item": feature_label(row["feature"]),
                     "quantity": "rank_biserial (mean of 2 videos, + = Ae. albopictus)",
                     "value": float(row["rank_biserial"]), "feature": row["feature"],
                     "n_videos": int(row["n_videos"]), "min_r": float(row["min_r"]),
                     "max_r": float(row["max_r"]),
                     "in_top_n": bool(_i in effects["rank_biserial"].abs()
                                      .nlargest(n_top).index)})
    for _i, row in dual.iterrows():
        rows.append({"panel": "e", "item": f"{row['run']} / {row['feature']}",
                     "quantity": "control_rank_biserial", "feature": row["feature"],
                     "value": float(row["control_rank_biserial"]),
                     "control_p": float(row["control_p"])})
    for _i, row in supp.reindex(
            supp["rank_biserial"].abs().sort_values(ascending=False).index).head(10).iterrows():
        rows.append({"panel": "supp", "item": feature_label(row["feature"]),
                     "quantity": "rank_biserial (mean of 2 videos, + = Ae. aegypti)",
                     "value": float(row["rank_biserial"]), "feature": row["feature"]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to configs/figures.yaml")
    args = ap.parse_args(argv)

    cfg = fs.load_config(args.config)
    paths = fs.resolve_paths(cfg)
    paths.require("species_analysis", "dual_reanalysis")
    opts = cfg.get("fig05") or {}
    purity_threshold = float(opts.get("purity_threshold", 0.95))
    bins_x = int(opts.get("occupancy_bins_x", 192))
    bins_y = int(opts.get("occupancy_bins_y", 108))
    max_tracks = int(opts.get("max_tracks_plotted", 300))
    n_top = int(opts.get("top_features", 10))

    fs.apply_style()
    side = fs.Sidecar(
        figure=FIGURE,
        description=("Within-video species contrast under shared acquisition "
                     "conditions, with its artificial-split calibration control."),
        hash_max_bytes=cfg.get("hash_max_bytes"),
    )

    print(f"[{FIGURE}] loading dual-container runs")
    runs = load_dual_runs(paths, side)
    side.check("dual-container recordings", len(runs), MANUSCRIPT["n_recordings"], tol=0)
    for got, expected in zip(sorted(round(r["split_x_fraction"], 4) for r in runs),
                             sorted(MANUSCRIPT["split_fractions"]), strict=True):
        side.check(f"divider fraction {expected}", got, expected, tol=0.0005)

    dual_path = side.add_input(paths.species_analysis / "dual_container_contrast.parquet",
                               "within-video contrasts and artificial-split control")
    dual = pd.read_parquet(dual_path)
    for col in ("rank_biserial", "control_rank_biserial", "control_p", "species_a",
                "species_b", "feature", "run"):
        if col not in dual.columns:
            raise SystemExit(f"dual_container_contrast.parquet is missing column '{col}'.")
    if not np.isfinite(dual[["rank_biserial", "control_rank_biserial",
                             "control_p"]].to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite values in dual_container_contrast.parquet.")

    control_abs = dual["control_rank_biserial"].abs().to_numpy()
    n_sig = int((dual["control_p"] < 0.05).sum())
    side.check("artificial-split tests", len(dual), MANUSCRIPT["control_n_tests"], tol=0)
    side.check("artificial-split max |r|", round(float(control_abs.max()), 3),
               MANUSCRIPT["control_max_abs_r"], tol=0.0005)
    side.check("artificial-split nominal p < 0.05", n_sig,
               MANUSCRIPT["control_n_significant"], tol=0)

    effects = pair_effects(dual, PAIR_MAIN)
    supp = pair_effects(dual, PAIR_SUPP)
    for feature, expected in MANUSCRIPT["albo_vs_culex"].items():
        row = effects[effects["feature"] == feature]
        side.check(f"{PAIR_MAIN[0]} vs {PAIR_MAIN[1]}: {feature}",
                   None if row.empty else round(float(row["rank_biserial"].iloc[0]), 3),
                   expected, tol=0.0015)
    for feature, expected in MANUSCRIPT["aegypti_vs_albo"].items():
        row = supp[supp["feature"] == feature]
        side.check(f"{PAIR_SUPP[0]} vs {PAIR_SUPP[1]}: {feature}",
                   None if row.empty else round(float(row["rank_biserial"].iloc[0]), 3),
                   expected, tol=0.0015)

    ranked = effects.reindex(
        effects["rank_biserial"].abs().sort_values(ascending=False).index).head(n_top)
    extra = [f for f in ranked["feature"] if f not in MANUSCRIPT["albo_vs_culex"]]
    if extra:
        side.note(
            f"panel (d) shows the true top-{n_top} features by |mean rank-biserial|; "
            f"{', '.join(feature_label(f) for f in extra)} rank inside it but are absent "
            "from the manuscript's eight-row table."
        )

    # Panels (a)-(c) use one recording of the species pair in panel (d), chosen
    # deterministically by run identifier unless overridden in the config.
    candidates = [r for r in runs
                  if {r["species_left"], r["species_right"]} == set(PAIR_MAIN)]
    chosen_run = opts.get("representative_run")
    if chosen_run:
        matches = [r for r in runs if r["run"] == chosen_run]
        if not matches:
            raise SystemExit(f"representative_run not among the dual runs: {chosen_run}")
        run = matches[0]
        side.note(f"panels (a)-(c) overridden from configs/figures.yaml to {run['run']}")
    else:
        run = sorted(candidates or runs, key=lambda r: r["run"])[0]

    print(f"[{FIGURE}] building panels (a)-(c) from {run['video'][:60]}...")
    side.add_input(run["run_dir"] / "video_summary.json", "representative run summary")
    assignment, observations = track_side_assignment(run, purity_threshold)
    side.add_input(run["run_dir"] / "tracks_clean.parquet", "representative clean tracks")

    accepted = assignment[assignment["accepted"]]
    obs = observations.merge(assignment[["track_id", "side", "accepted"]], on="track_id",
                             how="left")
    acc_obs = obs[obs["accepted"].fillna(False)]
    hist, _xe, _ye = np.histogram2d(
        acc_obs["cx"].to_numpy(dtype=float), acc_obs["cy"].to_numpy(dtype=float),
        bins=[bins_x, bins_y], range=[[0, run["width"]], [0, run["height"]]])

    rng = np.random.default_rng(42)
    trajectories: dict[str, list[np.ndarray]] = {"left": [], "right": [], "rejected": []}
    for category, subset in (("left", accepted[accepted["side"] == "left"]),
                             ("right", accepted[accepted["side"] == "right"]),
                             ("rejected", assignment[~assignment["accepted"]])):
        ids = subset["track_id"].to_numpy()
        if ids.size > max_tracks:
            ids = np.sort(rng.choice(ids, max_tracks, replace=False))
        for _tid, g in obs[obs["track_id"].isin(ids)].groupby("track_id"):
            trajectories[category].append(
                g.sort_values("frame_idx")[["cx", "cy"]].to_numpy(dtype=float))

    frame_choice = fs.median_active_frame(run["run_dir"])
    video_path = fs.source_video_path(run["run_dir"])
    image, _scale = fs.read_video_frame(video_path, frame_choice["frame_idx"])
    # The dual re-analysis ran without input hashing, so the source video's
    # sha256 comes from the original full-eval run of the same video (same run
    # id). Falling back to hashing here would re-read several GB per rebuild.
    video_sha = fs.source_video_sha256(run["run_dir"])
    if video_sha is None:
        video_sha = fs.source_video_sha256(paths.full_eval / run["run"])
        if video_sha is not None:
            side.note("source-video sha256 taken from the original full-eval run; the "
                      "dual re-analysis was written without input hashing.")
    side.add_input(video_path, "representative source video", sha256=video_sha)

    ctx = {
        **run, "image": image, "occupancy": hist, "bins_x": bins_x, "bins_y": bins_y,
        "purity_threshold": purity_threshold, "trajectories": trajectories,
        "n_left": int((accepted["side"] == "left").sum()),
        "n_right": int((accepted["side"] == "right").sum()),
        "n_rejected": int((~assignment["accepted"]).sum()),
        "n_tracks_total": int(len(assignment)),
        **frame_choice,
    }
    side.select("panels_a_to_c", {
        "run": run["run"], "video": run["video"], "source_video": str(video_path),
        "frame_idx": ctx["frame_idx"], "frame_stride": ctx["frame_stride"],
        "n_active_in_frame": ctx["n_active_in_frame"],
        "split_x_fraction": run["split_x_fraction"],
        "species_left": run["species_left"], "species_right": run["species_right"],
        "native_resolution": f"{run['width']}x{run['height']}",
        "tracks_total": ctx["n_tracks_total"], "tracks_left": ctx["n_left"],
        "tracks_right": ctx["n_right"], "tracks_rejected": ctx["n_rejected"],
        "tracks_plotted_cap": max_tracks,
    })
    side.param("purity_threshold", purity_threshold)
    side.param("occupancy_bins", [bins_x, bins_y])
    side.param("occupancy_source", "accepted track observations")
    side.param("max_tracks_plotted_per_category", max_tracks)
    side.param("top_features_shown", n_top)
    side.note(
        f"{ctx['n_left'] + ctx['n_right']} of {ctx['n_tracks_total']} tracks meet the "
        f"{purity_threshold:.0%} same-side criterion in this recording; "
        f"{ctx['n_rejected']} cross the divider and are rejected as identity errors."
    )
    side.note("each pair of dual-container videos shares a date, camera position and "
              "container, so the two videos of a pair are probably one setup filmed "
              "twice - consistency across them is consistency across viewpoints.")

    print(f"[{FIGURE}] drawing")
    fig_w, left, right = 7.2, 0.075, 0.980
    wspace_top = 0.42
    top_w_in = fig_w * (right - left) / (3 + 2 * wspace_top)
    h_top = top_w_in * (ctx["image"].shape[0] / ctx["image"].shape[1]) + 0.30
    h_bot = 2.85
    top_pad, gap, bottom_pad = 0.34, 1.05, 0.68
    fig_h = top_pad + h_top + gap + h_bot + bottom_pad

    def band(y_in, h_in):
        return {"top": 1 - y_in / fig_h, "bottom": 1 - (y_in + h_in) / fig_h}

    fig = plt.figure(figsize=(fig_w, fig_h))
    row_top = gridspec.GridSpec(1, 3, figure=fig, left=left, right=right,
                                wspace=wspace_top, **band(top_pad, h_top))
    row_bot = gridspec.GridSpec(1, 2, figure=fig, left=left + 0.055, right=right,
                                wspace=0.52, width_ratios=[1.25, 1.0],
                                **band(top_pad + h_top + gap, h_bot))

    ax_a = fig.add_subplot(row_top[0])
    draw_frame(ax_a, ctx)
    fs.panel_label(ax_a, "a", dx=-0.03, dy=1.22)
    ax_b = fig.add_subplot(row_top[1])
    draw_occupancy(ax_b, ctx)
    fs.panel_label(ax_b, "b", dx=-0.28, dy=1.22)
    ax_c = fig.add_subplot(row_top[2])
    draw_assignment(ax_c, ctx)
    fs.panel_label(ax_c, "c", dx=-0.28, dy=1.22)
    ax_d = fig.add_subplot(row_bot[0])
    draw_effects(ax_d, effects, PAIR_MAIN, n_top)
    fs.panel_label(ax_d, "d", dx=-0.40, dy=1.10)
    ax_e = fig.add_subplot(row_bot[1])
    draw_control(ax_e, control_abs, ranked, len(dual), n_sig)
    fs.panel_label(ax_e, "e", dx=-0.26, dy=1.10)
    fig.text(0.5, 0.012,
             "Species come from SX/DX acquisition metadata and a divider verified against a "
             "video frame, never from detector predictions. Speeds are in apparent body "
             "lengths (BL).",
             ha="center", fontsize=6.2, color=fs.NEUTRAL["mid"])

    panels = {
        "a": lambda ax: draw_frame(ax, ctx),
        "b": lambda ax: draw_occupancy(ax, ctx),
        "c": lambda ax: draw_assignment(ax, ctx),
        "d": lambda ax: draw_effects(ax, effects, PAIR_MAIN, n_top),
        "e": lambda ax: draw_control(ax, control_abs, ranked, len(dual), n_sig),
        "supp_aegypti_vs_albopictus": lambda ax: draw_effects(ax, supp, PAIR_SUPP, n_top),
    }
    panel_size = {"a": (3.4, 2.2), "b": (3.4, 2.4), "c": (3.4, 2.6),
                  "d": (4.2, 3.0), "e": (3.6, 2.8),
                  "supp_aegypti_vs_albopictus": (4.2, 3.0)}

    fs.save_figure(fig, FIGURE, side,
                   build_table(ctx, effects, supp, dual, runs, n_top),
                   paths.output_dir, panels=panels, panel_size=panel_size)
    fs.report_checks(side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
