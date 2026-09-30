#!/usr/bin/env python3
"""Figure 4 - population-level trajectory information.

Shows that the 63-dimensional video-level trajectory representation carries
substantial species-associated information, that individual short tracks carry
far less, and that generalisation weakens sharply once whole acquisition
sessions are held out.

Species labels come from acquisition metadata, never from detector class
predictions. Movement variables are body-length normalised (apparent body
lengths), not physical units. The figure does not claim the classifier isolated
intrinsic species behaviour: the session-holdout panel is what bounds that.

Two quantities the pipeline summarises but does not persist - the per-video LOVO
predictions and the individual permutation draws - are regenerated here from the
stored feature table with the analysis's own seed, then checked against the
stored aggregates before anything is drawn.

Run from the repository root:

    python scripts/figures/fig04_population_analysis.py
"""

from __future__ import annotations

import argparse
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

from mosmon_tracking import species_analysis as sa  # noqa: E402
from mosmon_tracking.config import Config  # noqa: E402

FIGURE = "fig04_population_analysis"

MANUSCRIPT = {
    "n_videos": 45,
    "class_counts": {"Ae. aegypti": 7, "Ae. albopictus": 22,
                     "An. stephensi": 6, "Cx. pipiens": 10},
    "correct": {"Ae. aegypti": 6, "Ae. albopictus": 21,
                "An. stephensi": 5, "Cx. pipiens": 8},
    "total_correct": 40,
    "schemes": {
        "Leave-one-video-out":            (0.861, 0.231, 0.359, 0.005),
        "Leave-one-camera-model-out":     (0.802, 0.232, 0.333, 0.005),
        "Leave-one-recording-session-out": (0.396, 0.227, 0.336, 0.015),
    },
    "controls": {
        "Video-level LOVO": 0.861,
        "Density-residualised video features": 0.861,
        "Neighbour-distance features removed": 0.801,
        "Individual-track classification": 0.307,
        "Individual-track video majority vote": 0.250,
        "Density-only baseline": 0.239,
        "Camera-model prediction control": 0.270,
    },
    "density_matched_n_videos": 24,
    "density_matched_n_significant": 47,
    "top_features": {
        "path_length_bl_med": 0.828, "path_length_bl_iqr": 0.778,
        "aspect_ratio_cv_iqr": 0.762, "radius_of_gyration_bl_med": 0.740,
        "net_displacement_bl_med": 0.714, "pause_fraction_med": 0.685,
        "median_speed_bl_s_med": 0.680, "p90_speed_bl_s_med": 0.672,
    },
}

CONTROL_GROUPS = [
    ("Population representation", [
        ("Video-level LOVO", ("video_level", "balanced_accuracy_lovo")),
        ("Density-residualised video features", ("density_residualised", "balanced_accuracy_lovo")),
    ]),
    ("Ablation / nuisance controls", [
        ("Neighbour-distance features removed",
         ("individual_behaviour_only", "balanced_accuracy_lovo")),
        ("Density-only baseline", ("density_only_baseline", "balanced_accuracy_lovo")),
    ]),
    ("Individual-track baselines", [
        ("Individual-track classification", ("track_level", "balanced_accuracy_per_track_lovo")),
        ("Individual-track video majority vote",
         ("track_level", "balanced_accuracy_per_video_vote")),
    ]),
    ("Different task (not species)", [
        ("Camera-model prediction control", ("camera_control", "balanced_accuracy_lovo")),
    ]),
]

# Readable axis labels for the video-level trajectory summaries. Suffixes carry
# the statistic: `_med` is the per-video median, `_iqr` its spread.
_FEATURE_BASE = {
    "path_length_bl": "Path length (BL)",
    "net_displacement_bl": "Net displacement (BL)",
    "radius_of_gyration_bl": "Radius of gyration (BL)",
    "nn_distance_bl": "Nearest-neighbour distance (BL)",
    "aspect_ratio_cv": "Aspect-ratio variability",
    "aspect_ratio_ac1": "Aspect-ratio autocorrelation",
    "aspect_peak_freq_hz": "Aspect-ratio peak frequency (Hz)",
    "aspect_peak_power_fraction": "Aspect-ratio peak power fraction",
    "pause_fraction": "Pause fraction",
    "pause_bout_mean_s": "Mean pause-bout duration (s)",
    "move_bout_mean_s": "Mean movement-bout duration (s)",
    "bout_rate_hz": "Bout rate (Hz)",
    "mean_speed_bl_s": "Mean speed (BL/s)",
    "median_speed_bl_s": "Median speed (BL/s)",
    "p90_speed_bl_s": "90th-percentile speed (BL/s)",
    "max_speed_bl_s": "Maximum speed (BL/s)",
    "speed_cv": "Speed variability",
    "speed_burstiness": "Speed burstiness",
    "speed_ac1": "Speed autocorrelation",
    "speed_acorr_time_s": "Speed autocorrelation time (s)",
    "straightness": "Straightness",
    "log_tortuosity": "log Tortuosity",
    "msd_alpha": "MSD exponent",
    "msd_r2": "MSD fit $R^2$",
    "turn_abs_mean_rad": "Mean absolute turning angle",
    "turn_circular_variance": "Turning circular variance",
    "turn_ac1": "Turn autocorrelation",
    "reversal_fraction": "Reversal fraction",
    "spatial_entropy_track": "Spatial entropy of track",
}


def feature_label(name: str) -> str:
    """Human-readable label for a video-level feature column."""
    for suffix, tail in (("_med", ""), ("_iqr", " IQR")):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return _FEATURE_BASE.get(base, base.replace("_", " ")) + tail
    return _FEATURE_BASE.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------- #
# Recomputation
# --------------------------------------------------------------------------- #
def regenerate_predictions_and_nulls(video_table: pd.DataFrame, features: list[str],
                                     cfg: Config, n_permutations: int) -> dict:
    """Re-derive the per-video predictions and the three permutation nulls.

    ``species_analysis.classify_species`` keeps only aggregates, so the draws are
    regenerated here. The random draws are consumed in exactly the order that
    function uses them - camera scheme, then session scheme, then the
    leave-one-video-out null - off one generator seeded with the project seed, so
    the reproduced summaries must equal the stored ones. They are checked.
    """
    opts = cfg.species_analysis
    rng = np.random.default_rng(cfg.project.random_seed)
    y = sa.as_labels(video_table["meta_species"]).to_numpy()
    runs = video_table["run"].to_numpy()
    cams = sa.as_labels(video_table["meta_camera_model"]).to_numpy()
    dates = sa.as_labels(video_table["meta_recording_date"]).to_numpy()
    X = sa._feature_matrix(video_table, features)

    def score(groups):
        pred, _, _ = sa.cv_predict(X, y, groups, opts.logistic_l2, opts.logistic_max_iter)
        return pred, sa.balanced_accuracy(y, pred)

    def null_for(groups):
        draws = []
        for _ in range(n_permutations):
            yp = rng.permutation(y)
            p, _, _ = sa.cv_predict(X, yp, groups, opts.logistic_l2, opts.logistic_max_iter)
            draws.append(sa.balanced_accuracy(yp, p))
        return np.array([v for v in draws if np.isfinite(v)])

    pred_lovo, bacc_lovo = score(runs)
    _pred_cam, bacc_cam = score(cams)
    null_cam = null_for(cams)                      # 1st block of draws
    _pred_ses, bacc_ses = score(dates)
    null_ses = null_for(dates)                     # 2nd block of draws
    null_lovo = null_for(runs)                     # 3rd block of draws

    def summarise(obs, draws):
        return {
            "observed": float(obs), "draws": draws,
            "null_mean": float(draws.mean()), "null_p95": float(np.quantile(draws, 0.95)),
            "p_value": float((np.sum(draws >= obs) + 1) / (draws.size + 1)),
            "n_permutations": int(draws.size),
        }

    return {
        "y_true": y, "y_pred_lovo": pred_lovo, "runs": runs,
        "Leave-one-video-out": summarise(bacc_lovo, null_lovo),
        "Leave-one-camera-model-out": summarise(bacc_cam, null_cam),
        "Leave-one-recording-session-out": summarise(bacc_ses, null_ses),
    }


def confusion_from_predictions(y_true, y_pred) -> pd.DataFrame:
    """4x4 count matrix in manuscript species order, rows = truth."""
    labels = fs.SPECIES_ORDER
    m = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        m.loc[fs.species_label(t), fs.species_label(p)] += 1
    return m


def validate(stored: dict, regen: dict, confusion: pd.DataFrame,
             controls: dict, uni: pd.DataFrame, side: fs.Sidecar) -> None:
    """Check every manuscript number and every regenerated quantity."""
    side.check("cohort size (videos)", int(stored["n_videos"]), MANUSCRIPT["n_videos"], tol=0)
    side.check("representation dimensionality", int(stored["n_features"]), 63, tol=0)
    for label, expected in MANUSCRIPT["class_counts"].items():
        got = int(confusion.loc[label].sum())
        side.check(f"cohort {label}", got, expected, tol=0)

    # The regenerated confusion must equal the one the analysis stored, which is
    # what licenses using regenerated predictions at all.
    #
    # Orientation matters: the stored matrix was written with
    # ``DataFrame.to_dict()``, whose default orientation nests column-major, so
    # the JSON is {predicted: {true: count}} even though the crosstab that
    # produced it had true on the index. Reading it row-major transposes the two
    # off-diagonal Aedes cells. The nesting below is the one whose row sums equal
    # the cohort counts, and it reproduces the published confusion table exactly.
    stored_conf = stored["video_level"]["confusion"]
    for pred_label, column in stored_conf.items():
        for true_label, count in column.items():
            side.check(f"confusion[{fs.species_label(true_label)} -> "
                       f"{fs.species_label(pred_label)}]",
                       int(confusion.loc[fs.species_label(true_label),
                                         fs.species_label(pred_label)]),
                       int(count), tol=0)
    for label, expected in MANUSCRIPT["correct"].items():
        side.check(f"correct {label}", int(confusion.loc[label, label]), expected, tol=0)
    side.check("total correct", int(np.trace(confusion.to_numpy())),
               MANUSCRIPT["total_correct"], tol=0)

    for scheme, (bacc, null_mean, null_p95, p) in MANUSCRIPT["schemes"].items():
        r = regen[scheme]
        side.check(f"{scheme}: balanced accuracy", round(r["observed"], 3), bacc, tol=0.0005)
        side.check(f"{scheme}: null mean", round(r["null_mean"], 3), null_mean, tol=0.0005)
        side.check(f"{scheme}: null 95th pct", round(r["null_p95"], 3), null_p95, tol=0.0005)
        side.check(f"{scheme}: p", round(r["p_value"], 3), p, tol=0.0005)

    for label, expected in MANUSCRIPT["controls"].items():
        side.check(f"control: {label}", round(controls[label], 3), expected, tol=0.0005)

    dm = uni[uni["variant"] == "density_matched"]
    side.check("density-matched cohort (videos)", int(dm["n_videos"].iloc[0]),
               MANUSCRIPT["density_matched_n_videos"], tol=0)
    side.check("density-matched features passing BH", int(dm["significant"].sum()),
               MANUSCRIPT["density_matched_n_significant"], tol=0)
    for feature, expected in MANUSCRIPT["top_features"].items():
        row = dm[dm["feature"] == feature]
        if row.empty:
            side.check(f"epsilon^2 {feature}", None, expected)
            continue
        side.check(f"epsilon^2 {feature}", round(float(row["epsilon_sq"].iloc[0]), 3),
                   expected, tol=0.0015)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def draw_confusion(ax, confusion: pd.DataFrame) -> None:
    counts = confusion.to_numpy(dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    n = len(fs.SPECIES_ORDER)
    for i in range(n):
        for j in range(n):
            colour = "white" if pct[i, j] > 0.55 else fs.NEUTRAL["ink"]
            ax.text(j, i - 0.10, f"{int(counts[i, j])}", ha="center", va="center",
                    fontsize=9, color=colour)
            ax.text(j, i + 0.26, f"{100 * pct[i, j]:.0f}%", ha="center", va="center",
                    fontsize=5.6, color=colour, alpha=0.85)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(fs.SPECIES_ORDER, rotation=30, ha="right", style="italic")
    ax.set_yticklabels(fs.SPECIES_ORDER, style="italic")
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title(f"Leave-one-video-out  ·  {int(np.trace(counts))}/{int(counts.sum())} correct",
                 pad=4)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


SCHEME_SHORT = {
    "Leave-one-video-out": "Leave one\nvideo out",
    "Leave-one-camera-model-out": "Leave one\ncamera model out",
    "Leave-one-recording-session-out": "Leave one\nsession out",
}


def draw_schemes(ax, regen: dict) -> None:
    schemes = list(MANUSCRIPT["schemes"])
    for i, scheme in enumerate(schemes):
        r = regen[scheme]
        parts = ax.violinplot([r["draws"]], positions=[i], widths=0.62,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(fs.NEUTRAL["light"])
            body.set_alpha(0.45)
            body.set_edgecolor(fs.NEUTRAL["mid"])
            body.set_linewidth(0.5)
        ax.hlines(r["null_p95"], i - 0.31, i + 0.31, color=fs.NEUTRAL["ink"],
                  lw=1.1, zorder=4)
        ax.plot([i], [r["null_mean"]], marker="_", ms=11, color=fs.NEUTRAL["mid"], zorder=4)
        ax.plot([i], [r["observed"]], marker="o", ms=7.5, color=fs.NEUTRAL["accent"],
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.text(i, r["observed"] + 0.045, f"{r['observed']:.3f}", ha="center",
                va="bottom", fontsize=7.2, color=fs.NEUTRAL["accent"])
        ax.text(i, 0.035, f"$p$ = {r['p_value']:.3f}", ha="center", va="bottom",
                fontsize=6.2, color=fs.NEUTRAL["mid"])
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([SCHEME_SHORT[s] for s in schemes], fontsize=6.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Observed vs its own permutation null", pad=4)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=fs.NEUTRAL["accent"], ms=6,
                   label="observed"),
        plt.Line2D([], [], marker="_", ls="", color=fs.NEUTRAL["ink"], ms=11,
                   label="null 95th pct"),
        plt.Line2D([], [], marker="_", ls="", color=fs.NEUTRAL["mid"], ms=11,
                   label="null mean"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.2)


def draw_controls(ax, controls: dict) -> None:
    rows: list[tuple[str, str, float]] = []
    for group, entries in CONTROL_GROUPS:
        for label, _key in entries:
            rows.append((group, label, controls[label]))
    rows = rows[::-1]                       # matplotlib y grows upward
    y = np.arange(len(rows))
    is_camera = np.array([label.startswith("Camera-model") for _g, label, _v in rows])
    values = np.array([v for _g, _l, v in rows])

    ax.hlines(y, 0, values, color=fs.NEUTRAL["faint"], lw=1.0, zorder=1)
    ax.scatter(values[~is_camera], y[~is_camera], s=34, color=fs.NEUTRAL["accent"],
               edgecolor="white", linewidth=0.5, zorder=3, marker="o")
    ax.scatter(values[is_camera], y[is_camera], s=38, color=fs.NEUTRAL["mid"],
               edgecolor="white", linewidth=0.5, zorder=3, marker="D")
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.022, yi, f"{v:.3f}", va="center", fontsize=6.6,
                color=fs.NEUTRAL["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels([label for _g, label, _v in rows], fontsize=6.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Balanced accuracy")

    # Group separators and headings. No single chance line is drawn: the camera
    # task has six classes, so the tasks do not share a nominal chance level.
    seen, boundaries = [], []
    for group, _label, _v in rows:
        if group not in seen:
            seen.append(group)
    for group in seen:
        idx = [i for i, (g, _l, _v) in enumerate(rows) if g == group]
        boundaries.append((group, min(idx), max(idx)))
    for _group, lo, _hi in boundaries[1:]:
        ax.axhline(lo - 0.5, color=fs.NEUTRAL["faint"], lw=0.6)
    for group, lo, hi in boundaries:
        ax.text(0.995, (lo + hi) / 2, group, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.0, color=fs.NEUTRAL["light"],
                style="italic")
    ax.axvline(0.25, color=fs.NEUTRAL["light"], lw=0.6, ls=":")
    ax.text(0.243, len(rows) - 2.0, "species chance 0.25", fontsize=5.6,
            color=fs.NEUTRAL["light"], ha="center", va="center", rotation=90)
    cam_y = float(y[is_camera][0])
    ax.plot([0.167], [cam_y], marker="|", ms=8, color=fs.NEUTRAL["light"], zorder=2)
    ax.text(0.167, cam_y - 0.38, "camera chance 0.167", fontsize=5.6,
            color=fs.NEUTRAL["light"], ha="center", va="top")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("Representation controls", pad=4)


def draw_effect_sizes(ax, dm: pd.DataFrame, n_top: int) -> None:
    top = dm.nlargest(n_top, "epsilon_sq").iloc[::-1]
    y = np.arange(len(top))
    ax.hlines(y, 0, top["epsilon_sq"], color=fs.NEUTRAL["faint"], lw=1.0, zorder=1)
    ax.scatter(top["epsilon_sq"], y, s=34, color=fs.NEUTRAL["accent"],
               edgecolor="white", linewidth=0.5, zorder=3)
    for yi, (_i, row) in zip(y, top.iterrows(), strict=True):
        mark = "*" if bool(row["significant"]) else ""
        ax.text(row["epsilon_sq"] + 0.012, yi, f"{row['epsilon_sq']:.3f}{mark}",
                va="center", fontsize=6.6, color=fs.NEUTRAL["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(f) for f in top["feature"]], fontsize=6.6)
    ax.set_xlim(0, max(1.0, float(top["epsilon_sq"].max()) * 1.16))
    ax.set_xlabel("Kruskal-Wallis $\\epsilon^2$ (between-species effect size)")
    n_sig = int(dm["significant"].sum())
    ax.set_title(f"Density-matched subset · {int(dm['n_videos'].iloc[0])} videos · "
                 f"{n_sig}/{len(dm)} pass BH", pad=4)


def draw_species_heatmap(ax, dm: pd.DataFrame, n_top: int) -> None:
    """Supplementary: row-standardised species medians for the top features."""
    top = dm.nlargest(n_top, "epsilon_sq").iloc[::-1]
    cols = [f"median[{s}]" for s in
            ["Aedes aegypti", "Aedes albopictus", "Anopheles stephensi", "Culex pipiens"]]
    values = top[cols].to_numpy(dtype=float)
    mu = values.mean(axis=1, keepdims=True)
    sd = values.std(axis=1, keepdims=True)
    z = np.divide(values - mu, sd, out=np.zeros_like(values), where=sd > 0)
    lim = float(np.abs(z).max())
    im = ax.imshow(z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(z[i, j]) > 0.75 * lim else fs.NEUTRAL["ink"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(fs.SPECIES_ORDER, rotation=30, ha="right", style="italic", fontsize=6.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feature_label(f) for f in top["feature"]], fontsize=6.4)
    ax.set_title("Species medians (row-standardised)", pad=4)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("row $z$-score", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_table(confusion: pd.DataFrame, regen: dict, controls: dict,
                dm: pd.DataFrame, n_top: int) -> pd.DataFrame:
    rows: list[dict] = []
    for true_label in fs.SPECIES_ORDER:
        for pred_label in fs.SPECIES_ORDER:
            rows.append({"panel": "a", "item": f"{true_label} -> {pred_label}",
                         "value": int(confusion.loc[true_label, pred_label]),
                         "quantity": "confusion count"})
    for scheme in MANUSCRIPT["schemes"]:
        r = regen[scheme]
        for quantity in ("observed", "null_mean", "null_p95", "p_value", "n_permutations"):
            rows.append({"panel": "b", "item": scheme, "value": r[quantity],
                         "quantity": quantity})
    for group, entries in CONTROL_GROUPS:
        for label, _key in entries:
            rows.append({"panel": "c", "item": label, "value": controls[label],
                         "quantity": "balanced accuracy", "group": group})
    for _i, row in dm.nlargest(n_top, "epsilon_sq").iterrows():
        rows.append({"panel": "d", "item": feature_label(row["feature"]),
                     "value": float(row["epsilon_sq"]), "quantity": "epsilon_sq",
                     "feature": row["feature"], "p_fdr": float(row["p_fdr"]),
                     "significant": bool(row["significant"]),
                     "median_Ae_aegypti": float(row["median[Aedes aegypti]"]),
                     "median_Ae_albopictus": float(row["median[Aedes albopictus]"]),
                     "median_An_stephensi": float(row["median[Anopheles stephensi]"]),
                     "median_Cx_pipiens": float(row["median[Culex pipiens]"])})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to configs/figures.yaml")
    args = ap.parse_args(argv)

    cfg_fig = fs.load_config(args.config)
    paths = fs.resolve_paths(cfg_fig)
    paths.require("species_analysis")
    opts = cfg_fig.get("fig04") or {}
    n_perm = int(opts.get("n_permutations", 200))
    n_top = int(opts.get("top_features", 8))

    fs.apply_style()
    side = fs.Sidecar(
        figure=FIGURE,
        description=("Species-associated information in the video-level trajectory "
                     "representation, its controls, and its acquisition-holdout limits."),
        hash_max_bytes=cfg_fig.get("hash_max_bytes"),
    )

    print(f"[{FIGURE}] loading stored analysis")
    clf_path = side.add_input(paths.species_analysis / "classification_results.json",
                              "stored classification results")
    vf_path = side.add_input(paths.species_analysis / "video_features.parquet",
                             "video-level feature table")
    uni_path = side.add_input(paths.species_analysis / "univariate_tests.parquet",
                              "univariate between-species tests")
    with open(clf_path) as fh:
        stored = json.load(fh)
    video_table = pd.read_parquet(vf_path)
    uni = pd.read_parquet(uni_path)
    features = list(stored["features"])

    if len(video_table) != stored["n_videos"]:
        raise SystemExit(f"video_features.parquet has {len(video_table)} rows but the stored "
                         f"analysis reports {stored['n_videos']} videos.")
    if video_table["run"].duplicated().any():
        raise SystemExit("Duplicate run identifiers in video_features.parquet.")
    missing = [f for f in features if f not in video_table.columns]
    if missing:
        raise SystemExit(f"Feature columns missing from video_features.parquet: {missing}")
    unexpected = sorted(set(video_table["meta_species"].map(fs.species_label))
                        - set(fs.SPECIES_ORDER))
    if unexpected:
        raise SystemExit(f"Unexpected species labels in the cohort: {unexpected}")

    cfg = Config()
    side.param("n_permutations", n_perm)
    side.param("logistic_l2", cfg.species_analysis.logistic_l2)
    side.param("random_seed", cfg.project.random_seed)
    side.param("top_features_shown", n_top)
    side.param("feature_columns", features)

    print(f"[{FIGURE}] regenerating LOVO predictions and {n_perm} permutation draws "
          f"per scheme (seed {cfg.project.random_seed})")
    regen = regenerate_predictions_and_nulls(video_table, features, cfg, n_perm)
    confusion = confusion_from_predictions(regen["y_true"], regen["y_pred_lovo"])

    controls = {}
    for _group, entries in CONTROL_GROUPS:
        for label, (section, key) in entries:
            controls[label] = float(stored[section][key])

    dm = uni[uni["variant"] == "density_matched"].copy()
    if dm.empty:
        raise SystemExit("univariate_tests.parquet has no 'density_matched' variant rows.")

    print(f"[{FIGURE}] validating against the manuscript")
    validate(stored, regen, confusion, controls, uni, side)

    ranked = dm.nlargest(n_top, "epsilon_sq")["feature"].tolist()
    extra = [f for f in ranked if f not in MANUSCRIPT["top_features"]]
    if extra:
        side.note(
            "panel (d) shows the true top-"
            f"{n_top} features by epsilon^2; {', '.join(feature_label(f) for f in extra)} "
            "rank inside it but are absent from the manuscript's eight-row table."
        )
    if stored["track_level"].get("confusion_per_video_vote", {}) and \
            len(stored["track_level"]["confusion_per_video_vote"]) == 1:
        side.note(
            "the individual-track majority vote is degenerate - every video is voted "
            "into one class - which is what its 0.250 balanced accuracy reflects."
        )
    side.note("species labels come from acquisition metadata, never from detector class "
              "predictions; movement features are in apparent body lengths, not physical units.")
    side.note("classification_results.json stores video_level.confusion column-major "
              "({predicted: {true: count}}), a consequence of DataFrame.to_dict()'s default "
              "orientation; read row-major it transposes two cells. This figure reads it "
              "column-major, which reproduces the published confusion table.")

    print(f"[{FIGURE}] drawing")
    fig = plt.figure(figsize=(7.2, 6.8))
    grid = gridspec.GridSpec(2, 2, figure=fig, left=0.115, right=0.975, top=0.945,
                             bottom=0.105, wspace=0.66, hspace=0.58,
                             width_ratios=[1.0, 1.10])
    ax_a = fig.add_subplot(grid[0, 0])
    draw_confusion(ax_a, confusion)
    fs.panel_label(ax_a, "a", dx=-0.30)
    ax_b = fig.add_subplot(grid[0, 1])
    draw_schemes(ax_b, regen)
    fs.panel_label(ax_b, "b", dx=-0.17)
    ax_c = fig.add_subplot(grid[1, 0])
    draw_controls(ax_c, controls)
    fs.panel_label(ax_c, "c", dx=-0.60, dy=1.09)
    ax_d = fig.add_subplot(grid[1, 1])
    draw_effect_sizes(ax_d, dm, n_top)
    fs.panel_label(ax_d, "d", dx=-0.44, dy=1.09)
    fig.text(0.5, 0.012,
             "* significant after Benjamini-Hochberg correction. Movement variables are "
             "in apparent body lengths (BL); species labels come from acquisition metadata.",
             ha="center", fontsize=6.2, color=fs.NEUTRAL["mid"])

    panels = {
        "a": lambda ax: draw_confusion(ax, confusion),
        "b": lambda ax: draw_schemes(ax, regen),
        "c": lambda ax: draw_controls(ax, controls),
        "d": lambda ax: draw_effect_sizes(ax, dm, n_top),
        # Supplementary: kept out of the main figure so panel (d) stays readable.
        "supp_species_medians": lambda ax: draw_species_heatmap(ax, dm, n_top),
    }
    panel_size = {"a": (3.1, 2.9), "b": (3.6, 2.9), "c": (4.2, 2.9),
                  "d": (4.0, 2.9), "supp_species_medians": (4.2, 3.2)}

    fs.save_figure(fig, FIGURE, side,
                   build_table(confusion, regen, controls, dm, n_top),
                   paths.output_dir, panels=panels, panel_size=panel_size)
    fs.report_checks(side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
