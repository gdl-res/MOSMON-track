"""Does the choice of association algorithm change the biology? (§21-§24)

This is the part of the benchmark that matters most to the manuscript. Tracking
metrics say how the arms differ; these functions say whether that difference
survives all the way through to the population-level descriptors and the species
classification the paper's claims rest on.

Nothing here re-implements the downstream analysis. Each arm's run folders are
handed to the existing :func:`mosmon_tracking.species_analysis.species_analysis`
unchanged, which is the only way to guarantee §19's requirement that downstream
methodology does not drift between arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Descriptor agreement bands used to label how tracker-sensitive a feature is.
STABILITY_BANDS = [
    (0.95, "tracker-stable"),
    (0.80, "moderately tracker-sensitive"),
    (0.00, "strongly tracker-sensitive"),
]


# --------------------------------------------------------------------------- #
# 1. Cohorts (§22)
# --------------------------------------------------------------------------- #
def arm_cohorts(benchmark_dir: str | Path, cfg=None) -> pd.DataFrame:
    """Per-arm QC cohort, using the pipeline's own gate unchanged.

    Tracker choice can move a video across the QC line, so the eligible set is
    not a constant. Both cohorts required by §22 are derived from this table:
    the common cohort (videos every arm passes -- the fairest comparison) and
    each arm's own cohort (which shows operational yield).
    """
    from ..config import Config
    from ..species_analysis import build_cohort
    from .benchmark_utils import find_arm_dirs

    cfg = cfg or Config()
    benchmark_dir = Path(benchmark_dir)
    frames = []
    for arm_dir in find_arm_dirs(benchmark_dir):
        cohort = build_cohort(arm_dir, cfg)
        if cohort.empty:
            continue
        cohort = cohort.copy()
        cohort["arm"] = arm_dir.name
        frames.append(cohort)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def common_cohort(cohorts: pd.DataFrame) -> list[str]:
    """Videos in the cohort for *every* arm."""
    if cohorts.empty or "in_cohort" not in cohorts:
        return []
    key = "video" if "video" in cohorts.columns else "run"
    per_arm = [set(g.loc[g["in_cohort"], key]) for _, g in cohorts.groupby("arm")]
    return sorted(set.intersection(*per_arm)) if per_arm else []


def cohort_summary(cohorts: pd.DataFrame) -> pd.DataFrame:
    """Per-arm cohort size and why videos were excluded -- never silently."""
    if cohorts.empty:
        return pd.DataFrame()
    rows = []
    common = set(common_cohort(cohorts))
    key = "video" if "video" in cohorts.columns else "run"
    for arm, g in cohorts.groupby("arm"):
        reasons = (g.loc[~g["in_cohort"], "excluded_reason"]
                   .str.replace(r"=[\d.]+", "=<n>", regex=True).value_counts().to_dict())
        rows.append({
            "arm": arm,
            "n_runs": int(len(g)),
            "n_in_cohort": int(g["in_cohort"].sum()),
            "n_in_common_cohort": int(g.loc[g["in_cohort"], key].isin(common).sum()),
            "exclusion_reasons": json.dumps(reasons),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Descriptor stability (§24)
# --------------------------------------------------------------------------- #
def load_video_features(species_analysis_dirs: dict[str, str | Path]) -> pd.DataFrame:
    """Stack every arm's ``video_features.parquet`` into one long table."""
    from ..video_io import load_table

    frames = []
    for arm, d in species_analysis_dirs.items():
        t = load_table(Path(d) / "video_features.parquet")
        if t is None or t.empty:
            continue
        t = t.copy()
        t["arm"] = arm
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def descriptor_stability(video_features: pd.DataFrame, feature_cols: list[str],
                         key: str = "run") -> pd.DataFrame:
    """How much each video-level descriptor moves when the tracker changes.

    Restricted to videos present for every arm, so a descriptor cannot look
    stable merely because the arms were scored on different videos.

    Three complementary views, because they fail differently:

    * **Spearman rho** -- is the *ordering* of videos preserved? This is what
      matters for a classifier, which only ever sees relative values.
    * **Normalised absolute difference** -- is the *value* preserved? Matters if
      the descriptor is reported as a biological quantity.
    * **Rank stability** -- the mean absolute shift in a video's rank position.
    """
    from scipy import stats

    if video_features.empty:
        return pd.DataFrame()
    arms = sorted(video_features["arm"].unique())
    shared = set.intersection(*[set(g[key]) for _, g in video_features.groupby("arm")])
    if not shared:
        return pd.DataFrame()
    vf = video_features[video_features[key].isin(shared)]

    rows = []
    for feat in feature_cols:
        if feat not in vf.columns:
            continue
        wide = vf.pivot_table(index=key, columns="arm", values=feat, aggfunc="first")
        rhos, nads, rank_shifts = [], [], []
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                if a not in wide or b not in wide:
                    continue
                d = wide[[a, b]].dropna()
                if len(d) < 3 or d[a].nunique() < 2 or d[b].nunique() < 2:
                    continue
                rho, _ = stats.spearmanr(d[a], d[b])
                rhos.append(rho)
                scale = np.nanmedian(np.abs(pd.concat([d[a], d[b]])))
                if scale and np.isfinite(scale) and scale > 0:
                    nads.append(float(np.nanmedian(np.abs(d[a] - d[b]) / scale)))
                rank_shifts.append(float(np.abs(d[a].rank() - d[b].rank()).mean()))
        if not rhos:
            continue
        rho_mean = float(np.nanmean(rhos))
        rows.append({
            "feature": feat,
            "n_videos": int(len(shared)),
            "n_pairs": int(len(rhos)),
            "spearman_mean": rho_mean,
            "spearman_min": float(np.nanmin(rhos)),
            "abs_norm_diff_median": float(np.nanmedian(nads)) if nads else np.nan,
            "rank_shift_mean": float(np.nanmean(rank_shifts)) if rank_shifts else np.nan,
            "stability": _band(rho_mean),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("spearman_mean", ascending=False).reset_index(drop=True) \
        if not out.empty else out


def _band(rho: float) -> str:
    for cut, label in STABILITY_BANDS:
        if np.isfinite(rho) and rho >= cut:
            return label
    return "strongly tracker-sensitive"


# --------------------------------------------------------------------------- #
# 3. Classification comparison (§21, §23)
# --------------------------------------------------------------------------- #
def classification_table(species_analysis_dirs: dict[str, str | Path]) -> pd.DataFrame:
    """One row per arm from each ``classification_results.json``.

    Every column is a balanced accuracy under a group-disjoint scheme with its
    own permutation null, exactly as the single-tracker analysis reports them.
    Reading them straight out of the arm's own result file is what keeps the
    downstream methodology identical across arms.
    """
    rows = []
    for arm, d in species_analysis_dirs.items():
        path = Path(d) / "classification_results.json"
        if not path.exists():
            continue
        clf = json.loads(path.read_text())
        v = clf.get("video_level", {})
        rows.append({
            "arm": arm,
            "n_videos": v.get("n_videos"),
            "chance": clf.get("chance_balanced_accuracy"),
            "video_lovo_ba": v.get("balanced_accuracy_lovo"),
            "camera_out_ba": clf.get("leave_one_camera_out", {}).get("balanced_accuracy"),
            "session_out_ba": clf.get("leave_one_session_out", {}).get("balanced_accuracy"),
            "density_residualised_ba":
                clf.get("density_residualised", {}).get("balanced_accuracy_lovo"),
            "density_only_ba":
                clf.get("density_only_baseline", {}).get("balanced_accuracy_lovo"),
            "camera_control_ba":
                clf.get("camera_control", {}).get("balanced_accuracy_lovo"),
            "track_level_ba":
                clf.get("track_level", {}).get("balanced_accuracy_per_track_lovo"),
            "track_vote_ba":
                clf.get("track_level", {}).get("balanced_accuracy_per_video_vote"),
            "permutation_null_mean":
                clf.get("permutation_null", {}).get("null_mean"),
            "permutation_p": clf.get("permutation_null", {}).get("p_value"),
        })
    return pd.DataFrame(rows)


def downstream_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """The §21 per-arm trajectory summary, aggregated over videos."""
    cols = ["median_track_duration_s", "p95_track_duration_s", "n_tracks_clean",
            "n_tracks_ge_3s", "interpolation_fraction", "fragments_per_object",
            "duplicate_track_proxy", "duplicate_id_proxy"]
    present = [c for c in cols if c in metrics.columns]
    agg = metrics.groupby("arm")[present].median().reset_index()
    agg["n_videos"] = metrics.groupby("arm")["video_id"].nunique().to_numpy()
    return agg
