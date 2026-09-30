"""Tests for the species-discrimination analysis.

Everything here runs on tiny synthetic frames — no model weights, no videos.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mosmon_tracking.config import Config, SpeciesAnalysisConfig
from mosmon_tracking.species_analysis import (
    autocorrelation,
    balanced_accuracy,
    benjamini_hochberg,
    build_cohort,
    cv_predict,
    estimate_split_x_fraction,
    residualize,
    track_shape_features,
    univariate_tests,
    video_feature_table,
)
from mosmon_tracking.track_postprocess import add_compartment_species, classify_compartment


def _track(n=60, step=2.0, box=20.0, fps=30.0, jitter=0.0, seed=0):
    """A synthetic track: constant-speed motion with a fixed bbox size."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    cx = np.cumsum(np.full(n, step)) + rng.normal(0, jitter, n)
    cy = np.zeros(n) + rng.normal(0, jitter, n)
    return pd.DataFrame({
        "track_id": 1, "frame_idx": np.arange(n), "time_s": t,
        "cx": cx, "cy": cy, "cx_norm": cx / cx.max(), "cy_norm": 0.5,
        "w": box, "h": box / 2.0,
        "speed_px_s": np.r_[np.nan, np.hypot(np.diff(cx), np.diff(cy)) * fps],
        "turn_angle_rad": 0.0,
        "nearest_neighbor_distance_px": 100.0,
    })


# --------------------------------------------------------------------------- #
# Scale normalisation and shape features
# --------------------------------------------------------------------------- #
def test_speed_is_normalised_by_body_length():
    opts = SpeciesAnalysisConfig()
    # box 20x10 -> body length 20 px; 2 px/frame at 30 fps = 60 px/s = 3 BL/s
    feats = track_shape_features(_track(step=2.0, box=20.0, fps=30.0), opts)
    assert feats["body_length_px"] == pytest.approx(20.0)
    assert feats["mean_speed_bl_s"] == pytest.approx(3.0, rel=1e-6)


def test_body_length_normalisation_is_resolution_invariant():
    """The same motion filmed at 2x resolution must give the same BL/s."""
    opts = SpeciesAnalysisConfig()
    small = track_shape_features(_track(step=2.0, box=20.0), opts)
    big = track_shape_features(_track(step=4.0, box=40.0), opts)
    assert big["mean_speed_bl_s"] == pytest.approx(small["mean_speed_bl_s"])
    assert big["path_length_bl"] == pytest.approx(small["path_length_bl"])
    assert big["body_length_px"] == pytest.approx(2 * small["body_length_px"])


def test_msd_alpha_ballistic_vs_diffusive():
    opts = SpeciesAnalysisConfig()
    ballistic = track_shape_features(_track(n=200, step=2.0, jitter=0.0), opts)
    assert ballistic["msd_alpha"] == pytest.approx(2.0, abs=0.05)

    rng = np.random.default_rng(3)
    n = 600
    cx = np.cumsum(rng.normal(0, 1, n))
    cy = np.cumsum(rng.normal(0, 1, n))
    walk = _track(n=n)
    walk["cx"], walk["cy"] = cx, cy
    assert track_shape_features(walk, opts)["msd_alpha"] == pytest.approx(1.0, abs=0.25)


def test_autocorrelation_of_constant_signal_is_nan_not_one():
    assert np.all(np.isnan(autocorrelation(np.ones(20), 3)))


# --------------------------------------------------------------------------- #
# Cohort hygiene
# --------------------------------------------------------------------------- #
def _write_run(root, name, species, sha, dup=0.01, cov=99.0):
    d = root / name
    (d).mkdir(parents=True)
    (d / "video_summary.json").write_text(json.dumps({
        "video_info": {"filename": f"{name}.mp4", "meta_species": species,
                       "meta_camera_model": "cam1", "width": 1920, "height": 1080},
        "model_settings": {"tracker": "botsort", "frame_stride": 1},
        "qc": {"pct_frames_with_detections": cov, "duplicate_track_proxy": dup},
        "behaviour": {"total_tracks": 10, "mean_active_tracks_per_frame": 5.0},
    }))
    (d / "qc_report.json").write_text(json.dumps(
        {"pct_frames_with_detections": cov, "duplicate_track_proxy": dup, "warnings": []}))
    (d / "provenance.json").write_text(json.dumps(
        {"inputs": [{"role": "source-video", "sha256": sha}]}))
    return d


def test_cohort_drops_duplicates_conflicts_and_qc_failures(tmp_path):
    _write_run(tmp_path, "keep_a", "Aedes aegypti", "sha_unique_1")
    # identical footage, consistent labels -> keep one
    _write_run(tmp_path, "dupe_1", "Culex pipiens", "sha_dup")
    _write_run(tmp_path, "dupe_2", "Culex pipiens", "sha_dup")
    # identical footage, contradictory species -> drop both
    _write_run(tmp_path, "conflict_1", "Aedes albopictus", "sha_conflict")
    _write_run(tmp_path, "conflict_2", "Culex pipiens", "sha_conflict")
    # QC failures
    _write_run(tmp_path, "crowded", "Aedes aegypti", "sha_unique_2", dup=0.4)
    _write_run(tmp_path, "sparse", "Aedes aegypti", "sha_unique_3", cov=80.0)

    cohort = build_cohort(tmp_path, Config()).set_index("run")
    assert cohort.loc["keep_a", "in_cohort"]
    assert cohort.loc[["dupe_1", "dupe_2"], "in_cohort"].sum() == 1
    assert not cohort.loc[["conflict_1", "conflict_2"], "in_cohort"].any()
    assert "conflicting" in cohort.loc["conflict_1", "excluded_reason"]
    assert not cohort.loc["crowded", "in_cohort"]
    assert not cohort.loc["sparse", "in_cohort"]
    # nothing disappears silently: every excluded run keeps a reason
    assert (cohort.loc[~cohort["in_cohort"], "excluded_reason"] != "").all()


def test_qc_gate_can_be_disabled(tmp_path):
    _write_run(tmp_path, "crowded", "Aedes aegypti", "sha1", dup=0.4)
    cfg = Config()
    cfg.species_analysis.apply_qc_gate = False
    assert build_cohort(tmp_path, cfg)["in_cohort"].all()


# --------------------------------------------------------------------------- #
# Video-disjoint evaluation
# --------------------------------------------------------------------------- #
def test_cv_predict_is_group_disjoint():
    """A feature that only identifies the *group* must not predict its label.

    Each group gets a unique constant feature value; labels are arbitrary. A
    leaky split would memorise group -> label and score ~1.0.
    """
    rng = np.random.default_rng(0)
    runs, y, x = [], [], []
    for g in range(12):
        for _ in range(10):
            runs.append(f"v{g}")
            y.append(["a", "b", "c"][g % 3])
            x.append(g + rng.normal(0, 0.01))
    pred, _, _ = cv_predict(np.array(x).reshape(-1, 1), np.array(y),
                            np.array(runs), l2=0.05, max_iter=100)
    assert balanced_accuracy(np.array(y), pred) < 0.6


def test_cv_predict_recovers_a_real_group_level_signal():
    rng = np.random.default_rng(1)
    runs, y, x = [], [], []
    for g in range(18):
        cls = ["a", "b", "c"][g % 3]
        mu = {"a": 0.0, "b": 5.0, "c": 10.0}[cls]
        for _ in range(10):
            runs.append(f"v{g}")
            y.append(cls)
            x.append(rng.normal(mu, 0.5))
    y = np.array(y)
    pred, _, _ = cv_predict(np.array(x).reshape(-1, 1), y, np.array(runs), 0.05, 200)
    observed = balanced_accuracy(y, pred)
    assert observed > 0.8
    # the same pipeline on permuted labels must not reproduce it
    perm = {g: c for g, c in zip(sorted(set(runs)),
                                 rng.permutation([["a", "b", "c"][i % 3] for i in range(18)]),
                                 strict=True)}
    yp = np.array([perm[r] for r in runs])
    pred_p, _, _ = cv_predict(np.array(x).reshape(-1, 1), yp, np.array(runs), 0.05, 200)
    assert balanced_accuracy(yp, pred_p) < observed


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_is_monotone_and_nan_safe():
    p = np.array([0.001, 0.02, 0.3, np.nan, 0.7])
    adj = benjamini_hochberg(p)
    assert np.isnan(adj[3])
    finite = adj[np.isfinite(adj)]
    assert np.all(np.diff(finite[np.argsort(p[np.isfinite(p)])]) >= -1e-12)
    assert np.all(finite <= 1.0)
    assert adj[0] == pytest.approx(0.004)


def test_residualize_removes_a_linear_covariate():
    x = np.linspace(1, 10, 20)
    y = 3 * x + 5
    assert np.allclose(residualize(y, x), 0.0, atol=1e-9)


def test_univariate_tests_report_all_three_variants():
    rng = np.random.default_rng(2)
    n = 12
    table = pd.DataFrame({
        "run": [f"v{i}" for i in range(2 * n)],
        "meta_species": ["a"] * n + ["b"] * n,
        # separates species but only through density
        "confounded_med": np.r_[rng.normal(1, 0.1, n), rng.normal(5, 0.1, n)],
        "density": np.r_[np.linspace(5, 25, n), np.linspace(5, 25, n)],
    })
    table["confounded_med"] = 2.0 * np.log10(table["density"]) + rng.normal(0, 0.01, 2 * n)
    out = univariate_tests(table, ["confounded_med"])
    assert set(out["variant"]) == {"raw", "density_adjusted", "density_matched"}
    adjusted = out[out["variant"] == "density_adjusted"].iloc[0]
    assert adjusted["p"] > 0.05  # the difference was density all along


# --------------------------------------------------------------------------- #
# Dual-container compartments
# --------------------------------------------------------------------------- #
def test_classify_compartment_sides_and_default():
    cx = np.array([10.0, 40.0, 60.0, 90.0])
    labels = classify_compartment(cx, width=100.0, split_x_fraction=0.5)
    assert list(labels) == ["left", "left", "right", "right"]
    # no configured split -> nothing is guessed
    assert set(classify_compartment(cx, 100.0, None)) == {"unassigned"}


def test_add_compartment_species_uses_filename_metadata():
    name = ("20251008_mosmon_iss_aedes_albopictusSX_aegyptiDX_stage3and4_goprohero13black_"
            "video_5dot7Kwidescreen_lenslinear_lightartcold_cameraposition4_"
            "big2boxcontainer_depth45mm.MP4")
    df = pd.DataFrame({"video_name": name, "cx": [10.0, 90.0], "cy": [5.0, 5.0],
                       "frame_width": 100.0, "frame_height": 50.0, "track_id": [1, 2]})
    cfg = Config()
    cfg.regions.dual_container_split_x_fraction = 0.5
    out = add_compartment_species(df, cfg)
    assert list(out["compartment"]) == ["left", "right"]
    assert out["species_compartment"].tolist() == ["Aedes albopictus", "Aedes aegypti"]


def test_add_compartment_species_assigns_nothing_without_a_split():
    df = pd.DataFrame({"video_name": "x.mp4", "cx": [1.0], "cy": [1.0],
                       "frame_width": 10.0, "frame_height": 10.0, "track_id": [1]})
    out = add_compartment_species(df, Config())
    assert out["compartment"].iloc[0] == "unassigned"
    assert out["species_compartment"].iloc[0] is None


def test_estimate_split_x_fraction_finds_the_gap(tmp_path):
    occ = np.ones((8, 100))
    occ[:, 48:53] = 0.0  # empty divider band
    (tmp_path / "heatmaps").mkdir()
    np.save(tmp_path / "heatmaps" / "occupancy.npy", occ)
    assert estimate_split_x_fraction(tmp_path) == pytest.approx(0.505, abs=0.01)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_video_feature_table_is_one_row_per_video():
    tracks = pd.DataFrame({
        "run": ["r1"] * 5 + ["r2"] * 5,
        "mean_speed_bl_s": np.arange(10, dtype=float),
        "straightness": np.linspace(0, 1, 10),
    })
    cohort = pd.DataFrame({"run": ["r1", "r2"], "video": ["a.mp4", "b.mp4"],
                           "density": [5.0, 50.0], "meta_species": ["a", "b"]})
    out = video_feature_table(tracks, cohort,
                              feature_cols=["mean_speed_bl_s", "straightness"])
    assert len(out) == 2
    assert out.loc[out["run"] == "r1", "mean_speed_bl_s_med"].iloc[0] == 2.0
    assert out.loc[out["run"] == "r1", "n_tracks_used"].iloc[0] == 5
