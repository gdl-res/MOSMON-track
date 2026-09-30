"""Unit tests for the benchmark analysis, evaluation and reproduction modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosmon_tracking.benchmark.analyze_density import (
    bootstrap_ci,
    density_bins,
    density_trend,
    paired_arm_comparison,
    summarise_by_arm,
)
from mosmon_tracking.benchmark.analyze_downstream_stability import (
    cohort_summary,
    common_cohort,
    descriptor_stability,
)
from mosmon_tracking.benchmark.benchmark_utils import (
    check_detection_identity,
    check_detection_table,
    check_track_table,
    find_arm_dirs,
)
from mosmon_tracking.benchmark.evaluate import per_arm_agreement
from mosmon_tracking.benchmark.reproduction import compare_track_tables, verdict


def _metrics(n_videos: int = 10, arms=("a", "b")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for v in range(n_videos):
        dens = 5.0 * (1.6 ** v)  # spans a wide range, like the real corpus
        for k, arm in enumerate(arms):
            rows.append({
                "arm": arm, "video_id": f"v{v}",
                "detections_per_frame": dens,
                "duration_min": 5.0,
                "n_tracks_raw": int(dens * (2 + k)),
                "median_track_duration_s": 4.0 - 0.2 * k + rng.normal(0, 0.05),
                "fragments_per_object": (2 + k) + 0.4 * np.log10(dens),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# density and pairing
# --------------------------------------------------------------------------- #
def test_density_bins_are_shared_across_arms():
    # A video must land in the same bin whichever arm is being scored, or the
    # arm under test would be choosing its own difficulty stratification.
    t = density_bins(_metrics(8, arms=("a", "b")))
    per_video = t.groupby("video_id")["density_bin"].nunique()
    assert (per_video == 1).all()


def test_density_bins_are_ordered_low_to_very_high():
    t = density_bins(_metrics(12))
    order = (t.sort_values("detections_per_frame")["density_bin"]
             .dropna().astype(str).unique().tolist())
    assert order == ["Low", "Medium", "High", "Very high"]


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert lo <= mean <= hi
    assert mean == pytest.approx(3.0)


def test_bootstrap_ci_handles_degenerate_input():
    assert all(np.isnan(v) for v in bootstrap_ci(np.array([])))
    assert bootstrap_ci(np.array([2.0])) == (2.0, 2.0, 2.0)


def test_summarise_by_arm_reports_one_row_per_arm():
    out = summarise_by_arm(_metrics(6), ["median_track_duration_s"])
    assert sorted(out["arm"]) == ["a", "b"]
    assert set(out["n_videos"]) == {6}


def test_paired_comparison_uses_the_video_as_the_replicate():
    t = _metrics(10)
    out = paired_arm_comparison(t, "fragments_per_object", reference="a")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_videos"] == 10
    # arm b was constructed to fragment more than arm a on every video
    assert row["mean_diff"] > 0
    assert row["rank_biserial"] == pytest.approx(1.0)
    assert row["lower_is_better"]


def test_density_trend_recovers_a_planted_slope():
    t = _metrics(12)
    out = density_trend(t, "fragments_per_object")
    # fragments_per_object was built as 0.4*log10(density) + offset
    assert np.allclose(out["slope_per_log10_density"], 0.4, atol=1e-6)
    assert (out["spearman_rho"] > 0.99).all()


# --------------------------------------------------------------------------- #
# fairness checks
# --------------------------------------------------------------------------- #
def test_detection_identity_passes_when_every_arm_saw_the_same_pool():
    fp = {"n_detections": 10, "n_frames": 2, "frame_hash": "a", "box_hash": "b"}
    assert check_detection_identity({"x": fp, "y": dict(fp)}).ok


def test_detection_identity_fails_on_a_differing_box_hash():
    fp = {"n_detections": 10, "n_frames": 2, "frame_hash": "a", "box_hash": "b"}
    rep = check_detection_identity({"x": fp, "y": {**fp, "box_hash": "c"}})
    assert not rep.ok
    assert any("box coordinates" in f for f in rep.failed)


def test_track_table_check_flags_duplicate_frame_id_pairs():
    t = pd.DataFrame({"frame_idx": [0, 0], "track_id": [1, 1], "time_s": [0.0, 0.0],
                      "x1": [0, 0], "y1": [0, 0], "x2": [10, 10], "y2": [10, 10],
                      "frame_width": [100, 100], "frame_height": [100, 100]})
    rep = check_track_table(t, "arm")
    assert not rep.ok
    assert any("duplicate" in f for f in rep.failed)


def test_track_table_check_flags_degenerate_boxes():
    t = pd.DataFrame({"frame_idx": [0, 2], "track_id": [1, 1], "time_s": [0.0, 0.1],
                      "x1": [0, 0], "y1": [0, 0], "x2": [10, 0], "y2": [10, 10],
                      "frame_width": [100, 100], "frame_height": [100, 100]})
    assert any("non-positive" in f for f in check_track_table(t, "arm").failed)


def test_detection_table_check_detects_max_det_truncation():
    # A spike at exactly max_det is the signature of silent truncation, and it
    # bites hardest in the dense regime a crowding analysis is about.
    rows = [{"processed_frame": 0, "x1": i, "y1": 0, "x2": i + 5, "y2": 5}
            for i in range(300)]
    rows += [{"processed_frame": 1, "x1": 0, "y1": 0, "x2": 5, "y2": 5}]
    rep = check_detection_table(pd.DataFrame(rows), max_det=300)
    assert not rep.ok
    assert any("truncated" in f for f in rep.failed)


def test_find_arm_dirs_excludes_benchmark_bookkeeping(tmp_path):
    for name in ("botsort", "ocsort", "detections_canonical", "evaluation",
                 "reports", "_agreement"):
        (tmp_path / name).mkdir()
    assert [p.name for p in find_arm_dirs(tmp_path)] == ["botsort", "ocsort"]


# --------------------------------------------------------------------------- #
# reproduction gate
# --------------------------------------------------------------------------- #
def _tracks(offset: float = 0.0, ids=(1, 1, 2, 2)) -> pd.DataFrame:
    return pd.DataFrame({
        "frame_idx": [0, 2, 0, 2], "track_id": list(ids),
        "time_s": [0.0, 0.1, 0.0, 0.1],
        "x1": [10.0 + offset, 12.0 + offset, 50.0 + offset, 52.0 + offset],
        "y1": [10.0, 11.0, 50.0, 51.0],
        "x2": [30.0 + offset, 32.0 + offset, 70.0 + offset, 72.0 + offset],
        "y2": [30.0, 31.0, 70.0, 71.0],
    })


def test_reproduction_passes_on_an_identical_table():
    cmp = compare_track_tables(_tracks(), _tracks())
    ok, msg = verdict(cmp)
    assert ok, msg
    assert cmp["ari"] == pytest.approx(1.0)


def test_reproduction_tolerates_subpixel_kalman_drift():
    # Trackers emit Kalman states, not the detections they consumed, so two runs
    # that made identical decisions still differ in the last bits.
    cmp = compare_track_tables(_tracks(offset=0.09), _tracks())
    assert verdict(cmp)[0]
    assert cmp["box_delta_px_median"] == pytest.approx(0.09, abs=1e-6)


def test_reproduction_fails_when_grouping_differs():
    cmp = compare_track_tables(_tracks(ids=(1, 2, 3, 4)), _tracks())
    ok, msg = verdict(cmp)
    assert not ok
    assert "grouping" in msg


def test_reproduction_fails_when_the_box_sets_differ():
    ref = _tracks()
    cmp = compare_track_tables(_tracks(offset=50.0), ref)
    assert not verdict(cmp)[0]


# --------------------------------------------------------------------------- #
# downstream
# --------------------------------------------------------------------------- #
def test_common_cohort_is_the_intersection_across_arms():
    c = pd.DataFrame({
        "arm": ["a", "a", "a", "b", "b", "b"],
        "run": ["v1", "v2", "v3", "v1", "v2", "v3"],
        "in_cohort": [True, True, False, True, False, True],
        "excluded_reason": ["", "", "dup=0.2 > 0.15", "", "coverage=90% < 95%", ""],
    })
    assert common_cohort(c) == ["v1"]
    s = cohort_summary(c)
    assert set(s["n_in_cohort"]) == {2}
    assert set(s["n_in_common_cohort"]) == {1}


def test_descriptor_stability_ranks_a_stable_feature_above_a_scrambled_one():
    rng = np.random.default_rng(0)
    base = rng.normal(size=20)
    rows = []
    for arm, noise in (("a", 0.0), ("b", 0.02), ("c", 0.02)):
        for i, v in enumerate(base):
            rows.append({"arm": arm, "run": f"v{i}",
                         "stable": v + rng.normal(0, noise),
                         "scrambled": rng.normal()})
    out = descriptor_stability(pd.DataFrame(rows), ["stable", "scrambled"])
    assert out.iloc[0]["feature"] == "stable"
    assert out.iloc[0]["spearman_mean"] > 0.9
    assert out.iloc[0]["stability"] == "tracker-stable"
    assert out.iloc[-1]["feature"] == "scrambled"


def test_descriptor_stability_uses_only_videos_every_arm_has():
    # Otherwise a descriptor could look stable because the arms were scored on
    # different videos.
    rows = [{"arm": "a", "run": "v1", "f": 1.0}, {"arm": "a", "run": "v2", "f": 2.0},
            {"arm": "a", "run": "v3", "f": 3.0}, {"arm": "b", "run": "v1", "f": 1.0},
            {"arm": "b", "run": "v2", "f": 2.0}, {"arm": "b", "run": "v3", "f": 3.0},
            {"arm": "b", "run": "v4", "f": 9.0}]
    out = descriptor_stability(pd.DataFrame(rows), ["f"])
    assert out.iloc[0]["n_videos"] == 3


def test_per_arm_agreement_averages_both_directions_of_each_pair():
    ag = pd.DataFrame({"arm_a": ["x", "x"], "arm_b": ["y", "z"],
                       "ari": [0.5, 0.9], "identity_f1": [0.5, 0.9],
                       "video_id": ["v1", "v1"]})
    out = per_arm_agreement(ag).set_index("arm")
    assert out.loc["x", "identity_agreement_mean"] == pytest.approx(0.7)
    assert out.loc["y", "identity_agreement_mean"] == pytest.approx(0.5)
    assert out.loc["z", "n_pairs"] == 1
