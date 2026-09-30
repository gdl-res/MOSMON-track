"""Unit tests for the benchmark tracker adapters and the shared detection replay.

Nothing here needs model weights, a GPU or a video: the adapters are driven with
hand-written detection arrays, which is the whole point of a fixed-detector
benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosmon_tracking.benchmark.detections_io import (
    apply_conf_floor,
    detection_fingerprint,
    iter_frame_detections,
)
from mosmon_tracking.benchmark.identity_agreement import (
    adjusted_rand_index,
    compare_assignments,
    identity_f1,
    pairwise_agreement,
    v_measure,
)
from mosmon_tracking.benchmark.run_tracker_benchmark import replay
from mosmon_tracking.trackers import build_arm

BYTETRACK = "configs/tracker_bytetrack.yaml"
BYTETRACK_IOU = "configs/tracker_bytetrack_iou.yaml"
OCSORT = "configs/tracker_ocsort.yaml"


def _detections(n_frames: int = 20, n_objects: int = 3, conf: float = 0.9) -> pd.DataFrame:
    """Objects drifting at constant velocity - trivially trackable."""
    start = np.array([[100.0, 100.0], [400.0, 300.0], [800.0, 150.0]])[:n_objects]
    rows = []
    for p in range(n_frames):
        for _i, (x0, y0) in enumerate(start):
            x, y = x0 + 6.0 * p, y0 + 2.0 * p
            rows.append({
                "video_id": "v", "video_path": "v.mp4", "video_name": "v.mp4",
                "model_name": "m", "model_path": "m.pt",
                "source_frame": p * 2, "processed_frame": p, "time_s": p / 15.0,
                "x1": x, "y1": y, "x2": x + 40, "y2": y + 40,
                "confidence": conf, "predicted_class": 0, "class_name": "larva",
                "frame_width": 1920, "frame_height": 1080,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# detection cache
# --------------------------------------------------------------------------- #
def test_frames_without_detections_are_still_yielded():
    # A frame with no detections is what ages a lost track out. Skipping it
    # would silently lengthen every track that spans a gap.
    det = _detections(n_frames=4)
    det = det[det["processed_frame"] != 2]
    yielded = list(iter_frame_detections(det))
    assert [f.processed_frame for f in yielded] == [0, 1, 2, 3]
    assert len(yielded[2].array) == 0


def test_empty_frame_identity_is_reconstructed_from_the_cadence():
    det = _detections(n_frames=4)
    det = det[det["processed_frame"] != 2]
    gap = list(iter_frame_detections(det))[2]
    assert gap.source_frame == 4  # stride 2, so processed 2 is source 4


def test_conf_floor_matches_ultralytics_half_precision_semantics():
    # Ultralytics filters with a strict > against an fp16 score, so a box at
    # exactly fp16(0.15) is dropped by a native pass and must be dropped here.
    at_threshold = float(np.float16(0.15))
    df = pd.DataFrame({"confidence": [at_threshold, 0.2, 0.05]})
    kept = apply_conf_floor(df, 0.15, half=True)
    assert list(kept["confidence"]) == [0.2]


def test_fingerprint_detects_a_changed_box():
    a = _detections()
    b = a.copy()
    b.loc[0, "x1"] = b.loc[0, "x1"] + 1.0
    assert detection_fingerprint(a)["frame_hash"] == detection_fingerprint(b)["frame_hash"]
    assert detection_fingerprint(a)["box_hash"] != detection_fingerprint(b)["box_hash"]


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def test_adapter_recovers_constant_velocity_tracks():
    det = _detections(n_frames=20, n_objects=3)
    arm = build_arm("bytetrack", BYTETRACK)
    results, _ = replay(det, [arm])
    tracks = results["bytetrack"].tracks
    assert tracks["track_id"].nunique() == 3
    assert len(tracks) == 60


def test_adapter_rejects_wrongly_shaped_detections():
    arm = build_arm("bytetrack", BYTETRACK)
    with pytest.raises(ValueError, match="detections must be"):
        arm.update(np.zeros((3, 4), dtype=np.float32))


def test_frame_consuming_arm_refuses_to_run_without_a_frame():
    # Silently tracking without the image would disable global motion
    # compensation and misreport the arm as its configured self.
    arm = build_arm("botsort", "configs/tracker_botsort.yaml")
    assert arm.needs_frames
    with pytest.raises(ValueError, match="needs the decoded frame"):
        arm.update(np.zeros((0, 6), dtype=np.float32), frame=None)


def test_replay_refuses_frame_consuming_arms_without_a_video():
    det = _detections()
    arm = build_arm("botsort", "configs/tracker_botsort.yaml")
    with pytest.raises(ValueError, match="need decoded frames"):
        replay(det, [arm], video_path=None)


def test_arms_run_together_match_arms_run_alone():
    # Ultralytics numbers tracks from a single global counter, so without a
    # private counter per arm a shared-decode run would interleave IDs and an
    # arm's output would depend on which other arms happened to run with it.
    det = _detections(n_frames=15)

    def ids(arms):
        res, _ = replay(det, arms)
        return {n: sorted(r.tracks["track_id"].unique().tolist()) for n, r in res.items()}

    alone = {}
    alone.update(ids([build_arm("bytetrack", BYTETRACK)]))
    alone.update(ids([build_arm("ocsort", OCSORT)]))
    together = ids([build_arm("bytetrack", BYTETRACK), build_arm("ocsort", OCSORT)])
    assert together == alone


def test_reset_restarts_track_numbering():
    det = _detections(n_frames=10)
    arm = build_arm("bytetrack", BYTETRACK)
    first, _ = replay(det, [arm])
    second, _ = replay(det, [arm])  # replay() resets every arm
    assert (sorted(first["bytetrack"].tracks["track_id"].unique())
            == sorted(second["bytetrack"].tracks["track_id"].unique()))


def test_single_stage_bytetrack_has_an_inert_low_confidence_branch():
    arm = build_arm("bytetrack_iou", BYTETRACK_IOU)
    assert arm.args.track_low_thresh == arm.args.track_high_thresh
    assert not arm.needs_frames


def test_appearance_arm_refuses_the_auto_encoder():
    # `model: auto` reuses detector backbone features that do not exist when
    # replaying a detection table; accepting it would report an arm as
    # appearance-based while its ReID silently did nothing.
    from mosmon_tracking.trackers import ReidUnavailable

    arm = build_arm("bytetrack", BYTETRACK)
    arm.args.with_reid = True
    arm.args.model = "auto"
    with pytest.raises(ReidUnavailable, match="auto"):
        arm.reset()


def test_appearance_arm_falls_back_then_refuses_if_no_encoder_loads():
    # An arm that quietly loses its ReID encoder is no longer an appearance arm,
    # so a total failure must raise rather than degrade to plain OC-SORT.
    from mosmon_tracking.trackers import ReidUnavailable

    arm = build_arm("bytetrack", BYTETRACK)
    arm.args.with_reid = True
    arm.args.model = "definitely-not-a-model.onnx"
    arm.reid_fallback = "also-not-a-model.pt"
    with pytest.raises(ReidUnavailable) as excinfo:
        arm.reset()
    message = str(excinfo.value)
    assert "must not be reported as an appearance arm" in message
    # Both candidates are named, so the failure is diagnosable.
    assert "definitely-not-a-model.onnx" in message
    assert "also-not-a-model.pt" in message


def test_reid_asset_resolves_to_a_local_copy_before_downloading(tmp_path, monkeypatch):
    from mosmon_tracking.trackers.ultralytics_adapter import UltralyticsTrackerAdapter

    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights" / "reid").mkdir(parents=True)
    (tmp_path / "weights" / "reid" / "some-reid.onnx").write_bytes(b"x")
    assert (UltralyticsTrackerAdapter._resolve_encoder("some-reid.onnx")
            == "weights/reid/some-reid.onnx")
    # An unknown name is passed through untouched, for Ultralytics to fetch.
    assert UltralyticsTrackerAdapter._resolve_encoder("other.onnx") == "other.onnx"
    # An explicit path is never rewritten.
    assert UltralyticsTrackerAdapter._resolve_encoder("a/b.onnx") == "a/b.onnx"


def test_provenance_records_the_implementation():
    arm = build_arm("ocsort", OCSORT)
    prov = arm.provenance()
    assert prov["implementation"] == "ultralytics"
    assert prov["tracker_type"] == "ocsort"
    assert prov["implementation_version"]


def test_assignments_map_tracks_back_to_shared_detections():
    det = _detections(n_frames=10, n_objects=3)
    res, _ = replay(det, [build_arm("bytetrack", BYTETRACK)])
    a = res["bytetrack"].assignments
    assert set(a.columns) == {"processed_frame", "det_row", "track_id"}
    # Every det_row is a valid row of the shared detection table.
    assert a["det_row"].between(0, len(det) - 1).all()
    assert not a.duplicated(subset=["processed_frame", "det_row"]).any()


# --------------------------------------------------------------------------- #
# identity agreement
# --------------------------------------------------------------------------- #
def test_agreement_is_invariant_to_relabelling():
    a = np.array([1, 1, 1, 2, 2, 3])
    b = np.array([7, 7, 7, 8, 8, 9])
    assert adjusted_rand_index(a, b) == pytest.approx(1.0)
    assert v_measure(a, b) == pytest.approx(1.0)
    assert identity_f1(a, b) == pytest.approx(1.0)


def test_agreement_falls_when_a_track_fragments():
    a = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    b = np.array([1, 1, 9, 9, 2, 2, 2, 2])
    assert adjusted_rand_index(a, b) < 1.0
    assert identity_f1(a, b) < 1.0


def test_full_fragmentation_gives_zero_ari():
    a = np.array([1, 1, 1, 2, 2, 2])
    b = np.arange(6)
    assert adjusted_rand_index(a, b) == pytest.approx(0.0, abs=1e-9)


def test_identical_arms_agree_perfectly():
    det = _detections(n_frames=12)
    res, _ = replay(det, [build_arm("a", BYTETRACK), build_arm("b", BYTETRACK)])
    pair = pairwise_agreement({n: r.assignments for n, r in res.items()})
    assert len(pair) == 1
    assert pair.iloc[0]["ari"] == pytest.approx(1.0)
    assert pair.iloc[0]["identity_f1"] == pytest.approx(1.0)


def test_agreement_reports_coverage_separately_from_similarity():
    # An arm that emits fewer detections is not thereby "in agreement": the
    # comparison is over the intersection, so coverage has to be reported too.
    left = pd.DataFrame({"det_row": [0, 1, 2, 3], "track_id": [1, 1, 2, 2]})
    right = pd.DataFrame({"det_row": [0, 1], "track_id": [5, 5]})
    out = compare_assignments(left, right)
    assert out["n_common"] == 2
    assert out["coverage_left"] == pytest.approx(0.5)
    assert out["coverage_right"] == pytest.approx(1.0)
