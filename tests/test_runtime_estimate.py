from mosmon_tracking import runtime
from mosmon_tracking.config import Config


def test_seconds_per_frame_matches_calibration_points():
    assert runtime.seconds_per_frame(8.29) == 0.302    # 4K
    assert runtime.seconds_per_frame(33.18) == 0.754   # 8K


def test_seconds_per_frame_interpolates_between_points():
    spf = runtime.seconds_per_frame(12.0)  # between 4K and 5.3K
    assert 0.302 < spf < 0.409


def test_bytetrack_is_faster():
    assert runtime.seconds_per_frame(33.18, "bytetrack") < runtime.seconds_per_frame(33.18, "botsort")


def test_processed_frames_respects_stride_and_cap():
    cfg = Config()
    cfg.video.frame_stride = 2
    cfg.video.max_frames = None
    assert runtime.processed_frames(1000, 30.0, 33.0, cfg) == 500
    cfg.video.max_frames = 200
    assert runtime.processed_frames(1000, 30.0, 33.0, cfg) == 200


def test_estimate_video_seconds_positive():
    cfg = Config()
    s = runtime.estimate_video_seconds(5312, 2988, 18000, 30.0, 600.0, cfg)
    assert s > 0


def test_lean_output_estimate_is_smaller():
    full = Config()
    lean = Config()
    lean.reports.export_mot = False
    lean.reports.export_trajectories_geojson = False
    args = (18000, 30.0, 600.0)
    full_b = runtime.estimate_video_output_bytes(*args, full)
    lean_b = runtime.estimate_video_output_bytes(*args, lean)
    assert lean_b < full_b
    # the dropped components are MOT (6k) + GeoJSON (12k) = 18k bytes/frame
    n = runtime.processed_frames(*args, full)
    assert abs((full_b - lean_b) - n * 18_000) < 1
