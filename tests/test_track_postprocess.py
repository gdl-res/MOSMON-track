import numpy as np
import pandas as pd

from mosmon_tracking.config import Config
from mosmon_tracking.track_postprocess import (
    classify_region,
    clean_tracks,
    distance_to_border,
    wrap_angle,
)


def _make_track(track_id=1, n=20, fps=10.0, vx=5.0, vy=0.0, x0=100.0, y0=100.0,
                W=640, H=480, frames=None):
    frames = frames if frames is not None else list(range(n))
    rows = []
    for f in frames:
        cx = x0 + vx * f
        cy = y0 + vy * f
        rows.append({
            "video_name": "v.mp4", "model_name": "m", "tracker": "bytetrack",
            "frame_idx": f, "time_s": f / fps, "track_id": track_id,
            "class_id": 0, "class_name": "Aedes albopictus", "confidence": 0.9,
            "x1": cx - 5, "y1": cy - 5, "x2": cx + 5, "y2": cy + 5,
            "w": 10.0, "h": 10.0, "cx": cx, "cy": cy,
            "cx_norm": cx / W, "cy_norm": cy / H,
            "frame_width": W, "frame_height": H,
        })
    return pd.DataFrame(rows)


def _cfg(**over):
    cfg = Config()
    cfg.tracker.min_track_length_frames = 5
    cfg.tracker.min_mean_confidence = 0.0
    cfg.postprocess.smoothing = "none"
    for k, v in over.items():
        section, field = k.split(".")
        setattr(getattr(cfg, section), field, v)
    return cfg


def test_wrap_angle():
    assert np.isclose(wrap_angle(np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(wrap_angle(-np.pi), np.pi) or np.isclose(wrap_angle(-np.pi), -np.pi)


def test_constant_velocity_speed():
    raw = _make_track(vx=5.0, fps=10.0)  # 5 px / 0.1 s = 50 px/s
    clean, dropped = clean_tracks(raw, _cfg())
    speeds = clean["speed_px_s"].dropna()
    assert np.allclose(speeds, 50.0, atol=1e-6)
    accel = clean["acceleration_px_s2"].dropna()
    assert np.allclose(accel, 0.0, atol=1e-6)
    # straight horizontal motion -> heading 0, turn 0
    assert np.allclose(clean["heading_rad"].dropna(), 0.0, atol=1e-6)


def test_gap_interpolation_flags():
    frames = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]  # gap of 3 missing (5,6,7)
    raw = _make_track(frames=frames, vx=5.0)
    cfg = _cfg(**{"postprocess.interpolate_gaps": True,
                  "postprocess.interpolation_max_gap_frames": 5})
    clean, _ = clean_tracks(raw, cfg)
    assert clean["is_interpolated"].sum() == 3
    # interpolated centers lie on the straight line (cx = 100 + 5*frame)
    interp = clean[clean["is_interpolated"]]
    assert np.allclose(interp["cx"], 100 + 5 * interp["frame_idx"], atol=1e-6)


def test_long_gap_not_filled():
    frames = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]  # gap of 15 > max
    raw = _make_track(frames=frames, vx=5.0)
    cfg = _cfg(**{"postprocess.interpolate_gaps": True,
                  "postprocess.interpolation_max_gap_frames": 5})
    clean, _ = clean_tracks(raw, cfg)
    assert clean["is_interpolated"].sum() == 0  # gap too long, dropped not fabricated


def test_short_track_dropped():
    raw = _make_track(n=3)
    cfg = _cfg(**{"tracker.min_track_length_frames": 10})
    clean, dropped = clean_tracks(raw, cfg)
    assert clean.empty
    assert len(dropped) == 1
    assert "short_track" in dropped["drop_reason"].iloc[0]


def test_classify_region():
    W, H = 100, 100
    labels = classify_region(np.array([5, 50, 20]), np.array([50, 50, 20]), W, H,
                             Config().regions)
    assert labels[0] == "border"        # near left edge (< 10px margin)
    assert labels[1] == "center"        # middle, inside central 50% box [25,75]
    assert labels[2] == "intermediate"  # (20,20): not border, outside center box


def test_distance_to_border():
    d = distance_to_border(np.array([10.0]), np.array([50.0]), 100, 100)
    assert np.isclose(d[0], 10.0)
