import numpy as np
import pandas as pd

from mosmon_tracking.config import Config
from mosmon_tracking.features import compute_population_timeseries


def _clean():
    # 2 frames; frame 0 has 2 larvae, frame 1 has 1 larva.
    return pd.DataFrame([
        {"video_name": "v", "model_name": "m", "tracker": "bytetrack",
         "frame_idx": 0, "time_s": 0.0, "track_id": 1, "speed_px_s": 10.0,
         "movement_state": "active", "roi_label": "border", "is_interpolated": False,
         "nearest_neighbor_distance_px": 50.0},
        {"video_name": "v", "model_name": "m", "tracker": "bytetrack",
         "frame_idx": 0, "time_s": 0.0, "track_id": 2, "speed_px_s": np.inf,
         "movement_state": "freezing", "roi_label": "center", "is_interpolated": True,
         "nearest_neighbor_distance_px": 50.0},
        {"video_name": "v", "model_name": "m", "tracker": "bytetrack",
         "frame_idx": 1, "time_s": 0.1, "track_id": 1, "speed_px_s": 20.0,
         "movement_state": "burst", "roi_label": "border", "is_interpolated": False,
         "nearest_neighbor_distance_px": np.nan},
    ])


def test_one_row_per_frame_and_counts():
    ts = compute_population_timeseries(_clean(), Config())
    assert len(ts) == 2  # two distinct frames
    assert ts.loc[ts["frame_idx"] == 0, "n_larvae"].iloc[0] == 2
    assert ts.loc[ts["frame_idx"] == 1, "n_larvae"].iloc[0] == 1


def test_column_order_keys_then_frame_then_time():
    ts = compute_population_timeseries(_clean(), Config())
    assert list(ts.columns[:5]) == ["video_name", "model_name", "tracker", "frame_idx", "time_s"]


def test_speed_ignores_inf():
    ts = compute_population_timeseries(_clean(), Config())
    # frame 0: speeds {10, inf} -> mean over finite = 10
    assert ts.loc[ts["frame_idx"] == 0, "mean_speed_px_s"].iloc[0] == 10.0


def test_fraction_columns():
    ts = compute_population_timeseries(_clean(), Config())
    f0 = ts[ts["frame_idx"] == 0].iloc[0]
    # frame 0: one active + one freezing -> frac_active 0.5, frac_freezing 0.5
    assert f0["frac_active"] == 0.5
    assert f0["frac_freezing"] == 0.5
    assert f0["frac_border"] == 0.5 and f0["frac_center"] == 0.5
    assert f0["interpolated_fraction"] == 0.5
    f1 = ts[ts["frame_idx"] == 1].iloc[0]
    assert f1["frac_burst"] == 1.0  # burst counts as active too
    assert f1["frac_active"] == 1.0


def test_empty_input_returns_empty():
    assert compute_population_timeseries(pd.DataFrame(), Config()).empty
