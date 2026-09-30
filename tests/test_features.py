import numpy as np
import pandas as pd

from mosmon_tracking.features import (
    add_cross_track_features,
    anomalous_diffusion_exponent,
    compute_track_summary,
    mean_squared_displacement,
    net_displacement,
    path_length,
    radius_of_gyration,
    straightness,
    tortuosity,
)
from mosmon_tracking.track_postprocess import clean_tracks
from tests.test_track_postprocess import _cfg, _make_track


def test_path_length_and_displacement():
    cx = np.array([0.0, 3.0, 3.0])
    cy = np.array([0.0, 0.0, 4.0])
    assert path_length(cx, cy) == 7.0       # 3 + 4
    assert net_displacement(cx, cy) == 5.0  # 3-4-5 triangle


def test_tortuosity_straight_line_is_one():
    cx = np.linspace(0, 10, 11)
    cy = np.zeros(11)
    assert np.isclose(tortuosity(cx, cy), 1.0)
    assert np.isclose(straightness(cx, cy), 1.0)


def test_tortuosity_backtrack_greater_than_one():
    cx = np.array([0.0, 5.0, 0.0])  # out and back: path 10, net 0
    cy = np.zeros(3)
    assert np.isnan(tortuosity(cx, cy)) or tortuosity(cx, cy) > 1.0


def test_radius_of_gyration():
    cx = np.array([-1.0, 1.0, -1.0, 1.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0])
    assert np.isclose(radius_of_gyration(cx, cy), np.sqrt(2.0))


def test_msd_linear_motion():
    # constant velocity v=2 -> MSD(tau) = (v*tau)^2
    cx = 2.0 * np.arange(10, dtype=float)
    cy = np.zeros(10)
    msd = mean_squared_displacement(cx, cy)
    assert np.isclose(msd[0], 4.0)    # tau=1 -> (2*1)^2
    assert np.isclose(msd[1], 16.0)   # tau=2 -> (2*2)^2
    # ballistic motion -> alpha ~ 2
    alpha = anomalous_diffusion_exponent(msd)
    assert 1.8 < alpha < 2.2


def test_nearest_neighbor_two_tracks():
    a = _make_track(track_id=1, n=10, x0=100, y0=100, vx=0, vy=0)
    b = _make_track(track_id=2, n=10, x0=130, y0=100, vx=0, vy=0)
    raw = pd.concat([a, b], ignore_index=True)
    clean, _ = clean_tracks(raw, _cfg())
    clean = add_cross_track_features(clean, _cfg())
    # two stationary tracks 30px apart
    assert np.allclose(clean["nearest_neighbor_distance_px"].dropna(), 30.0, atol=1e-6)


def test_track_summary_schema():
    raw = _make_track(n=20, vx=5.0)
    cfg = _cfg()
    clean, _ = clean_tracks(raw, cfg)
    clean = add_cross_track_features(clean, cfg)
    ts = compute_track_summary(clean, cfg)
    assert len(ts) == 1
    for col in ("track_id", "n_frames", "duration_s", "mean_speed_px_s",
                "tortuosity", "straightness", "quality_score"):
        assert col in ts.columns
    assert 0.0 <= ts["quality_score"].iloc[0] <= 1.0
    assert np.isclose(ts["mean_speed_px_s"].iloc[0], 50.0, atol=1e-6)
