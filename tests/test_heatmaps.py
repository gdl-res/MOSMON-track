import numpy as np
import pandas as pd

from mosmon_tracking.config import Config
from mosmon_tracking.heatmaps import (
    heatmap_entropy,
    occupancy_heatmap,
    speed_heatmap,
)


def _df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cx_norm": rng.uniform(0, 1, n),
        "cy_norm": rng.uniform(0, 1, n),
        "speed_px_s": rng.uniform(0, 10, n),
        "dt": np.full(n, 0.1),
        "movement_state": rng.choice(["active", "slow", "freezing"], n),
    })


def test_occupancy_probability_sums_to_one():
    cfg = Config()
    cfg.heatmaps.normalize = "probability"
    hm = occupancy_heatmap(_df(), cfg)
    assert hm.array.shape == (cfg.heatmaps.bins_y, cfg.heatmaps.bins_x)
    assert np.isclose(hm.array.sum(), 1.0)


def test_occupancy_count_mode():
    cfg = Config()
    cfg.heatmaps.normalize = "count"
    df = _df(n=500)
    hm = occupancy_heatmap(df, cfg)
    assert np.isclose(hm.array.sum(), len(df))


def test_speed_heatmap_empty_bins_are_nan():
    cfg = Config()
    cfg.heatmaps.bins_x = 8
    cfg.heatmaps.bins_y = 8
    # only put points in one corner -> most bins empty -> NaN
    df = pd.DataFrame({
        "cx_norm": [0.01, 0.02], "cy_norm": [0.01, 0.02],
        "speed_px_s": [4.0, 6.0],
    })
    hm = speed_heatmap(df, cfg)
    assert np.isnan(hm.array).sum() > 0
    assert np.isclose(np.nanmean(hm.array), 5.0)


def test_heatmap_entropy_uniform_is_high():
    cfg = Config()
    cfg.heatmaps.normalize = "probability"
    uniform = occupancy_heatmap(_df(n=20000), cfg)
    point = occupancy_heatmap(
        pd.DataFrame({"cx_norm": [0.5] * 50, "cy_norm": [0.5] * 50}), cfg
    )
    assert heatmap_entropy(uniform) > heatmap_entropy(point)
