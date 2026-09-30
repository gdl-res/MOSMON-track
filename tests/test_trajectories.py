import json

import numpy as np
import pandas as pd

from mosmon_tracking.calibration import Calibrator
from mosmon_tracking.trajectories import tracks_to_geojson, write_geojson


def _clean():
    return pd.DataFrame([
        {"track_id": 1, "frame_idx": 1, "cx": 10.0, "cy": 20.0},
        {"track_id": 1, "frame_idx": 0, "cx": 0.0, "cy": 0.0},   # out of order on purpose
        {"track_id": 1, "frame_idx": 2, "cx": 20.0, "cy": 40.0},
        {"track_id": 2, "frame_idx": 0, "cx": 5.0, "cy": 5.0},   # single point -> Point
    ])


def _summary():
    return pd.DataFrame([
        {"track_id": 1, "class_name": "Aedes albopictus", "duration_s": 0.2,
         "quality_score": 0.9, "mean_speed_px_s": np.nan},
        {"track_id": 2, "class_name": "Culex pipiens", "duration_s": 0.0,
         "quality_score": 0.1, "mean_speed_px_s": 1.0},
    ])


def test_feature_collection_shape():
    fc = tracks_to_geojson(_clean(), _summary())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    assert fc["metadata"]["crs_units"] == "px"


def test_linestring_is_time_ordered():
    fc = tracks_to_geojson(_clean(), _summary())
    feat = next(f for f in fc["features"] if f["properties"]["track_id"] == 1)
    assert feat["geometry"]["type"] == "LineString"
    # ordered by frame_idx 0,1,2 despite shuffled input
    assert feat["geometry"]["coordinates"] == [[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]]


def test_single_point_track_is_point():
    fc = tracks_to_geojson(_clean(), _summary())
    feat = next(f for f in fc["features"] if f["properties"]["track_id"] == 2)
    assert feat["geometry"]["type"] == "Point"
    assert feat["geometry"]["coordinates"] == [5.0, 5.0]


def test_properties_attached_and_nan_is_null():
    fc = tracks_to_geojson(_clean(), _summary())
    feat = next(f for f in fc["features"] if f["properties"]["track_id"] == 1)
    assert feat["properties"]["class_name"] == "Aedes albopictus"
    assert feat["properties"]["mean_speed_px_s"] is None  # NaN -> None
    assert feat["properties"]["n_points"] == 3


def test_calibrated_coordinates_in_mm():
    cal = Calibrator(mode="scalar", pixels_per_mm=10.0)
    fc = tracks_to_geojson(_clean(), _summary(), calibrator=cal)
    assert fc["metadata"]["crs_units"] == "mm"
    feat = next(f for f in fc["features"] if f["properties"]["track_id"] == 1)
    # 10 px / 10 px-per-mm = 1 mm
    assert feat["geometry"]["coordinates"][1] == [1.0, 2.0]


def test_write_is_valid_json(tmp_path):
    fc = tracks_to_geojson(_clean(), _summary())
    path = write_geojson(fc, tmp_path / "trajectories.geojson")
    assert path.exists()
    loaded = json.loads(path.read_text())  # must be strict JSON (no NaN tokens)
    assert loaded["type"] == "FeatureCollection"


def test_empty_input_returns_empty_collection():
    fc = tracks_to_geojson(pd.DataFrame())
    assert fc["type"] == "FeatureCollection" and fc["features"] == []
