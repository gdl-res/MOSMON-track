import configparser

import pandas as pd

from mosmon_tracking.mot import MOT_COLUMNS, tracks_to_mot, write_mot


def _tracks():
    return pd.DataFrame([
        # deliberately out of frame/id order to test sorting
        {"frame_idx": 1, "track_id": 2, "x1": 10, "y1": 20, "x2": 30, "y2": 60,
         "w": 20, "h": 40, "confidence": 0.8, "is_interpolated": False},
        {"frame_idx": 0, "track_id": 2, "x1": 5, "y1": 5, "x2": 15, "y2": 25,
         "w": 10, "h": 20, "confidence": 0.9, "is_interpolated": False},
        {"frame_idx": 0, "track_id": 1, "x1": 0, "y1": 0, "x2": 8, "y2": 8,
         "w": 8, "h": 8, "confidence": 0.7, "is_interpolated": True},
    ])


def test_columns_and_frame_is_one_based():
    mot = tracks_to_mot(_tracks())
    assert list(mot.columns) == MOT_COLUMNS
    # frame_idx 0 -> MOT frame 1
    assert mot["frame"].min() == 1
    # world coords fixed to -1 for 2-D tracking
    assert (mot[["x", "y", "z"]] == -1).all().all()


def test_sorted_by_frame_then_id():
    mot = tracks_to_mot(_tracks())
    assert mot[["frame", "id"]].values.tolist() == [[1, 1], [1, 2], [2, 2]]


def test_bbox_is_left_top_width_height():
    mot = tracks_to_mot(_tracks())
    row = mot[(mot["frame"] == 1) & (mot["id"] == 2)].iloc[0]
    assert (row["bb_left"], row["bb_top"], row["bb_width"], row["bb_height"]) == (5, 5, 10, 20)


def test_width_height_derived_when_missing():
    df = _tracks().drop(columns=["w", "h"])
    mot = tracks_to_mot(df)
    row = mot[(mot["frame"] == 2) & (mot["id"] == 2)].iloc[0]
    assert row["bb_width"] == 20 and row["bb_height"] == 40  # x2-x1, y2-y1


def test_exclude_interpolated():
    full = tracks_to_mot(_tracks(), include_interpolated=True)
    no_interp = tracks_to_mot(_tracks(), include_interpolated=False)
    assert len(full) == 3 and len(no_interp) == 2
    assert not ((no_interp["frame"] == 1) & (no_interp["id"] == 1)).any()


def test_empty_input_returns_empty_with_columns():
    mot = tracks_to_mot(pd.DataFrame())
    assert mot.empty and list(mot.columns) == MOT_COLUMNS


def test_write_mot_creates_file_and_seqinfo(tmp_path):
    path = write_mot(
        _tracks(), tmp_path / "tracks_mot.txt",
        seqinfo={"name": "vid", "frame_rate": 29.97, "seq_length": 2,
                 "im_width": 1920, "im_height": 1080},
    )
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0].split(",")[:2] == ["1", "1"]  # first row: frame 1, id 1

    seqinfo = tmp_path / "seqinfo.ini"
    assert seqinfo.exists()
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(seqinfo)
    assert cfg["Sequence"]["frameRate"] == "30"  # rounded
    assert cfg["Sequence"]["seqLength"] == "2"
    assert cfg["Sequence"]["imWidth"] == "1920"


def test_write_mot_empty_still_writes(tmp_path):
    path = write_mot(pd.DataFrame(), tmp_path / "tracks_mot.txt")
    assert path.exists() and path.read_text().strip() == ""
