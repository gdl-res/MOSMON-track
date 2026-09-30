"""Tests for the detection-cache reader.

The reader was rewritten on 2026-08-27 after it OOM-killed the benchmark replay
three times: a dense Culex cache is 30.5 M rows whose six constant label columns
expand to 20.2 GB of a 22.3 GB pandas table. It now keeps those columns
dictionary-encoded and pushes the confidence floor into pyarrow.

That is a *memory* change, and these tests exist to hold it to that. The pool
every arm receives must be byte-identical to what the old path produced --
otherwise the videos already benchmarked would not be comparable with the rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosmon_tracking.benchmark.detections_io import (
    LABEL_COLUMNS,
    apply_conf_floor,
    conf_threshold,
    detection_fingerprint,
    iter_frame_detections,
    read_detections,
)
from mosmon_tracking.video_io import load_table, save_table
from mosmon_tracking.yolo_tracker import DETECTION_COLUMNS

NUMERIC = ["source_frame", "processed_frame", "time_s", "x1", "y1", "x2", "y2",
           "confidence", "predicted_class", "frame_width", "frame_height"]


def _boundary_confidences() -> list[np.float32]:
    """Scores that straddle the fp16 thresholds, plus ordinary ones.

    The exact-threshold values are the trap: Ultralytics filters with a strict
    ``>`` against an fp16-promoted threshold, so a box sitting precisely on
    fp16(0.15) is dropped by a native pass. Using ``>=``, or comparing against
    the float64 value, keeps it -- measured at 7 boxes in 6890 on real data.
    """
    out: list[np.float32] = []
    for c in (0.10, 0.15):
        e = np.float32(conf_threshold(c, half=True))
        out += [np.nextafter(e, 0, dtype=np.float32), e,
                np.nextafter(e, 1, dtype=np.float32)]
    out += [np.float32(x) for x in (0.05, 0.0999, 0.1001, 0.2, 0.9)]
    return out


def _detection_frame(confidences=None) -> pd.DataFrame:
    conf = list(_boundary_confidences() if confidences is None else confidences)
    n = len(conf)
    return pd.DataFrame({
        "video_id": ["vidA"] * n,
        # Deliberately long: the per-row cost of this column is the whole bug.
        "video_path": ["/media/ssd/all_videos/gopro_hero11black/vidA.mp4"] * n,
        "video_name": ["vidA.mp4"] * n,
        "model_name": ["yolo11x"] * n,
        "model_path": ["weights/yolo11x.pt"] * n,
        "source_frame": np.arange(n, dtype="int64") * 2,
        "processed_frame": np.arange(n, dtype="int64"),
        "time_s": np.arange(n, dtype=float) / 30.0,
        "x1": np.arange(n, dtype="float32"),
        "y1": np.arange(n, dtype="float32"),
        "x2": np.arange(n, dtype="float32") + 5.0,
        "y2": np.arange(n, dtype="float32") + 5.0,
        "confidence": np.asarray(conf, dtype="float32"),
        "predicted_class": np.zeros(n, dtype="int64"),
        "class_name": (["Culex pipiens", "Aedes aegypti"] * n)[:n],
        "frame_width": np.full(n, 3840, dtype="int64"),
        "frame_height": np.full(n, 2160, dtype="int64"),
    })[DETECTION_COLUMNS]


def _reference(path, conf_min, half=True) -> pd.DataFrame:
    """The pre-rewrite implementation, kept as the thing to match."""
    df = load_table(path)
    if conf_min is not None:
        df = apply_conf_floor(df, conf_min, half=half)
    return df.reset_index(drop=True)


@pytest.fixture()
def cache(tmp_path):
    path = tmp_path / "vidA.parquet"
    save_table(_detection_frame(), path)
    return path


@pytest.mark.parametrize("conf_min", [None, 0.05, 0.10, 0.15, 0.2, 0.99])
@pytest.mark.parametrize("half", [True, False])
def test_matches_the_pandas_reference_exactly(cache, conf_min, half):
    """Same rows, same values, same dtypes -- memory is all that changed."""
    new = read_detections(cache, conf_min=conf_min, half=half)
    ref = _reference(cache, conf_min, half=half)

    assert len(new) == len(ref)
    for col in NUMERIC:
        assert np.array_equal(new[col].to_numpy(), ref[col].to_numpy(),
                              equal_nan=True), col
        assert new[col].dtype == ref[col].dtype, col
    for col in LABEL_COLUMNS:
        assert list(new[col].astype(str)) == list(ref[col].astype(str)), col


@pytest.mark.parametrize("conf_min", [None, 0.10, 0.15])
def test_fingerprint_is_unchanged(cache, conf_min):
    """The §31 fairness assertion: every arm must be shown the same pool."""
    assert (detection_fingerprint(read_detections(cache, conf_min=conf_min))
            == detection_fingerprint(_reference(cache, conf_min)))


def test_threshold_is_strict_and_fp16(cache):
    """A box exactly on fp16(conf) is dropped; one just above survives."""
    kept = read_detections(cache, conf_min=0.15, half=True)["confidence"].to_numpy()
    edge = np.float32(conf_threshold(0.15, half=True))
    assert edge not in kept
    assert np.nextafter(edge, 1, dtype=np.float32) in kept
    assert np.nextafter(edge, 0, dtype=np.float32) not in kept


def test_labels_come_back_categorical_but_read_as_strings(cache):
    """Dictionary encoding is what makes dense caches loadable at all.

    Callers reach these columns via ``.iloc[0]`` (``run.py`` builds the run
    folder name from ``video_id``), so the values must still behave as plain
    strings.
    """
    det = read_detections(cache, conf_min=0.10)
    for col in LABEL_COLUMNS:
        assert isinstance(det[col].dtype, pd.CategoricalDtype), col
    assert det["video_id"].iloc[0] == "vidA"
    assert isinstance(det["video_id"].iloc[0], str)
    assert list(det.columns) == DETECTION_COLUMNS


def test_label_columns_cost_codes_not_strings(tmp_path):
    """The regression that matters: a row must not carry six whole strings.

    Dictionary encoding does not make the label columns free -- there is still
    one integer code per row -- but it drops the per-row cost by well over an
    order of magnitude, which is the difference between 1.3 GB and 22.3 GB on
    the real Culex cache.
    """
    path = tmp_path / "large.parquet"
    n = 20_000
    save_table(_detection_frame([0.5] * n), path)

    def label_bytes(df):
        return sum(df[c].memory_usage(deep=True) for c in LABEL_COLUMNS)

    new_cost = label_bytes(read_detections(path)) / n
    old_cost = label_bytes(_reference(path, None)) / n

    assert new_cost < old_cost / 10
    assert new_cost < 40  # bytes per row for all six columns together


def test_empty_and_single_row_caches(tmp_path):
    empty, single = tmp_path / "e.parquet", tmp_path / "s.parquet"
    save_table(_detection_frame().iloc[:0], empty)
    save_table(_detection_frame([0.5]), single)

    assert read_detections(empty).empty
    assert list(read_detections(empty).columns) == DETECTION_COLUMNS
    assert len(read_detections(single)) == 1
    # Filtered down to nothing: still empty, still the right columns.
    assert read_detections(single, conf_min=0.99).empty
    assert list(read_detections(single, conf_min=0.99).columns) == DETECTION_COLUMNS


def test_csv_cache_still_readable(tmp_path):
    """The CSV fallback path bypasses pyarrow but must behave identically."""
    path = tmp_path / "vidA.csv"
    _detection_frame().to_csv(path, index=False)
    det = read_detections(path, conf_min=0.10)
    assert len(det) == len(_reference(path, 0.10))


def test_iter_frame_detections_orders_unsorted_input(cache):
    """The monotonic fast path must not let out-of-order rows through unsorted."""
    det = read_detections(cache)
    shuffled = det.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert not shuffled["processed_frame"].is_monotonic_increasing

    frames = [fd.processed_frame for fd in iter_frame_detections(shuffled)]
    assert frames == sorted(frames)
    assert frames == [fd.processed_frame for fd in iter_frame_detections(det)]
