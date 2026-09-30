"""Read, write and replay the frozen detection cache.

The cache is the whole point of the benchmark: one detector pass per video,
replayed identically by every association algorithm. Nothing here may reorder,
resample or otherwise perturb detections, because "same detections for every
arm" is the claim the entire comparison rests on.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..video_io import load_table, save_table
from ..yolo_tracker import empty_detection_table

#: Order of the (N, 6) array handed to a tracker adapter.
DET_ARRAY_COLUMNS = ["x1", "y1", "x2", "y2", "confidence", "predicted_class"]

#: Columns that are constant, or near-constant, for a whole video.
#:
#: Materialising these as one Python string per row is what made dense caches
#: unloadable. On a 30.5 M-row Culex video they account for **20.2 GB of a
#: 22.3 GB table**, against 2.1 GB for every numeric column combined --
#: ``video_path`` alone is 7.0 GB of repeated 150-character path. They are held
#: dictionary-encoded instead, the same trick
#: :func:`..run_tracker_benchmark._constant_column` already applies to raw track
#: tables.
LABEL_COLUMNS = ["video_id", "video_path", "video_name",
                 "model_name", "model_path", "class_name"]


@dataclass(frozen=True)
class FrameDetections:
    """One frame's worth of detections, plus the indices they came from."""

    processed_frame: int
    source_frame: int
    time_s: float
    array: np.ndarray          # (N, 6) x1 y1 x2 y2 conf cls
    row_index: np.ndarray      # (N,) positions in the detection table
    frame_width: int
    frame_height: int


def write_detections(df: pd.DataFrame, path: str | Path) -> Path:
    return save_table(df, Path(path))


def apply_conf_floor(df: pd.DataFrame, conf_min: float, half: bool = True) -> pd.DataFrame:
    """Reconstruct the pool a detector pass at ``conf_min`` would have produced.

    This is exact, not an approximation. NMS is greedy over scores in descending
    order, so a lower-scoring box can never suppress a higher-scoring one, and
    the set surviving above any threshold is the same whether the threshold was
    applied before or after NMS -- provided ``max_det`` never bound.

    The comparison must match Ultralytics exactly, which takes two steps:

    * it filters with a **strict** ``>``, not ``>=``; and
    * under ``half=True`` the scores are fp16 and the threshold is promoted to
      fp16 before comparison, so the effective cut is ``fp16(conf_min)``.

    Skipping either step keeps the handful of boxes sitting exactly on the
    threshold that a native pass drops -- measured at 7 boxes in 6890 for a
    0.15 cut, every one of them at fp16(0.15) = 0.150024414.
    """
    return df[df["confidence"].astype(float) > conf_threshold(conf_min, half)]


def conf_threshold(conf_min: float, half: bool = True) -> float:
    """The exact value a detector pass at ``conf_min`` compares against.

    Split out of :func:`apply_conf_floor` so the pandas and pyarrow paths cannot
    drift apart -- the whole benchmark rests on every arm seeing the same pool.
    """
    return float(np.float16(conf_min) if half else np.float32(conf_min))


def read_detections(path: str | Path, conf_min: float | None = None,
                    half: bool = True) -> pd.DataFrame:
    """Load a cached detection table, optionally applying a confidence floor.

    Deliberately does **not** go through :func:`..video_io.load_table`. That
    helper calls ``decategorize``, which expands every label column into one
    Python string per row -- right for the small tables it was written for, and
    catastrophic here. See :data:`LABEL_COLUMNS` for the measured cost.

    On 2026-08-27 that expansion, plus the boolean-mask copy ``apply_conf_floor``
    needs on top of it, drove the replay past 31 GB of RAM and 19 GB of swap.
    The OOM killer took the job three times, and the thrashing took the desktop
    with it. So this reader:

    * keeps the label columns dictionary-encoded from the file onwards
      (they are already ``RLE_DICTIONARY`` on disk, ~0.02 MB per row group, so
      the strings are never materialised at all); and
    * pushes the confidence floor down into pyarrow, ahead of ``to_pandas``, so
      the filtered copy is the only pandas table that ever exists.

    Same rows, same values, same fingerprint -- 22.3 GB becomes 1.3 GB.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        # CSV fallback caches are small and rare; the memory problem cannot arise.
        df = load_table(path)
        if df is None or df.empty:
            return empty_detection_table()
        if conf_min is not None:
            df = apply_conf_floor(df, conf_min, half=half)
        return df.reset_index(drop=True)

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    present = set(pq.ParquetFile(path).schema_arrow.names)
    pf = pq.ParquetFile(path, read_dictionary=[c for c in LABEL_COLUMNS if c in present])
    if pf.metadata.num_rows == 0:
        return empty_detection_table()

    # Row group at a time, filtering as we go. Reading the whole file first and
    # filtering after needs the unfiltered table AND the filtered copy live at
    # once -- measured over 6 GB on the dense Culex cache even with the labels
    # dictionary-encoded. This way the only thing that accumulates is what
    # survives the floor, alongside one ~1 M-row group.
    thresh = conf_threshold(conf_min, half) if conf_min is not None else None
    pieces = []
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i)
        if thresh is not None:
            # Must reproduce Ultralytics exactly: a strict `>` against the
            # fp16-promoted threshold. fp16 -> float32 is always exact and the
            # stored confidences are float32, so comparing in the column's own
            # type is equivalent to the float64 promotion apply_conf_floor uses.
            # Not one box moves, including those sitting on the threshold.
            conf = tbl.column("confidence")
            tbl = tbl.filter(pc.greater(conf, pa.scalar(thresh, type=conf.type)))
        if tbl.num_rows:
            pieces.append(tbl)

    # A zero-row slice rather than empty_detection_table(): it keeps the file's
    # column dtypes, which the object-dtype fallback would flatten.
    table = (pa.concat_tables(pieces) if pieces
             else pf.read_row_group(0).slice(0, 0))
    del pieces

    # A dictionary-encoded ChunkedArray may carry a different dictionary per row
    # group; pandas needs one shared set of categories, so unify before convert.
    table = pa.table({
        name: (col.unify_dictionaries()
               if pa.types.is_dictionary(col.type) and hasattr(col, "unify_dictionaries")
               else col)
        for name, col in zip(table.schema.names, table.columns)
    })
    # split_blocks + self_destruct convert one column at a time and release each
    # arrow buffer as it goes. A plain to_pandas() instead consolidates every
    # int64 column into one contiguous block while the arrow table is still
    # live, which doubles peak memory -- it was the last thing still failing
    # under a 6 GB ceiling on the dense cache.
    #
    # No reset_index: a freshly built frame is already indexed 0..n-1, and
    # resetting would buy another full copy of the table this exists to shrink.
    return table.to_pandas(split_blocks=True, self_destruct=True)


def iter_frame_detections(det: pd.DataFrame) -> Iterator[FrameDetections]:
    """Yield detections frame by frame in processed-frame order.

    Frames with no detections are *not* skipped when they fall inside the
    observed range: a tracker must still see them, because an empty frame is
    what ages out a lost track. Skipping them would quietly lengthen tracks.
    """
    if det is None or det.empty:
        return

    # The cache is written in frame order, so this sort is almost always a
    # no-op that costs a full copy of the table -- another 1.3 GB on a dense
    # video. Pay for it only when the order is actually wrong.
    if not det["processed_frame"].is_monotonic_increasing:
        det = det.sort_values(["processed_frame"], kind="stable")
    pf = det["processed_frame"].to_numpy()
    arr = det[DET_ARRAY_COLUMNS].to_numpy(dtype=np.float32)
    rows = np.arange(len(det))
    src = det["source_frame"].to_numpy()
    ts = det["time_s"].to_numpy(dtype=float)
    W = int(pd.to_numeric(det["frame_width"], errors="coerce").dropna().iloc[0])
    H = int(pd.to_numeric(det["frame_height"], errors="coerce").dropna().iloc[0])

    # Frame boundaries in the sorted table.
    starts = np.searchsorted(pf, np.arange(pf[0], pf[-1] + 1), side="left")
    ends = np.searchsorted(pf, np.arange(pf[0], pf[-1] + 1), side="right")

    # A frame with no detections has start == end; reconstruct its identity from
    # the cadence rather than from rows that do not exist.
    stride = _infer_stride(det)
    first_src, first_pf = int(src[0]), int(pf[0])
    fps_dt = _infer_dt(det)

    for k, p in enumerate(range(int(pf[0]), int(pf[-1]) + 1)):
        s, e = int(starts[k]), int(ends[k])
        if s < e:
            yield FrameDetections(
                processed_frame=p,
                source_frame=int(src[s]),
                time_s=float(ts[s]),
                array=arr[s:e],
                row_index=rows[s:e],
                frame_width=W,
                frame_height=H,
            )
        else:
            sf = first_src + (p - first_pf) * stride
            yield FrameDetections(
                processed_frame=p,
                source_frame=int(sf),
                time_s=float(first_pf * 0 + (ts[0] + (p - first_pf) * fps_dt))
                if np.isfinite(fps_dt) else float("nan"),
                array=np.zeros((0, len(DET_ARRAY_COLUMNS)), dtype=np.float32),
                row_index=np.zeros(0, dtype=int),
                frame_width=W,
                frame_height=H,
            )


def _infer_stride(det: pd.DataFrame) -> int:
    """Source-frame step between consecutive processed frames."""
    pairs = det[["processed_frame", "source_frame"]].drop_duplicates()
    if len(pairs) < 2:
        return 1
    pairs = pairs.sort_values("processed_frame")
    dp = np.diff(pairs["processed_frame"].to_numpy())
    ds = np.diff(pairs["source_frame"].to_numpy())
    ok = dp > 0
    return int(round(float(np.median(ds[ok] / dp[ok])))) if ok.any() else 1


def _infer_dt(det: pd.DataFrame) -> float:
    """Seconds between consecutive processed frames."""
    pairs = det[["processed_frame", "time_s"]].drop_duplicates()
    if len(pairs) < 2:
        return float("nan")
    pairs = pairs.sort_values("processed_frame")
    dp = np.diff(pairs["processed_frame"].to_numpy())
    dt = np.diff(pairs["time_s"].to_numpy(dtype=float))
    ok = dp > 0
    return float(np.median(dt[ok] / dp[ok])) if ok.any() else float("nan")


def detection_fingerprint(det: pd.DataFrame) -> dict:
    """A cheap identity for a detection pool, for the §31 fairness assertions.

    Every arm must be shown to have received exactly this. Comparing the
    fingerprint is how the benchmark proves the detector was held fixed rather
    than merely asserting it.
    """
    if det is None or det.empty:
        return {"n_detections": 0, "n_frames": 0, "frame_hash": None, "box_hash": None}
    pf = np.ascontiguousarray(det["processed_frame"].to_numpy(dtype=np.int64))
    box = np.ascontiguousarray(
        det[["x1", "y1", "x2", "y2", "confidence"]].to_numpy(dtype=np.float64)
    )
    import hashlib

    return {
        "n_detections": int(len(det)),
        "n_frames": int(det["processed_frame"].nunique()),
        "first_frame": int(pf.min()),
        "last_frame": int(pf.max()),
        "frame_hash": hashlib.sha256(pf.tobytes()).hexdigest()[:16],
        "box_hash": hashlib.sha256(box.tobytes()).hexdigest()[:16],
    }
