"""Replay a frozen detection cache through every tracker arm.

All arms are driven from **one** pass over the video. That matters twice over:

* Cost — decoding 4K/8K footage dominates the runtime of the arms that need
  frames (global motion compensation, ReID crops). Decoding once instead of once
  per arm roughly halves the corpus run.
* Fairness — every arm provably sees the same detections in the same order, on
  the same frames, with the same timestamps, because there is only one source of
  them. The §31 assertions then verify what the structure already guarantees.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..trackers.base import TrackerAdapter
from ..yolo_tracker import RAW_TRACK_COLUMNS, empty_raw_table
from .detections_io import detection_fingerprint, iter_frame_detections


@dataclass
class ArmResult:
    """One arm's output for one video."""

    name: str
    tracks: pd.DataFrame                 # RAW_TRACK_COLUMNS
    assignments: pd.DataFrame            # processed_frame, det_row, track_id
    association_seconds: float = 0.0
    n_frames: int = 0
    n_output_rows: int = 0
    params: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def replay(
    det: pd.DataFrame,
    arms: list[TrackerAdapter],
    video_path: str | Path | None = None,
    model_name: str = "",
    model_path: str = "",
    progress_every: int = 0,
) -> tuple[dict[str, ArmResult], dict]:
    """Drive every arm over one detection table; return per-arm results + stats.

    ``video_path`` is required only if some arm reports ``needs_frames``.
    """
    if det is None or det.empty:
        return ({a.name: ArmResult(a.name, empty_raw_table(), _empty_assignments(),
                                   params=a.params(), provenance=a.provenance())
                 for a in arms},
                {"detections": detection_fingerprint(det), "decode_seconds": 0.0})

    need_frames = any(a.needs_frames for a in arms)
    if need_frames and video_path is None:
        needing = [a.name for a in arms if a.needs_frames]
        raise ValueError(f"arms {needing} need decoded frames but no video_path was given")

    for a in arms:
        a.reset()

    meta = _video_meta(det, video_path, model_name, model_path)
    rows: dict[str, list[pd.DataFrame]] = {a.name: [] for a in arms}
    assign: dict[str, list[pd.DataFrame]] = {a.name: [] for a in arms}
    assoc_s: dict[str, float] = {a.name: 0.0 for a in arms}
    decode_s = 0.0
    n_frames = 0

    reader = _FrameReader(video_path) if need_frames else None
    try:
        for fd in iter_frame_detections(det):
            frame = None
            if reader is not None:
                t0 = time.perf_counter()
                frame = reader.read(fd.source_frame)
                decode_s += time.perf_counter() - t0
                if frame is None:
                    # Ran past the end of the video; detections exist for frames
                    # the decoder cannot reach. Stop rather than feed None to a
                    # tracker that needs the image.
                    break
            n_frames += 1
            for a in arms:
                t0 = time.perf_counter()
                out = a.update(fd.array, frame=frame if a.needs_frames else None,
                               timestamp=fd.time_s)
                assoc_s[a.name] += time.perf_counter() - t0
                if len(out):
                    rows[a.name].append(_to_raw_rows(out, fd, meta, a.name))
                    assign[a.name].append(_to_assignments(out, fd))
            if progress_every and n_frames % progress_every == 0:
                print(f"  {n_frames} frames", flush=True)
    finally:
        if reader is not None:
            reader.close()
        for a in arms:
            a.finalize()

    results = {}
    for a in arms:
        if rows[a.name]:
            tracks = pd.concat(rows[a.name], ignore_index=True)
            rows[a.name].clear()          # the per-frame frames are dead now
            tracks = _attach_labels(tracks, meta, a.name)
        else:
            tracks = empty_raw_table()
        assignments = (pd.concat(assign[a.name], ignore_index=True)
                       if assign[a.name] else _empty_assignments())
        results[a.name] = ArmResult(
            name=a.name, tracks=tracks, assignments=assignments,
            association_seconds=assoc_s[a.name], n_frames=n_frames,
            n_output_rows=len(tracks), params=a.params(), provenance=a.provenance(),
        )
    stats = {
        "detections": detection_fingerprint(det),
        "decode_seconds": decode_s,
        "n_frames_replayed": n_frames,
    }
    return results, stats


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _empty_assignments() -> pd.DataFrame:
    return pd.DataFrame({"processed_frame": pd.Series(dtype="int64"),
                         "det_row": pd.Series(dtype="int64"),
                         "track_id": pd.Series(dtype="int64")})


def _video_meta(det: pd.DataFrame, video_path, model_name: str, model_path: str) -> dict:
    first = det.iloc[0]
    return {
        "video_path": str(video_path) if video_path else str(first.get("video_path", "")),
        "video_name": str(first.get("video_name", "")),
        "model_path": model_path or str(first.get("model_path", "")),
        "model_name": model_name or str(first.get("model_name", "")),
    }


def _to_raw_rows(out: np.ndarray, fd, meta: dict, arm: str) -> pd.DataFrame:
    """(M, 8) tracker output -> numeric raw-track rows for one frame.

    The six label columns (``video_path`` .. ``tracker``, ``class_name``) are
    deliberately NOT built here. They are constant, or near-constant, for the
    whole video, and materialising them per frame is what made this loop
    unaffordable: on a dense video they accounted for 4.0 GB of a 4.7 GB raw
    table, and the biggest videos are three times larger again. They are
    attached once, as dictionary-encoded categoricals, in
    :func:`_attach_labels` after the per-frame frames are concatenated.
    """
    x1, y1, x2, y2 = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    W, H = fd.frame_width, fd.frame_height
    return pd.DataFrame({
        # frame_idx stays the ORIGINAL-video index, matching the rest of the
        # pipeline: track_postprocess infers the sampling step from it and all
        # kinematics use time_s, never the index.
        "frame_idx": fd.source_frame,
        "time_s": fd.time_s,
        "track_id": out[:, 4].astype(int),
        "class_id": out[:, 6].astype(int),
        "confidence": out[:, 5].astype(float),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": w, "h": h, "cx": cx, "cy": cy,
        "cx_norm": cx / W if W else np.nan,
        "cy_norm": cy / H if H else np.nan,
        "frame_width": W, "frame_height": H,
    })


def _constant_column(value: str, n: int) -> pd.Categorical:
    """A one-category column: n int8 codes plus one string, not n strings."""
    return pd.Categorical.from_codes(np.zeros(n, dtype=np.int8), [str(value)])


def _attach_labels(tracks: pd.DataFrame, meta: dict, arm: str) -> pd.DataFrame:
    """Add the label columns to a concatenated raw table, as categoricals.

    ``class_name`` is derived from ``class_id`` through the model's name map
    rather than carried per row, so the whole set of labels costs one int8 code
    per row. :func:`mosmon_tracking.video_io.load_table` decodes these back to
    plain strings on read, so nothing downstream sees the difference.
    """
    n = len(tracks)
    for col, value in (("video_path", meta["video_path"]),
                       ("video_name", meta["video_name"]),
                       ("model_path", meta["model_path"]),
                       ("model_name", meta["model_name"]),
                       ("tracker", arm)):
        tracks[col] = _constant_column(value, n)

    names = _class_names(None)
    cls = tracks["class_id"].to_numpy()
    uniq = np.unique(cls)
    cats = [names.get(int(c), str(int(c))) for c in uniq]
    codes = np.searchsorted(uniq, cls).astype(np.int16)
    tracks["class_name"] = pd.Categorical.from_codes(codes, cats)
    return tracks[RAW_TRACK_COLUMNS]


_CLASS_NAME_CACHE: dict[int, str] = {}


def _class_names(fd) -> dict[int, str]:
    return _CLASS_NAME_CACHE


def set_class_names(names: dict) -> None:
    """Register the model's class names so track tables carry readable labels."""
    _CLASS_NAME_CACHE.clear()
    _CLASS_NAME_CACHE.update({int(k): str(v) for k, v in (names or {}).items()})


def _to_assignments(out: np.ndarray, fd) -> pd.DataFrame:
    """Map each emitted track back to the detection row it consumed.

    This is what lets two arms be compared as two partitions of one shared set of
    observations, which is the only rigorous identity comparison available
    without ground truth.
    """
    det_idx = out[:, 7].astype(int)
    valid = (det_idx >= 0) & (det_idx < len(fd.row_index))
    return pd.DataFrame({
        "processed_frame": np.full(int(valid.sum()), fd.processed_frame, dtype=np.int64),
        "det_row": fd.row_index[det_idx[valid]].astype(np.int64),
        "track_id": out[valid, 4].astype(np.int64),
    })


class _FrameReader:
    """Sequential video reader that serves frames by original-video index."""

    def __init__(self, video_path: str | Path) -> None:
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video for replay: {video_path}")
        self.pos = 0

    def read(self, source_frame: int) -> np.ndarray | None:
        """Advance to ``source_frame`` and return it, or None past end of video."""
        if source_frame < self.pos:
            raise ValueError(
                f"frames must be requested in order; asked for {source_frame} "
                f"after {self.pos}"
            )
        while self.pos < source_frame:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, frame = self.cap.read()
        self.pos += 1
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()
