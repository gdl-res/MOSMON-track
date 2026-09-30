"""Export tracks to the MOTChallenge format for interoperable evaluation.

The MOTChallenge ``gt.txt`` / result format is one comma-separated line per
detected object per frame::

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Frames are 1-based; for 2-D tracking the world coordinates ``x, y, z`` are set
to ``-1``. Writing this lets the tracks be scored with TrackEval / py-motmetrics
(MOTA, IDF1, HOTA) against ground truth, and consumed by other MOT tooling.

A ``seqinfo.ini`` is written alongside when sequence metadata is available, so
the output drops straight into a TrackEval sequence folder.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pandas as pd

MOT_COLUMNS = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]


def tracks_to_mot(tracks: pd.DataFrame, include_interpolated: bool = True) -> pd.DataFrame:
    """Convert a cleaned/raw tracks table to a MOTChallenge-format DataFrame.

    Parameters
    ----------
    tracks:
        Per-frame track table with at least ``frame_idx``, ``track_id``,
        ``x1``, ``y1`` and either ``w``/``h`` or ``x2``/``y2``. ``confidence``
        and ``is_interpolated`` are used when present.
    include_interpolated:
        When ``False``, rows flagged ``is_interpolated`` are dropped so the
        export contains only genuine detections (closer to a detector's output).

    Returns
    -------
    DataFrame with the ten MOT columns, sorted by frame then id. ``frame`` is
    1-based (MOT convention; our ``frame_idx`` is 0-based).
    """
    required = {"frame_idx", "track_id", "x1", "y1"}
    if tracks is None or tracks.empty or not required.issubset(tracks.columns):
        return pd.DataFrame(columns=MOT_COLUMNS)

    df = tracks
    if not include_interpolated and "is_interpolated" in df.columns:
        df = df[~df["is_interpolated"].fillna(False).astype(bool)]
    if df.empty:
        return pd.DataFrame(columns=MOT_COLUMNS)

    w = df["w"] if "w" in df.columns else (df["x2"] - df["x1"])
    h = df["h"] if "h" in df.columns else (df["y2"] - df["y1"])
    conf = df["confidence"] if "confidence" in df.columns else 1.0

    out = pd.DataFrame({
        "frame": df["frame_idx"].astype(int) + 1,  # MOT frames are 1-based
        "id": df["track_id"].astype(int),
        "bb_left": df["x1"].astype(float),
        "bb_top": df["y1"].astype(float),
        "bb_width": pd.Series(w, index=df.index).astype(float),
        "bb_height": pd.Series(h, index=df.index).astype(float),
        # interpolated rows have no detector score; treat them as certain.
        "conf": pd.Series(conf, index=df.index).astype(float).fillna(1.0),
        "x": -1,
        "y": -1,
        "z": -1,
    })
    return out.sort_values(["frame", "id"]).reset_index(drop=True)


def write_mot(
    tracks: pd.DataFrame,
    path: str | Path,
    include_interpolated: bool = True,
    seqinfo: dict | None = None,
) -> Path:
    """Write ``tracks`` to a MOTChallenge ``.txt`` and return the path.

    Always writes the file (empty if there are no tracks) so downstream tooling
    can rely on its presence. When ``seqinfo`` is given (keys among ``name``,
    ``seq_length``, ``frame_rate``, ``im_width``, ``im_height``), a sibling
    ``seqinfo.ini`` is written for TrackEval compatibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mot = tracks_to_mot(tracks, include_interpolated=include_interpolated)
    # %g keeps integer-valued coords compact while preserving sub-pixel boxes.
    mot.to_csv(path, header=False, index=False, float_format="%g")

    if seqinfo:
        _write_seqinfo(path.parent / "seqinfo.ini", seqinfo)
    return path


def _write_seqinfo(path: Path, info: dict) -> Path:
    """Write a minimal MOTChallenge ``seqinfo.ini``."""
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve MOT's CamelCase keys
    section = {"name": "Sequence", "imExt": ".jpg"}
    if info.get("name"):
        section["name"] = str(info["name"])
    if info.get("frame_rate"):
        section["frameRate"] = str(int(round(float(info["frame_rate"]))))
    if info.get("seq_length"):
        section["seqLength"] = str(int(info["seq_length"]))
    if info.get("im_width"):
        section["imWidth"] = str(int(info["im_width"]))
    if info.get("im_height"):
        section["imHeight"] = str(int(info["im_height"]))
    cfg["Sequence"] = section
    with path.open("w", encoding="utf-8") as fh:
        cfg.write(fh)
    return path
