"""Export per-track trajectories as a GeoJSON FeatureCollection.

Each track becomes one feature: a ``LineString`` of its centre points over time
(or a ``Point`` for a single-frame track), with the track's summary statistics
attached as feature ``properties``. This is a compact, interoperable per-track
view that drops straight into GIS/plotting tools (QGIS, kepler.gl, geopandas),
complementing the row-per-frame ``tracks_clean`` table.

Coordinates are image pixels ``[cx, cy]`` by default (note: image ``y`` points
*down*). When a physical calibrator is supplied, millimetre coordinates are used
instead and ``crs_units`` in the collection metadata says so.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibrator


def _clean_value(v):
    """Coerce numpy/pandas scalars to JSON-safe Python values (NaN -> None)."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def tracks_to_geojson(
    clean: pd.DataFrame,
    track_summary: pd.DataFrame | None = None,
    calibrator: Calibrator | None = None,
) -> dict:
    """Build a GeoJSON ``FeatureCollection`` with one feature per track.

    Parameters
    ----------
    clean:
        Per-frame track table with at least ``track_id``, ``cx``, ``cy`` and
        (for ordering) ``frame_idx``.
    track_summary:
        Optional one-row-per-track table; matching rows are attached as feature
        ``properties`` (duration, speed, tortuosity, quality_score, ...).
    calibrator:
        When physical, centre points are mapped to millimetres.
    """
    units = "px"
    coords_physical = calibrator is not None and calibrator.is_physical
    if coords_physical:
        units = calibrator.units

    collection = {
        "type": "FeatureCollection",
        "metadata": {"crs_units": units, "note": "image y increases downward"},
        "features": [],
    }
    required = {"track_id", "cx", "cy"}
    if clean is None or clean.empty or not required.issubset(clean.columns):
        return collection

    summary_by_id: dict = {}
    if track_summary is not None and not track_summary.empty and "track_id" in track_summary.columns:
        for r in track_summary.to_dict("records"):
            summary_by_id[int(r["track_id"])] = r

    sort_cols = [c for c in ("track_id", "frame_idx") if c in clean.columns]
    df = clean.sort_values(sort_cols) if sort_cols else clean

    for tid, g in df.groupby("track_id", sort=True):
        xy = g[["cx", "cy"]].to_numpy(dtype=float)
        if coords_physical:
            mapped = calibrator.to_mm(xy)
            if mapped is not None:
                xy = mapped
        coords = [[c[0], c[1]] for c in xy if np.isfinite(c[0]) and np.isfinite(c[1])]
        if not coords:
            continue
        geometry = (
            {"type": "Point", "coordinates": coords[0]}
            if len(coords) == 1
            else {"type": "LineString", "coordinates": coords}
        )

        props = {k: _clean_value(v) for k, v in summary_by_id.get(int(tid), {}).items()}
        props.setdefault("track_id", int(tid))
        props.setdefault("n_points", len(coords))
        collection["features"].append(
            {"type": "Feature", "geometry": geometry, "properties": props}
        )
    return collection


def write_geojson(collection: dict, path: str | Path) -> Path:
    """Write a GeoJSON dict to ``path`` (pretty-printed, JSON-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=2, allow_nan=False)
    return path
