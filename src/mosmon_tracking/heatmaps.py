"""Spatial heatmaps: occupancy, dwell-time, speed, and activity.

Heatmaps are computed on the *normalized* image plane (cx_norm, cy_norm in
[0, 1]) so they are comparable across resolutions. Arrays are returned with shape
(bins_y, bins_x) — i.e. row = y, column = x — which is the natural orientation
for ``imshow(origin='upper')`` overlaid on a frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

EPS = 1e-12


@dataclass
class Heatmap:
    name: str
    array: np.ndarray  # (bins_y, bins_x)
    kind: str  # occupancy | dwell | speed | activity
    normalize: str
    class_name: str | None = None

    def as_dataframe(self) -> pd.DataFrame:
        ny, nx = self.array.shape
        ys, xs = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return pd.DataFrame(
            {"bin_y": ys.ravel(), "bin_x": xs.ravel(), "value": self.array.ravel()}
        )


def _hist2d(cx_norm, cy_norm, bins_x, bins_y, weights=None):
    """2D histogram with x->columns, y->rows. Returns (bins_y, bins_x)."""
    H, _, _ = np.histogram2d(
        cy_norm, cx_norm, bins=[bins_y, bins_x], range=[[0, 1], [0, 1]], weights=weights
    )
    return H


def _normalize(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "probability":
        s = arr.sum()
        return arr / s if s > EPS else arr
    if mode == "density":
        m = arr.max()
        return arr / m if m > EPS else arr
    return arr  # count


def occupancy_heatmap(df: pd.DataFrame, cfg: Config) -> Heatmap:
    bx, by = cfg.heatmaps.bins_x, cfg.heatmaps.bins_y
    H = _hist2d(df["cx_norm"], df["cy_norm"], bx, by)
    return Heatmap("occupancy", _normalize(H, cfg.heatmaps.normalize), "occupancy",
                   cfg.heatmaps.normalize)


def dwell_time_heatmap(df: pd.DataFrame, cfg: Config) -> Heatmap:
    """Time spent per bin, weighting each detection by its frame dt."""
    bx, by = cfg.heatmaps.bins_x, cfg.heatmaps.bins_y
    w = df["dt"].to_numpy(dtype=float) if "dt" in df.columns else None
    if w is not None:
        w = np.where(np.isfinite(w), w, 0.0)
    H = _hist2d(df["cx_norm"], df["cy_norm"], bx, by, weights=w)
    return Heatmap("dwell_time", _normalize(H, cfg.heatmaps.normalize), "dwell",
                   cfg.heatmaps.normalize)


def speed_heatmap(df: pd.DataFrame, cfg: Config) -> Heatmap:
    """Mean speed per bin = sum(speed) / count, with empty bins as NaN."""
    bx, by = cfg.heatmaps.bins_x, cfg.heatmaps.bins_y
    sp = df["speed_px_s"].to_numpy(dtype=float)
    valid = np.isfinite(sp)
    sub = df[valid]
    total = _hist2d(sub["cx_norm"], sub["cy_norm"], bx, by, weights=sp[valid])
    count = _hist2d(sub["cx_norm"], sub["cy_norm"], bx, by)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return Heatmap("speed", mean, "speed", "mean")


def activity_heatmap(df: pd.DataFrame, cfg: Config) -> Heatmap:
    """Fraction of active/burst frames per bin."""
    bx, by = cfg.heatmaps.bins_x, cfg.heatmaps.bins_y
    if "movement_state" not in df.columns:
        H = _hist2d(df["cx_norm"], df["cy_norm"], bx, by)
        return Heatmap("activity", _normalize(H, "density"), "activity", "density")
    active = df["movement_state"].isin(["active", "burst"]).to_numpy(dtype=float)
    total = _hist2d(df["cx_norm"], df["cy_norm"], bx, by, weights=active)
    count = _hist2d(df["cx_norm"], df["cy_norm"], bx, by)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(count > 0, total / count, np.nan)
    return Heatmap("activity", frac, "activity", "fraction")


def heatmap_entropy(hm: Heatmap) -> float:
    """Shannon entropy (nats) of a heatmap treated as a spatial distribution."""
    p = hm.array[np.isfinite(hm.array)]
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def compute_all_heatmaps(df: pd.DataFrame, cfg: Config) -> list[Heatmap]:
    """Compute the standard heatmap set, optionally split per class."""
    if df.empty:
        return []
    builders = [occupancy_heatmap, dwell_time_heatmap, speed_heatmap, activity_heatmap]
    out: list[Heatmap] = [b(df, cfg) for b in builders]
    if cfg.heatmaps.per_class and "class_name" in df.columns and df["class_name"].nunique() > 1:
        for cls, g in df.groupby("class_name"):
            for b in builders:
                hm = b(g, cfg)
                hm.name = f"{hm.name}_{cls}"
                hm.class_name = str(cls)
                out.append(hm)
    return out
