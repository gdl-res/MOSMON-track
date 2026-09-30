"""Pixel-to-physical calibration.

Three modes are supported, in increasing order of fidelity:

1. ``pixel``    - no calibration; all distances stay in pixels (the honest default).
2. ``scalar``   - a single ``pixels_per_mm`` scale factor (assumes a flat, fronto-parallel plane).
3. ``homography`` - a 3x3 matrix mapping image points to container-plane coordinates (mm).

We never silently pretend pixels are millimetres: if no calibration is supplied,
:meth:`Calibrator.to_mm` returns ``None`` and downstream code keeps pixel units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CalibrationConfig


@dataclass
class Calibrator:
    mode: str = "pixel"  # pixel | scalar | homography
    pixels_per_mm: float | None = None
    homography: np.ndarray | None = None  # 3x3, image -> plane(mm)

    @property
    def is_physical(self) -> bool:
        return self.mode in ("scalar", "homography")

    @classmethod
    def from_config(cls, cfg: CalibrationConfig) -> Calibrator:
        if not cfg.enabled:
            return cls(mode="pixel")
        if cfg.homography_path:
            H = load_homography(cfg.homography_path)
            return cls(mode="homography", homography=H)
        if cfg.pixels_per_mm:
            if cfg.pixels_per_mm <= 0:
                raise ValueError("pixels_per_mm must be positive")
            return cls(mode="scalar", pixels_per_mm=float(cfg.pixels_per_mm))
        return cls(mode="pixel")

    def to_mm(self, xy_px: np.ndarray) -> np.ndarray | None:
        """Map an (N, 2) array of pixel coordinates to millimetres, or None if uncalibrated."""
        xy = np.asarray(xy_px, dtype=float)
        if self.mode == "scalar" and self.pixels_per_mm:
            return xy / self.pixels_per_mm
        if self.mode == "homography" and self.homography is not None:
            ones = np.ones((xy.shape[0], 1))
            hom = np.hstack([xy, ones])  # (N, 3)
            mapped = hom @ self.homography.T  # (N, 3)
            w = mapped[:, 2:3]
            w[w == 0] = np.nan
            return mapped[:, :2] / w
        return None

    def distance_mm(self, d_px: np.ndarray | float) -> np.ndarray | float | None:
        """Convert a pixel distance to mm (scalar mode only; homography is non-uniform)."""
        if self.mode == "scalar" and self.pixels_per_mm:
            return np.asarray(d_px, dtype=float) / self.pixels_per_mm
        return None

    @property
    def units(self) -> str:
        return "mm" if self.is_physical else "px"


def load_homography(path: str | Path) -> np.ndarray:
    """Load a 3x3 homography from a JSON (``{"homography": [[...]]}``) or .npy file."""
    path = Path(path)
    if path.suffix == ".npy":
        H = np.load(path)
    else:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        H = np.asarray(data["homography"], dtype=float)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H


def estimate_scalar_from_container(
    container_width_px: float, container_width_cm: float
) -> float:
    """Estimate ``pixels_per_mm`` from a known container width.

    Dataset reference dimensions (MOSMON-Larvae acquisition setup): standard container ~38x16 cm,
    large container ~52x17 cm. This is an aid, not an automatic calibration.
    """
    if container_width_cm <= 0:
        raise ValueError("container_width_cm must be positive")
    return container_width_px / (container_width_cm * 10.0)
