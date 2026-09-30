"""Fairness and sanity assertions for the benchmark (§31).

The benchmark's central claim is that detection was held fixed and only
association varied. That claim has to be *checked*, not assumed: these functions
turn it into assertions over the written artifacts, so a silent regression in
the replay path shows up as a failed check rather than as a plausible-looking
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Directories under the benchmark root that hold benchmark bookkeeping rather
#: than an arm's runs. Listed explicitly so a new output directory can never be
#: silently scored as if it were a tracker.
NON_ARM_DIRS = frozenset({
    "detections_canonical", "downstream", "evaluation", "figures", "reports",
})


def find_arm_dirs(benchmark_dir: str | Path) -> list[Path]:
    """Arm directories under a benchmark root, in name order."""
    root = Path(benchmark_dir)
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name not in NON_ARM_DIRS
    )


@dataclass
class CheckReport:
    """Outcome of a group of checks."""

    name: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def check(self, condition: bool, message: str) -> None:
        (self.passed if condition else self.failed).append(message)

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok,
                "passed": self.passed, "failed": self.failed}


def check_detection_identity(fingerprints: dict[str, dict]) -> CheckReport:
    """Every arm must have received an identical detection pool.

    Compares the fingerprints recorded when each arm ran. Detection counts,
    frame lists and box coordinates must all agree; anything else means the
    comparison is measuring the detector, not the association algorithm.
    """
    rep = CheckReport("detection_identity")
    if not fingerprints:
        rep.check(False, "no fingerprints recorded")
        return rep
    names = sorted(fingerprints)
    ref_name = names[0]
    ref = fingerprints[ref_name]
    for key, label in [("n_detections", "detection count"),
                       ("n_frames", "frame count"),
                       ("frame_hash", "frame list"),
                       ("box_hash", "box coordinates and confidences")]:
        differing = [n for n in names if fingerprints[n].get(key) != ref.get(key)]
        rep.check(
            not differing,
            f"identical {label} across arms"
            if not differing
            else f"{label} differs from {ref_name!r} for: {differing}",
        )
    return rep


def check_track_table(tracks: pd.DataFrame, arm: str,
                      frame_width: int | None = None,
                      frame_height: int | None = None) -> CheckReport:
    """Structural validity of one arm's raw track table."""
    rep = CheckReport(f"track_output[{arm}]")
    if tracks is None or tracks.empty:
        rep.check(False, "track table is empty")
        return rep

    dup = tracks.duplicated(subset=["frame_idx", "track_id"]).sum()
    rep.check(dup == 0, "no duplicate (frame, track_id) records"
              if dup == 0 else f"{dup} duplicate (frame, track_id) records")

    w = tracks["x2"] - tracks["x1"]
    h = tracks["y2"] - tracks["y1"]
    bad = int(((w <= 0) | (h <= 0)).sum())
    rep.check(bad == 0, "all boxes have positive width and height"
              if bad == 0 else f"{bad} boxes with non-positive width/height")

    # Timestamps must not go backwards within a track.
    ordered = tracks.sort_values(["track_id", "frame_idx"])
    dt = ordered.groupby("track_id", sort=False)["time_s"].diff()
    nonmono = int((dt.dropna() <= 0).sum())
    rep.check(nonmono == 0, "timestamps strictly increase within every track"
              if nonmono == 0 else f"{nonmono} non-increasing timestamps within tracks")

    W = frame_width or int(pd.to_numeric(tracks["frame_width"], errors="coerce").max())
    H = frame_height or int(pd.to_numeric(tracks["frame_height"], errors="coerce").max())
    if W and H:
        # A small tolerance: trackers extrapolate a box slightly past the edge.
        tol = 0.05
        oob = int((
            (tracks["x1"] < -tol * W) | (tracks["x2"] > (1 + tol) * W)
            | (tracks["y1"] < -tol * H) | (tracks["y2"] > (1 + tol) * H)
        ).sum())
        rep.check(oob == 0, f"all coordinates within frame bounds (+-{tol:.0%})"
                  if oob == 0 else f"{oob} boxes outside frame bounds by more than {tol:.0%}")

    ids = tracks["track_id"]
    rep.check(bool((ids > 0).all()), "all track ids are positive"
              if bool((ids > 0).all()) else "some track ids are non-positive")
    return rep


def check_detection_table(det: pd.DataFrame, max_det: int | None = None) -> CheckReport:
    """Validity of a cached detection table, plus the max_det truncation probe.

    Ultralytics caps detections per frame at ``max_det``. When that cap binds,
    the densest frames are silently truncated to the top-N by score — exactly the
    regime a crowding analysis is about. A spike at exactly ``max_det`` is the
    signature, so we look for it explicitly rather than trusting the setting.
    """
    rep = CheckReport("detection_table")
    if det is None or det.empty:
        rep.check(False, "detection table is empty")
        return rep

    w = det["x2"] - det["x1"]
    h = det["y2"] - det["y1"]
    bad = int(((w <= 0) | (h <= 0)).sum())
    rep.check(bad == 0, "all detection boxes have positive extent"
              if bad == 0 else f"{bad} detections with non-positive extent")

    pf = det["processed_frame"].to_numpy()
    rep.check(bool((np.diff(np.unique(pf)) == 1).all()),
              "processed_frame is contiguous")

    per_frame = det.groupby("processed_frame").size()
    if max_det:
        at_cap = int((per_frame >= max_det).sum())
        rep.check(at_cap == 0,
                  f"no frame reaches max_det={max_det}"
                  if at_cap == 0
                  else f"{at_cap} frames at or above max_det={max_det}: the densest "
                       "frames were truncated and crowding results are affected")
    rep.check(True, f"detections/frame: median {per_frame.median():.0f}, "
                    f"max {per_frame.max():.0f}")
    return rep


def summarise(reports: list[CheckReport]) -> dict:
    return {
        "all_ok": all(r.ok for r in reports),
        "n_failed": sum(len(r.failed) for r in reports),
        "reports": [r.as_dict() for r in reports],
    }
