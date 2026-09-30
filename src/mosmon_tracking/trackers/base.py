"""Common tracker interface for the multi-tracker benchmark.

Every arm of the benchmark receives the *same* frame-level detections and must
return tracks in one canonical format, so that the evaluation never has to know
which association algorithm produced a table.

Detections in, per frame::

    (N, 6) float array: x1, y1, x2, y2, confidence, class_id

Tracks out, per frame::

    (M, 8) float array: x1, y1, x2, y2, track_id, confidence, class_id, det_idx

``det_idx`` indexes back into the detection array of that same frame. It is what
makes the cross-tracker identity-agreement analysis possible: because all arms
see an identical detection pool, two arms' outputs can be compared as two
partitions of one shared set of observations, with no matching step and no
ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

#: Columns of the per-frame array an adapter returns from :meth:`TrackerAdapter.update`.
TRACK_OUTPUT_COLUMNS = (
    "x1", "y1", "x2", "y2", "track_id", "confidence", "class_id", "det_idx",
)

#: Columns of the per-frame detection array an adapter accepts.
DETECTION_INPUT_COLUMNS = ("x1", "y1", "x2", "y2", "confidence", "class_id")


def empty_track_output() -> np.ndarray:
    return np.zeros((0, len(TRACK_OUTPUT_COLUMNS)), dtype=np.float32)


class TrackerAdapter(ABC):
    """One association algorithm, driven frame by frame from fixed detections."""

    #: Arm name as it appears in every output table and figure.
    name: str = "tracker"

    #: Whether :meth:`update` needs the decoded frame. Frame-free arms can be
    #: replayed straight from a detection table with no video decode at all,
    #: which is roughly two orders of magnitude cheaper.
    needs_frames: bool = False

    @abstractmethod
    def reset(self) -> None:
        """Drop all state and restart track numbering. Call between videos."""

    @abstractmethod
    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> np.ndarray:
        """Associate one frame of detections; return the canonical track array."""

    def finalize(self) -> None:
        """Release any resources. Default is a no-op."""
        return None

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Every resolved parameter that governs this arm's behaviour."""

    def provenance(self) -> dict[str, Any]:
        """Implementation identity, for the benchmark metadata (§28)."""
        return {"name": self.name, "adapter": type(self).__name__}
