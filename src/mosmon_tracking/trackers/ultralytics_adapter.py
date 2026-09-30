"""One adapter over all four in-tree Ultralytics trackers.

ByteTrack, BoT-SORT, OC-SORT and Deep OC-SORT all take ``args`` alone and share
``update(results, img=None, feats=None) -> (N, 8)``, so a single adapter covers
every arm of the benchmark and there is no per-tracker pipeline to keep in sync.

Two Ultralytics behaviours have to be worked around, and both are load-bearing
for the benchmark's correctness:

1. **The track-ID counter is global.** ``BaseTrack._count`` is a class attribute
   shared by every tracker instance in the process. Running several arms in one
   shared-decode loop would interleave their ID allocation, so an arm's output
   would depend on which other arms happened to run alongside it. Each adapter
   therefore keeps a private counter and swaps it in around every call.

2. **`model: auto` cannot work when replaying detections.** It resolves to a
   pass-through over detector backbone features that only exist inside the
   Ultralytics predict loop. An arm declaring ``with_reid`` must name a real
   encoder, and we build it ourselves with an explicit device — Ultralytics
   passes none, and ONNX Runtime then refuses to register the CUDA provider and
   silently falls back to CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import DETECTION_INPUT_COLUMNS, TrackerAdapter, empty_track_output


class ReidUnavailable(RuntimeError):
    """Raised when an arm declares appearance matching but no encoder can load.

    Never swallowed: an arm that silently loses its ReID encoder is no longer an
    appearance arm, and reporting it as one would be a false claim.
    """


class UltralyticsTrackerAdapter(TrackerAdapter):
    """Drive an in-tree Ultralytics tracker from a fixed detection array."""

    def __init__(
        self,
        name: str,
        tracker_yaml: str | Path,
        device: str | None = None,
        reid_fallback: str | None = None,
    ) -> None:
        from ..yolo_tracker import load_tracker_args, tracker_needs_frames

        self.name = name
        self.tracker_yaml = str(tracker_yaml)
        self.device = device
        self.reid_fallback = reid_fallback
        self.args = load_tracker_args(tracker_yaml)
        self.needs_frames = tracker_needs_frames(self.args)
        self._reid_model: str | None = None
        self._id_count = 0
        self._tracker = None
        self.reset()

    # -- lifecycle --------------------------------------------------------- #
    def reset(self) -> None:
        """Rebuild the tracker and restart this arm's private ID numbering."""
        from ..yolo_tracker import build_tracker_from_args

        wants_reid = bool(getattr(self.args, "with_reid", False))
        # Build the tracker with ReID switched off, then attach our own encoder.
        # Ultralytics' constructor would otherwise build one with no device set,
        # which makes ONNX Runtime drop the CUDA provider and fall back to CPU --
        # and it would be immediately discarded anyway.
        if wants_reid:
            self.args.with_reid = False
        try:
            self._tracker = build_tracker_from_args(self.args)
        finally:
            if wants_reid:
                self.args.with_reid = True
        self._id_count = 0
        if wants_reid:
            self._attach_encoder()

    def _attach_encoder(self) -> None:
        """Replace Ultralytics' encoder with one built on an explicit device."""
        from ultralytics.trackers.utils.reid import ReID

        model = getattr(self.args, "model", None)
        if model in (None, "", "auto"):
            raise ReidUnavailable(
                f"arm {self.name!r} sets with_reid but model={model!r}. 'auto' reuses "
                "detector backbone features that do not exist when replaying a "
                "detection table; name an explicit ReID encoder instead."
            )
        fallback = self.reid_fallback or getattr(self.args, "reid_fallback", None)
        candidates = [model] + ([fallback] if fallback else [])
        candidates = [self._resolve_encoder(c) for c in candidates]
        errors = []
        for cand in candidates:
            try:
                self._tracker.encoder = ReID(str(cand), device=self._reid_device())
                self._reid_model = str(cand)
                return
            except Exception as exc:  # noqa: BLE001 - reported verbatim below
                errors.append(f"{cand}: {type(exc).__name__}: {exc}")
        raise ReidUnavailable(
            f"arm {self.name!r} declares appearance matching but no encoder loaded. "
            "This arm must not be reported as an appearance arm. Tried -> "
            + " | ".join(errors)
        )

    #: Where a bare ReID asset name is looked for before Ultralytics is asked to
    #: fetch it. Keeping the weights in one known place stops every run from
    #: re-downloading the asset into whatever the working directory happens to be.
    REID_SEARCH_DIRS = ("weights/reid", "weights", ".")

    @staticmethod
    def _resolve_encoder(model: str) -> str:
        """Point a bare asset name at a local copy when one already exists."""
        path = Path(str(model))
        if path.parent != Path("."):
            return str(model)
        for d in UltralyticsTrackerAdapter.REID_SEARCH_DIRS:
            cand = Path(d) / path.name
            if cand.exists():
                return str(cand)
        return str(model)  # let Ultralytics fetch it

    def _reid_device(self) -> str:
        """An explicit, indexed device string; 'cuda' alone loses the CUDA EP."""
        if self.device and self.device not in ("auto", "cpu"):
            return f"cuda:{self.device}" if str(self.device).isdigit() else str(self.device)
        if self.device == "cpu":
            return "cpu"
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def finalize(self) -> None:
        self._tracker = None

    # -- association ------------------------------------------------------- #
    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> np.ndarray:
        import torch
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.basetrack import BaseTrack

        det = np.asarray(detections, dtype=np.float32)
        if det.ndim != 2 or det.shape[1] != len(DETECTION_INPUT_COLUMNS):
            raise ValueError(
                f"detections must be (N, {len(DETECTION_INPUT_COLUMNS)}) "
                f"{DETECTION_INPUT_COLUMNS}, got {det.shape}"
            )
        if self.needs_frames and frame is None:
            raise ValueError(
                f"arm {self.name!r} needs the decoded frame (gmc_method="
                f"{getattr(self.args, 'gmc_method', None)!r}, with_reid="
                f"{getattr(self.args, 'with_reid', False)!r}) but frame=None"
            )

        H, W = (frame.shape[:2] if frame is not None else (0, 0))
        boxes = Boxes(torch.as_tensor(det), orig_shape=(H, W))

        # Swap in this arm's private ID counter so concurrent arms in one
        # shared-decode loop cannot interleave their track numbering.
        saved = BaseTrack._count
        BaseTrack._count = self._id_count
        try:
            tracked = self._tracker.update(boxes, frame)
        finally:
            self._id_count = BaseTrack._count
            BaseTrack._count = saved

        tracked = np.asarray(tracked, dtype=np.float32)
        if tracked.ndim != 2 or tracked.shape[0] == 0:
            return empty_track_output()
        return tracked

    # -- metadata ---------------------------------------------------------- #
    def params(self) -> dict[str, Any]:
        out = {k: v for k, v in vars(self.args).items()}
        out["tracker_yaml"] = self.tracker_yaml
        if self._reid_model:
            out["reid_model_resolved"] = self._reid_model
            out["reid_device"] = self._reid_device()
        return out

    def provenance(self) -> dict[str, Any]:
        import ultralytics

        return {
            "name": self.name,
            "adapter": type(self).__name__,
            "implementation": "ultralytics",
            "implementation_version": ultralytics.__version__,
            "tracker_type": getattr(self.args, "tracker_type", None),
            "tracker_class": type(self._tracker).__name__ if self._tracker else None,
            "needs_frames": self.needs_frames,
            "reid_model": self._reid_model,
        }


def build_arm(name: str, tracker_yaml: str | Path, device: str | None = None,
              reid_fallback: str | None = None) -> UltralyticsTrackerAdapter:
    """Construct one benchmark arm."""
    return UltralyticsTrackerAdapter(name, tracker_yaml, device=device,
                                     reid_fallback=reid_fallback)
