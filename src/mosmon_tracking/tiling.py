"""Geometry helpers for tiled inference on native-resolution frames.

These are the runtime counterpart of the *training* tiling scripts
(``training/tiling/``): the same sliding-window stride math and coordinate remapping,
but applied to a live video frame so a high-resolution image can be detected tile-by-tile
at 1:1 pixel fidelity and the results stitched back into full-frame coordinates.

Kept dependency-light (numpy only) and free of any Ultralytics/torch import so the unit
tests run without GPU or model weights. ``agnostic_nms`` uses ``torchvision`` when it is
importable and otherwise falls back to a pure-numpy implementation.
"""

from __future__ import annotations

import numpy as np


def tile_windows(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return ``(x0, y0, x1, y1)`` pixel windows covering a ``width`` x ``height`` frame.

    Mirrors the stride + edge-clamp logic used across the training tiling scripts:
    stride ``= int(tile * (1 - overlap))`` with a final window snapped to the right/bottom
    edge so the whole frame is covered even when the dimension is not a stride multiple.
    Windows are de-duplicated and clamped to the frame, so when ``tile`` exceeds a
    dimension a single, smaller edge window is returned.
    """
    if tile <= 0:
        raise ValueError("tile must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(tile * (1 - overlap)))

    def starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        s = list(range(0, extent - tile + 1, stride))
        if not s or s[-1] + tile < extent:
            s.append(extent - tile)
        return sorted(set(s))

    windows: list[tuple[int, int, int, int]] = []
    for y0 in starts(height):
        for x0 in starts(width):
            x1 = min(x0 + tile, width)
            y1 = min(y0 + tile, height)
            windows.append((x0, y0, x1, y1))
    return windows


def remap_boxes(xyxy_local: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Shift tile-local ``[x1, y1, x2, y2]`` boxes into full-frame coordinates."""
    boxes = np.asarray(xyxy_local, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    offset = np.array([x0, y0, x0, y0], dtype=np.float64)
    return boxes + offset


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy class-agnostic NMS. Returns kept indices (into the input arrays)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def agnostic_nms(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Class-agnostic non-max suppression; returns kept indices, highest score first.

    De-duplicates boxes that the same larva produces in the overlap zone of adjacent
    tiles. Class-agnostic on purpose: a larva split across a seam may be detected as
    different classes in each tile, and we want a single track per animal.
    """
    boxes = np.asarray(xyxy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=int)
    try:
        import torch
        from torchvision.ops import nms

        keep = nms(
            torch.as_tensor(boxes, dtype=torch.float32),
            torch.as_tensor(scores, dtype=torch.float32),
            float(iou_thr),
        )
        return keep.cpu().numpy().astype(int)
    except Exception:
        return _nms_numpy(boxes, scores, float(iou_thr))
