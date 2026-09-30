"""YOLO11 + Ultralytics tracking wrapper.

This produces the *raw* per-frame detection table (see ``RAW_TRACK_COLUMNS``).
The detector is isolated behind :func:`run_full_frame_tracking` so that a future
tiled-inference backend can be added without changing any downstream code: both
backends just need to return a DataFrame with the raw schema.

``ultralytics`` is imported lazily inside the functions so the rest of the
package (and the unit tests) import cleanly without GPU/torch present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .tiling import agnostic_nms, remap_boxes, tile_windows
from .video_io import probe_video

RAW_TRACK_COLUMNS = [
    "video_path", "video_name", "model_path", "model_name", "tracker",
    "frame_idx", "time_s", "track_id", "class_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "w", "h", "cx", "cy", "cx_norm", "cy_norm",
    "frame_width", "frame_height",
]


DETECTION_COLUMNS = [
    "video_id", "video_path", "video_name", "model_name", "model_path",
    "source_frame", "processed_frame", "time_s",
    "x1", "y1", "x2", "y2", "confidence",
    "predicted_class", "class_name", "frame_width", "frame_height",
]


def empty_detection_table() -> pd.DataFrame:
    return pd.DataFrame(columns=DETECTION_COLUMNS)


def _resolve_device(device: str) -> object:
    if device and device != "auto":
        return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def empty_raw_table() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_TRACK_COLUMNS)


def run_full_frame_tracking(
    video_path: str | Path,
    model_path: str | Path,
    cfg: Config,
    tracker_yaml: str | Path | None = None,
) -> pd.DataFrame:
    """Run full-frame Ultralytics tracking; return the raw per-frame table.

    This is the only function that talks to Ultralytics. A tiled backend would
    mirror its signature and output schema.
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    model_path = Path(model_path)
    info = probe_video(video_path)
    fps = info.fps or 0.0
    if not fps:
        # Without fps we cannot derive time_s reliably; caller is warned via QC.
        fps = 0.0

    device = _resolve_device(cfg.model.device)
    use_half = bool(cfg.model.half) and device not in ("cpu", None)
    tracker_yaml = str(tracker_yaml or cfg.tracker.yaml)
    stride = max(1, cfg.video.frame_stride)

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}

    results = model.track(
        source=str(video_path),
        tracker=tracker_yaml,
        persist=True,
        conf=cfg.model.conf,
        iou=cfg.model.iou,
        imgsz=cfg.model.imgsz,
        device=device,
        half=use_half,
        classes=cfg.model.classes,
        agnostic_nms=cfg.model.agnostic_nms,
        max_det=cfg.model.max_det,
        vid_stride=stride,
        stream=True,
        verbose=False,
    )

    rows: list[dict] = []
    processed = 0
    for result in results:
        frame_idx = processed * stride  # original-video frame index
        time_s = frame_idx / fps if fps else np.nan
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            H, W = result.orig_shape if getattr(result, "orig_shape", None) else (info.height, info.width)
            for (x1, y1, x2, y2), tid, cls, conf in zip(xyxy, ids, clss, confs):
                w = float(x2 - x1)
                h = float(y2 - y1)
                cx = float(x1 + w / 2.0)
                cy = float(y1 + h / 2.0)
                rows.append({
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "model_path": str(model_path),
                    "model_name": model_path.stem,
                    "tracker": cfg.tracker.type,
                    "frame_idx": frame_idx,
                    "time_s": time_s,
                    "track_id": int(tid),
                    "class_id": int(cls),
                    "class_name": names.get(int(cls), str(cls)),
                    "confidence": float(conf),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "w": w, "h": h, "cx": cx, "cy": cy,
                    "cx_norm": cx / W if W else np.nan,
                    "cy_norm": cy / H if H else np.nan,
                    "frame_width": int(W) if W else None,
                    "frame_height": int(H) if H else None,
                })
        processed += 1
        if cfg.video.max_frames and processed >= cfg.video.max_frames:
            break

    if not rows:
        return empty_raw_table()
    return pd.DataFrame(rows, columns=RAW_TRACK_COLUMNS)


def run_detection_only(
    video_path: str | Path,
    model_path: str | Path,
    cfg: Config,
) -> pd.DataFrame:
    """Detector-only pass: no tracker, no association, every surviving box kept.

    This is the *fixed detector output* the multi-tracker benchmark replays. It uses
    exactly the preprocessing of :func:`run_full_frame_tracking` (same ``imgsz``,
    ``iou``, ``agnostic_nms``, ``half``, ``vid_stride``) so that filtering this table
    at a higher confidence reproduces a run made at that confidence directly: NMS
    ranks by score descending, so a low-scoring box can never suppress a higher one.

    That equivalence holds only while ``max_det`` does not bind, which is why the
    benchmark config raises it well above the ~350 larvae/frame the corpus reaches.
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    model_path = Path(model_path)
    info = probe_video(video_path)
    fps = info.fps or 0.0

    device = _resolve_device(cfg.model.device)
    use_half = bool(cfg.model.half) and device not in ("cpu", None)
    stride = max(1, cfg.video.frame_stride)

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}

    results = model.predict(
        source=str(video_path),
        conf=cfg.model.conf,
        iou=cfg.model.iou,
        imgsz=cfg.model.imgsz,
        device=device,
        half=use_half,
        classes=cfg.model.classes,
        agnostic_nms=cfg.model.agnostic_nms,
        max_det=cfg.model.max_det,
        vid_stride=stride,
        stream=True,
        verbose=False,
    )

    frames: list[pd.DataFrame] = []
    processed = 0
    for result in results:
        source_frame = processed * stride  # original-video frame index
        time_s = source_frame / fps if fps else np.nan
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            H, W = (result.orig_shape if getattr(result, "orig_shape", None)
                    else (info.height, info.width))
            frames.append(pd.DataFrame({
                "video_id": video_path.stem,
                "video_path": str(video_path),
                "video_name": video_path.name,
                "model_name": model_path.stem,
                "model_path": str(model_path),
                "source_frame": source_frame,
                "processed_frame": processed,
                "time_s": time_s,
                "x1": xyxy[:, 0], "y1": xyxy[:, 1],
                "x2": xyxy[:, 2], "y2": xyxy[:, 3],
                "confidence": confs,
                "predicted_class": clss,
                "class_name": [names.get(int(c), str(int(c))) for c in clss],
                "frame_width": int(W) if W else None,
                "frame_height": int(H) if H else None,
            }))
        processed += 1
        if cfg.video.max_frames and processed >= cfg.video.max_frames:
            break

    if not frames:
        return empty_detection_table()
    return pd.concat(frames, ignore_index=True)[DETECTION_COLUMNS]


#: In-tree Ultralytics trackers we can drive from detections we computed ourselves.
#: All four take ``args`` only and share ``update(results, img=None, feats=None)``.
SUPPORTED_TRACKERS = ("bytetrack", "botsort", "ocsort", "deepocsort")


def load_tracker_args(tracker_yaml: str | Path):
    """Load a tracker YAML into the ``IterableSimpleNamespace`` Ultralytics expects."""
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    return IterableSimpleNamespace(**YAML.load(check_yaml(str(tracker_yaml))))


def tracker_needs_frames(targs) -> bool:
    """Whether this tracker config consumes the decoded frame.

    Global motion compensation reads the image directly, and an explicit ReID
    encoder crops from it. Frame-free trackers can be replayed from a detection
    table alone, which is far cheaper because no video decode is needed.
    """
    gmc = str(getattr(targs, "gmc_method", "none") or "none").lower()
    return gmc not in ("none", "") or bool(getattr(targs, "with_reid", False))


def _build_tracker(tracker_yaml: str | Path, fps: float | None = None):
    """Instantiate a stand-alone Ultralytics tracker from a YAML.

    Mirrors ``ultralytics.trackers.track.on_predict_start`` but without a predictor, so we
    can feed it detections we computed ourselves (tile-by-tile, or replayed from a frozen
    detection table) instead of full-frame ones.

    ``fps`` is accepted and ignored: Ultralytics >= 8.4 dropped the ``frame_rate``
    argument and ``track_buffer`` is now a plain frame count that no longer scales
    with frame rate. Passing it raised ``TypeError`` on 8.4.82.
    """
    tcfg = load_tracker_args(tracker_yaml)
    return build_tracker_from_args(tcfg)


def build_tracker_from_args(targs):
    """Instantiate an in-tree Ultralytics tracker from an already-loaded args namespace."""
    from ultralytics.trackers.track import TRACKER_MAP

    ttype = getattr(targs, "tracker_type", None)
    if ttype not in SUPPORTED_TRACKERS:
        raise RuntimeError(
            f"Supported trackers are {list(SUPPORTED_TRACKERS)}, got {ttype!r}"
        )
    if ttype not in TRACKER_MAP:
        raise RuntimeError(
            f"ultralytics does not provide tracker {ttype!r}; available: {sorted(TRACKER_MAP)}"
        )
    return TRACKER_MAP[ttype](args=targs)


def run_tiled_tracking(
    video_path: str | Path,
    model_path: str | Path,
    cfg: Config,
    tracker_yaml: str | Path | None = None,
) -> pd.DataFrame:
    """Tiled-inference backend: detect each frame tile-by-tile, then track.

    High-resolution frames are sliced into overlapping ``tile_size`` windows, each detected
    at 1:1 pixel fidelity, remapped to full-frame coordinates, de-duplicated across seams
    with class-agnostic NMS, and handed to a stand-alone Ultralytics tracker. Returns the
    same raw per-frame table as :func:`run_full_frame_tracking` (identical schema), so all
    downstream code is unaffected.
    """
    try:
        import cv2
        import torch  # noqa: F401  (required by the detector + Boxes wrapper)
        from ultralytics import YOLO
        from ultralytics.engine.results import Boxes
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Tiled inference requires ultralytics + torch + opencv. "
            "Disable model.tiling.enabled to use the full-frame backend."
        ) from exc

    video_path = Path(video_path)
    model_path = Path(model_path)
    info = probe_video(video_path)
    fps = info.fps or 0.0

    tcfg = cfg.model.tiling
    tile = int(tcfg.tile_size)
    merge_iou = tcfg.merge_iou if tcfg.merge_iou is not None else cfg.model.iou
    min_area = float(tcfg.min_area_frac) * (tile * tile)
    batch = max(1, int(tcfg.batch))
    stride = max(1, cfg.video.frame_stride)

    device = _resolve_device(cfg.model.device)
    use_half = bool(cfg.model.half) and device not in ("cpu", None)
    tracker_yaml = str(tracker_yaml or cfg.tracker.yaml)

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}
    tracker = _build_tracker(tracker_yaml, fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for tiled inference: {video_path}")

    rows: list[dict] = []
    read_idx = 0
    processed = 0
    try:
        while True:
            if cfg.video.max_frames and processed >= cfg.video.max_frames:
                break
            if read_idx % stride != 0:
                if not cap.grab():
                    break
                read_idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = read_idx
            read_idx += 1
            processed += 1

            H, W = frame.shape[:2]
            windows = tile_windows(W, H, tile, tcfg.overlap)
            crops = [frame[y0:y1, x0:x1] for (x0, y0, x1, y1) in windows]

            all_boxes: list[np.ndarray] = []
            all_conf: list[np.ndarray] = []
            all_cls: list[np.ndarray] = []
            for start in range(0, len(crops), batch):
                chunk = crops[start:start + batch]
                wins = windows[start:start + batch]
                preds = model.predict(
                    chunk,
                    imgsz=cfg.model.imgsz,
                    conf=cfg.model.conf,
                    iou=cfg.model.iou,
                    classes=cfg.model.classes,
                    max_det=cfg.model.max_det,
                    device=device,
                    half=use_half,
                    verbose=False,
                )
                for pred, (x0, y0, _x1, _y1) in zip(preds, wins):
                    b = getattr(pred, "boxes", None)
                    if b is None or len(b) == 0:
                        continue
                    xyxy = b.xyxy.cpu().numpy()
                    all_boxes.append(remap_boxes(xyxy, x0, y0))
                    all_conf.append(b.conf.cpu().numpy())
                    all_cls.append(b.cls.cpu().numpy())

            time_s = frame_idx / fps if fps else np.nan
            if not all_boxes:
                tracker.update(Boxes(torch.zeros((0, 6)), orig_shape=(H, W)), frame)
                continue

            boxes = np.concatenate(all_boxes, axis=0)
            conf = np.concatenate(all_conf, axis=0)
            cls = np.concatenate(all_cls, axis=0)
            # Clamp to frame bounds: per-tile predictions can round a hair past the
            # crop edge, which after remapping would sit just outside the full frame.
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)

            if min_area > 0:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                keep_area = areas >= min_area
                boxes, conf, cls = boxes[keep_area], conf[keep_area], cls[keep_area]

            if len(boxes):
                keep = agnostic_nms(boxes, conf, merge_iou)
                boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

            det = np.concatenate(
                [boxes, conf[:, None], cls[:, None]], axis=1
            ).astype(np.float32)
            tracked = np.asarray(tracker.update(Boxes(torch.as_tensor(det), orig_shape=(H, W)), frame))

            if tracked.ndim != 2 or tracked.shape[0] == 0:
                continue
            for x1, y1, x2, y2, tid, tconf, tcls, _idx in tracked:
                w = float(x2 - x1)
                h = float(y2 - y1)
                cx = float(x1 + w / 2.0)
                cy = float(y1 + h / 2.0)
                cls_i = int(tcls)
                rows.append({
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "model_path": str(model_path),
                    "model_name": model_path.stem,
                    "tracker": cfg.tracker.type,
                    "frame_idx": frame_idx,
                    "time_s": time_s,
                    "track_id": int(tid),
                    "class_id": cls_i,
                    "class_name": names.get(cls_i, str(cls_i)),
                    "confidence": float(tconf),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "w": w, "h": h, "cx": cx, "cy": cy,
                    "cx_norm": cx / W if W else np.nan,
                    "cy_norm": cy / H if H else np.nan,
                    "frame_width": int(W), "frame_height": int(H),
                })
    finally:
        cap.release()

    if not rows:
        return empty_raw_table()
    return pd.DataFrame(rows, columns=RAW_TRACK_COLUMNS)


def track_video(video_path, model_path, cfg: Config, tracker_yaml=None) -> pd.DataFrame:
    """Public entry point. Selects the tracking backend from ``cfg.model.tiling``."""
    tiling_cfg = getattr(cfg.model, "tiling", None)
    if tiling_cfg is not None and tiling_cfg.enabled:
        return run_tiled_tracking(video_path, model_path, cfg, tracker_yaml)
    return run_full_frame_tracking(video_path, model_path, cfg, tracker_yaml)
