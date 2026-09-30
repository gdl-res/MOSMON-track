"""Video inspection and tabular I/O helpers.

``probe_video`` prefers ``ffprobe`` (accurate codec/fps/duration) and falls back
to OpenCV. ``save_table`` writes Parquet when pyarrow is available, otherwise CSV,
so the pipeline degrades gracefully.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import InputConfig
from .metadata import parse_filename


@dataclass
class VideoInfo:
    path: str
    filename: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    codec: str | None = None
    probe_backend: str | None = None
    warnings: str = ""


def find_videos(root: str | Path, cfg: InputConfig) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    exts = {e.lower() for e in cfg.video_extensions}
    pattern = "**/*" if cfg.recursive else "*"
    return sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts
    )


def _ffprobe(path: Path) -> VideoInfo | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        data = json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    s = streams[0]
    warnings = []

    def _rate(val: str | None) -> float | None:
        if not val or val == "0/0":
            return None
        try:
            num, den = val.split("/")
            return float(num) / float(den) if float(den) else None
        except (ValueError, ZeroDivisionError):
            return None

    fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate"))
    duration = None
    for src in (s.get("duration"), (data.get("format") or {}).get("duration")):
        if src:
            try:
                duration = float(src)
                break
            except ValueError:
                pass
    frame_count = None
    if s.get("nb_frames") and s["nb_frames"].isdigit():
        frame_count = int(s["nb_frames"])
    elif fps and duration:
        frame_count = int(round(fps * duration))
        warnings.append("frame_count estimated from fps*duration")
    return VideoInfo(
        path=str(path), filename=path.name,
        width=s.get("width"), height=s.get("height"), fps=fps,
        frame_count=frame_count, duration_s=duration, codec=s.get("codec_name"),
        probe_backend="ffprobe", warnings="|".join(warnings),
    )


def _opencv_probe(path: Path) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(path))
    warnings = []
    if not cap.isOpened():
        return VideoInfo(path=str(path), filename=path.name,
                         probe_backend="opencv", warnings="could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()
    duration = frame_count / fps if (fps and frame_count) else None
    if not fps:
        warnings.append("fps unavailable from OpenCV")
    return VideoInfo(
        path=str(path), filename=path.name, width=width, height=height, fps=fps,
        frame_count=frame_count, duration_s=duration, probe_backend="opencv",
        warnings="|".join(warnings),
    )


def probe_video(path: str | Path) -> VideoInfo:
    """Probe a single video, ffprobe first then OpenCV fallback."""
    path = Path(path)
    info = _ffprobe(path)
    if info is None:
        info = _opencv_probe(path)
    if info.fps is None:
        info.warnings = "|".join(filter(None, [info.warnings, "no fps -> temporal features unreliable"]))
    return info


def build_inventory(root: str | Path, cfg: InputConfig) -> pd.DataFrame:
    """Probe every video under ``root`` and merge in parsed filename metadata."""
    rows = []
    for vp in find_videos(root, cfg):
        info = probe_video(vp)
        meta = parse_filename(vp.name)
        row = asdict(info)
        md = meta.to_dict()
        md.pop("filename", None)
        row.update({f"meta_{k}": v for k, v in md.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def decategorize(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with any categorical column expanded back to plain values.

    Frame-level track tables carry their label columns (video/model/tracker
    names) as one-category categoricals, because materialising a 150-character
    path once per row costs gigabytes on a dense video. That is an in-memory
    and on-disk storage detail; no caller should have to know about it, so
    tables are decoded here on the way back in.
    """
    cats = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    if not cats:
        return df
    return df.assign(**{c: df[c].astype(df[c].cat.categories.dtype) for c in cats})


def save_table(df: pd.DataFrame, path: str | Path, keep_categories: bool = False) -> Path:
    """Save a DataFrame as Parquet if possible, else CSV (path suffix adjusted).

    Categorical columns are written out as plain values unless
    ``keep_categories`` is set. Only the huge frame-level track tables set it:
    for them, expanding the labels for the write would undo the whole point of
    storing them as codes. Everything else keeps a schema that a bare
    ``pd.read_parquet`` reads exactly as it did before.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not keep_categories:
        df = decategorize(df)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return decategorize(pd.read_parquet(path))
    return pd.read_csv(path)


def iter_frames(path: str | Path, stride: int = 1, max_frames: int | None = None) -> Iterator[tuple[int, object]]:
    """Yield (frame_idx, BGR frame) pairs, honouring stride and max_frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    idx = 0
    emitted = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def read_representative_frame(path: str | Path, position: float = 0.5):
    """Read a single frame (RGB) at a fractional position for plot backgrounds."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * position))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame[:, :, ::-1]  # BGR -> RGB
