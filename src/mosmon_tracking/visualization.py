"""Static plots for QC and behaviour summaries.

All functions write a PNG and return its path. We force the non-interactive
``Agg`` backend so plots render on headless machines / inside batch jobs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .heatmaps import Heatmap  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trajectories(clean: pd.DataFrame, out_path: str | Path,
                      background: np.ndarray | None = None, max_tracks: int = 200) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    if background is not None:
        ax.imshow(background)
    else:
        if {"frame_width", "frame_height"}.issubset(clean.columns):
            ax.set_xlim(0, clean["frame_width"].iloc[0])
            ax.set_ylim(clean["frame_height"].iloc[0], 0)
    track_ids = clean["track_id"].dropna().unique()[:max_tracks]
    cmap = plt.get_cmap("tab20")
    for i, tid in enumerate(track_ids):
        g = clean[clean["track_id"] == tid].sort_values("frame_idx")
        ax.plot(g["cx"], g["cy"], "-", lw=0.8, color=cmap(i % 20), alpha=0.8)
    ax.set_title(f"Trajectories (n={len(track_ids)})")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return _save(fig, out_path)


def plot_heatmap(hm: Heatmap, out_path: str | Path,
                 background: np.ndarray | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    extent = None
    if background is not None:
        h, w = background.shape[:2]
        ax.imshow(background)
        extent = (0, w, h, 0)
    im = ax.imshow(
        hm.array, origin="upper", aspect="auto", cmap="inferno",
        alpha=0.7 if background is not None else 1.0, extent=extent,
    )
    fig.colorbar(im, ax=ax, label=hm.kind)
    ax.set_title(f"{hm.name} heatmap ({hm.normalize})")
    return _save(fig, out_path)


def plot_speed_over_time(clean: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    if "time_s" in clean.columns:
        g = clean.dropna(subset=["speed_px_s"]).groupby("frame_idx").agg(
            t=("time_s", "first"), mean_speed=("speed_px_s", "mean")
        )
        ax.plot(g["t"], g["mean_speed"], lw=1.0)
        ax.set_xlabel("time (s)")
    ax.set_ylabel("mean speed (px/s)")
    ax.set_title("Population mean speed over time")
    return _save(fig, out_path)


def plot_active_over_time(clean: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    g = clean.groupby("frame_idx").agg(
        t=("time_s", "first") if "time_s" in clean.columns else ("frame_idx", "first"),
        n=("track_id", "nunique"),
    )
    ax.plot(g["t"], g["n"], lw=1.0)
    ax.set_xlabel("time (s)" if "time_s" in clean.columns else "frame")
    ax.set_ylabel("active larvae")
    ax.set_title("Active larvae over time")
    return _save(fig, out_path)


def plot_track_duration_hist(track_summary: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    col = "duration_s" if track_summary["duration_s"].notna().any() else "n_frames"
    ax.hist(track_summary[col].dropna(), bins=30, color="steelblue")
    ax.set_xlabel(col)
    ax.set_ylabel("tracks")
    ax.set_title("Track duration distribution")
    return _save(fig, out_path)


def plot_speed_distribution(clean: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sp = clean["speed_px_s"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(sp, bins=50, color="darkorange")
    ax.set_xlabel("speed (px/s)")
    ax.set_ylabel("count")
    ax.set_title("Speed distribution")
    return _save(fig, out_path)


def plot_turn_angle_distribution(clean: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": "polar"})
    turn = clean["turn_angle_rad"].dropna()
    ax.hist(turn, bins=36, color="purple")
    ax.set_title("Turn-angle distribution")
    return _save(fig, out_path)


def plot_nn_distance_distribution(clean: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    nn = clean.get("nearest_neighbor_distance_px")
    if nn is not None:
        ax.hist(nn.dropna(), bins=40, color="seagreen")
    ax.set_xlabel("nearest-neighbour distance (px)")
    ax.set_ylabel("count")
    ax.set_title("Nearest-neighbour distance distribution")
    return _save(fig, out_path)


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour per track id."""
    rng = (int(track_id) * 2654435761) & 0xFFFFFFFF
    return (rng & 255, (rng >> 8) & 255, (rng >> 16) & 255)


def render_debug_video(
    video_path: str | Path,
    tracks: pd.DataFrame,
    out_path: str | Path,
    fps: int = 20,
    scale: float = 0.35,
    max_frames: int | None = 300,
    trail: int = 20,
) -> Path | None:
    """Write an annotated MP4 (boxes + track IDs + fading trails) for validation.

    Only frames present in ``tracks`` are written, so it aligns with the frames
    that were actually processed. Output is downscaled by ``scale`` to keep size
    manageable for high-resolution sources.
    """
    from collections import defaultdict, deque

    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tracks.empty or not {"x1", "y1", "x2", "y2", "track_id", "frame_idx"}.issubset(tracks.columns):
        return None

    frames_wanted = sorted(tracks["frame_idx"].unique())
    if max_frames:
        frames_wanted = frames_wanted[:max_frames]
    wanted = set(frames_wanted)
    by_frame = {f: g for f, g in tracks[tracks["frame_idx"].isin(wanted)].groupby("frame_idx")}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = max(1, int(W * scale)), max(1, int(H * scale))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail))

    idx = 0
    last_wanted = max(wanted)
    try:
        while idx <= last_wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                g = by_frame.get(idx)
                if g is not None:
                    for r in g.itertuples(index=False):
                        tid = int(r.track_id)
                        color = _id_color(tid)
                        p1 = (int(r.x1), int(r.y1))
                        p2 = (int(r.x2), int(r.y2))
                        cv2.rectangle(frame, p1, p2, color, 2)
                        cv2.putText(frame, str(tid), (p1[0], max(0, p1[1] - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        trails[tid].append((int(r.cx), int(r.cy)))
                    for tid, pts in trails.items():
                        for a, b in zip(list(pts)[:-1], list(pts)[1:]):
                            cv2.line(frame, a, b, _id_color(tid), 1)
                small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(small)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path


def plot_group_bar(df: pd.DataFrame, group_col: str, value_col: str,
                   out_path: str | Path, title: str | None = None) -> Path:
    """Bar chart of mean ``value_col`` per ``group_col`` (for cross-video roll-ups)."""
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * df[group_col].nunique() + 3), 4))
    g = (df.dropna(subset=[value_col])
           .groupby(group_col)[value_col].mean().sort_values(ascending=False))
    ax.bar([str(i) for i in g.index], g.to_numpy(), color="steelblue")
    ax.set_xlabel(group_col)
    ax.set_ylabel(f"mean {value_col}")
    ax.set_title(title or f"{value_col} by {group_col}")
    fig.autofmt_xdate(rotation=45)
    return _save(fig, out_path)


def render_heatmap_accumulation_video(
    clean: pd.DataFrame,
    out_path: str | Path,
    background: np.ndarray | None = None,
    bins_x: int = 128,
    bins_y: int = 72,
    fps: int = 20,
    scale: float = 0.5,
    max_frames: int | None = 300,
    alpha: float = 0.6,
) -> Path | None:
    """Write an MP4 of the occupancy heatmap *accumulating* over time.

    For each processed frame the larvae centres are binned into a running 2-D
    histogram; each output frame shows that histogram so far (``log1p`` +
    running-max normalised so the spatial pattern stays visible while it builds),
    coloured with the ``inferno`` map and blended over ``background`` only where
    occupancy is non-zero so empty areas keep the real frame.

    ``background`` is expected RGB (as from :func:`read_representative_frame`);
    when ``None`` the heatmap is drawn on black. Output frames are capped to
    ``max_frames`` by striding (the histogram still accumulates every frame).
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if clean is None or clean.empty or not {"cx_norm", "cy_norm", "frame_idx"}.issubset(clean.columns):
        return None

    if background is not None:
        bg = background[:, :, ::-1].copy()  # RGB -> BGR for OpenCV
        H, W = bg.shape[:2]
        out_w, out_h = max(1, int(W * scale)), max(1, int(H * scale))
        bg = cv2.resize(bg, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        out_w, out_h = 1280, 720
        bg = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    frames = sorted(clean["frame_idx"].dropna().unique())
    if not frames:
        return None
    stride = 1
    if max_frames and len(frames) > max_frames:
        stride = int(np.ceil(len(frames) / max_frames))
    by_frame = {f: g for f, g in clean.groupby("frame_idx")}

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    acc = np.zeros((bins_y, bins_x), dtype=np.float64)
    try:
        for i, f in enumerate(frames):
            g = by_frame.get(f)
            if g is not None and len(g):
                hist, _, _ = np.histogram2d(
                    g["cy_norm"], g["cx_norm"], bins=[bins_y, bins_x],
                    range=[[0, 1], [0, 1]],
                )
                acc += hist
            if i % stride != 0 and i != len(frames) - 1:
                continue
            disp = np.log1p(acc)
            m = disp.max()
            disp = disp / m if m > 0 else disp
            big = cv2.resize(disp.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            heat_u8 = np.clip(big * 255.0, 0, 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
            a = (big[..., None] * alpha)  # per-pixel blend weight, 0 where unvisited
            blended = (bg * (1.0 - a) + heat_color * a).astype(np.uint8)
            cv2.putText(blended, f"frame {int(f)}", (8, out_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(blended)
    finally:
        writer.release()
    return out_path


def plot_qc_dashboard(qc: dict, out_path: str | Path) -> Path:
    """Compact multi-panel QC summary from the qc dict."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    dpf = qc.get("detections_per_frame", {})
    if isinstance(dpf, dict) and dpf.get("histogram"):
        axes[0].bar(range(len(dpf["histogram"])), dpf["histogram"], color="steelblue")
    axes[0].set_title("Detections per frame")
    tl = qc.get("track_length_hist")
    if tl:
        axes[1].bar(range(len(tl)), tl, color="seagreen")
    axes[1].set_title("Track length histogram")
    metrics = {k: v for k, v in qc.items() if isinstance(v, (int, float))}
    keys = list(metrics)[:8]
    axes[2].axis("off")
    txt = "\n".join(f"{k}: {metrics[k]:.3g}" for k in keys)
    axes[2].text(0.0, 1.0, txt, va="top", fontsize=9, family="monospace")
    axes[2].set_title("QC metrics")
    warnings = qc.get("warnings", [])
    axes[3].axis("off")
    axes[3].text(0.0, 1.0, "\n".join(warnings[:10]) or "no warnings", va="top",
                 fontsize=9, color="firebrick")
    axes[3].set_title("Warnings")
    fig.suptitle("QC dashboard")
    return _save(fig, out_path)
