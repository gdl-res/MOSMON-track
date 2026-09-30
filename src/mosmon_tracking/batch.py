"""Pipeline orchestration: single run, batch, analyse-from-raw, model comparison.

The core is :func:`pipeline_from_raw`, which takes a raw per-frame table and
produces the full set of cleaned tracks, features, summaries, heatmaps, plots,
and reports. Both ``track-video`` (after running YOLO) and ``analyse-tracks``
(re-using an existing raw table) funnel through it, so behaviour computation is
identical and YOLO never has to be re-run for feature changes.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import fair, mot
from . import features as feat
from . import heatmaps as hm
from . import reports as rep
from . import trajectories as traj
from . import visualization as viz
from .calibration import Calibrator
from .config import Config
from .metadata import parse_filename
from .quality_control import compute_qc
from .track_postprocess import add_compartment_species, clean_tracks
from .video_io import probe_video, read_representative_frame, save_table


def pipeline_from_raw(
    raw: pd.DataFrame,
    cfg: Config,
    out_dir: str | Path,
    video_path: str | Path | None = None,
    calibrator: Calibrator | None = None,
    n_video_frames: int | None = None,
) -> dict:
    """Run cleaning -> features -> summaries -> heatmaps -> plots -> report."""
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "heatmaps").mkdir(parents=True, exist_ok=True)
    calibrator = calibrator or Calibrator.from_config(cfg.calibration)

    # NOTE: this used to also write a byte-identical ``detections.parquet``.
    # It was not the detector's output -- ``run_full_frame_tracking`` keeps only
    # boxes that received a track id -- so the name promised raw detections while
    # the file held BoT-SORT's output, with nothing below the tracker's own
    # ``track_high_thresh``. Anything genuinely needing detector output should
    # use the detection cache built by ``mosmon detect-videos``.
    save_table(raw, out / "tracks_raw.parquet", keep_categories=True)

    clean, dropped = clean_tracks(raw, cfg)
    if not clean.empty:
        clean = feat.add_cross_track_features(clean, cfg)
        clean = feat.refine_movement_state(clean, cfg)
        clean = add_compartment_species(clean, cfg)
    save_table(clean, out / "tracks_clean.parquet", keep_categories=True)

    track_summary = feat.compute_track_summary(clean, cfg, calibrator)
    save_table(track_summary, out / "track_summary.parquet")

    video_summaries = feat.compute_video_summary(clean, track_summary, cfg)
    video_summary = video_summaries[0] if video_summaries else {}

    # Per-frame population time-series (data behind the activity/speed-over-time plots).
    if cfg.features.export_population_timeseries and not clean.empty:
        pop_ts = feat.compute_population_timeseries(clean, cfg)
        save_table(pop_ts, out / "population_timeseries.parquet")

    qc = compute_qc(raw, clean, cfg, n_video_frames=n_video_frames, dropped=dropped)

    # Heatmaps. The arrays and the entropy are *data* and are always written;
    # only the rendered PNGs answer to `reports.save_plots`. Rendering them is
    # the single most expensive step in this function -- ~39 s per run for 16
    # images over a decoded 5.3K background frame, against 0.03 s to compute the
    # heatmaps themselves -- so a batch that does not want pictures should not
    # pay for them.
    heatmap_paths: list[str] = []
    if not clean.empty:
        heats = hm.compute_all_heatmaps(clean, cfg)
        background = (read_representative_frame(video_path)
                      if video_path and cfg.reports.save_plots else None)
        for h in heats:
            npy = out / "heatmaps" / f"{h.name}.npy"
            np.save(npy, h.array)
            save_table(h.as_dataframe(), out / "heatmaps" / f"{h.name}.parquet")
            if cfg.reports.save_plots:
                png = viz.plot_heatmap(h, out / "heatmaps" / f"{h.name}.png",
                                       background=background)
                heatmap_paths.append(str(png.relative_to(out)))
        video_summary["heatmap_entropy_occupancy"] = next(
            (hm.heatmap_entropy(h) for h in heats if h.name == "occupancy"), np.nan
        )

    # Plots
    plot_paths: list[str] = []
    if not clean.empty and cfg.reports.save_plots:
        bg = read_representative_frame(video_path) if video_path else None
        plotters = [
            ("trajectories.png", lambda p: viz.plot_trajectories(clean, p, background=bg)),
            ("speed_over_time.png", lambda p: viz.plot_speed_over_time(clean, p)),
            ("active_over_time.png", lambda p: viz.plot_active_over_time(clean, p)),
            ("speed_distribution.png", lambda p: viz.plot_speed_distribution(clean, p)),
            ("turn_angle_distribution.png", lambda p: viz.plot_turn_angle_distribution(clean, p)),
            ("nn_distance_distribution.png", lambda p: viz.plot_nn_distance_distribution(clean, p)),
            ("qc_dashboard.png", lambda p: viz.plot_qc_dashboard(qc, p)),
        ]
        if not track_summary.empty:
            plotters.insert(3, ("track_duration_hist.png",
                                lambda p: viz.plot_track_duration_hist(track_summary, p)))
        for fname, fn in plotters:
            try:
                path = fn(out / "plots" / fname)
                plot_paths.append(str(Path(path).relative_to(out)))
            except Exception as exc:  # a single bad plot must not kill the run
                qc.setdefault("warnings", []).append(f"plot {fname} failed: {exc}")

    # Optional annotated debug video for visual validation.
    if cfg.video.save_debug_video and video_path and not clean.empty:
        try:
            viz.render_debug_video(
                video_path, clean, out / "debug_video.mp4",
                fps=cfg.video.debug_video_fps, scale=cfg.video.debug_video_scale,
                max_frames=cfg.video.debug_video_max_frames, trail=cfg.video.debug_video_trail,
            )
        except Exception as exc:
            qc.setdefault("warnings", []).append(f"debug video failed: {exc}")

    # Optional full-resolution, full-length annotated tracking video (every frame, all tracks).
    if cfg.video.save_tracking_video and video_path and not clean.empty:
        try:
            tv_fps = cfg.video.tracking_video_fps
            if tv_fps is None:
                tv_fps = int(round(probe_video(video_path).fps or cfg.video.debug_video_fps))
            viz.render_debug_video(
                video_path, clean, out / "tracking_video.mp4",
                fps=tv_fps, scale=cfg.video.tracking_video_scale,
                max_frames=None, trail=cfg.video.tracking_video_trail,
            )
        except Exception as exc:
            qc.setdefault("warnings", []).append(f"tracking video failed: {exc}")

    # Optional heatmap-accumulation video (occupancy building up over time on a frame).
    if cfg.video.save_heatmap_video and not clean.empty:
        try:
            bg = read_representative_frame(video_path) if video_path else None
            viz.render_heatmap_accumulation_video(
                clean, out / "heatmap_video.mp4", background=bg,
                bins_x=cfg.heatmaps.bins_x, bins_y=cfg.heatmaps.bins_y,
                fps=cfg.video.heatmap_video_fps, scale=cfg.video.heatmap_video_scale,
                max_frames=cfg.video.heatmap_video_max_frames, alpha=cfg.video.heatmap_video_alpha,
            )
        except Exception as exc:
            qc.setdefault("warnings", []).append(f"heatmap video failed: {exc}")

    # Metadata + settings
    video_info = {"filename": Path(video_path).name if video_path else ""}
    if video_path:
        info = probe_video(video_path)
        video_info.update({
            "width": info.width, "height": info.height, "fps": info.fps,
            "frame_count": info.frame_count, "duration_s": info.duration_s,
            "codec": info.codec,
        })
        md = parse_filename(Path(video_path).name).to_dict()
        video_info.update({f"meta_{k}": v for k, v in md.items() if v})
    model_settings = {
        "imgsz": cfg.model.imgsz, "conf": cfg.model.conf, "iou": cfg.model.iou,
        "tracker": cfg.tracker.type, "frame_stride": cfg.video.frame_stride,
        "calibration_mode": calibrator.mode, "units": calibrator.units,
    }

    # MOTChallenge export for interoperable evaluation (TrackEval / py-motmetrics).
    if cfg.reports.export_mot:
        try:
            seqinfo = {
                "name": Path(video_path).stem if video_path else out.name,
                "frame_rate": video_info.get("fps"),
                "seq_length": n_video_frames or video_info.get("frame_count"),
                "im_width": video_info.get("width"),
                "im_height": video_info.get("height"),
            }
            mot.write_mot(
                clean, out / "tracks_mot.txt",
                include_interpolated=cfg.reports.mot_include_interpolated,
                seqinfo=seqinfo,
            )
        except Exception as exc:
            qc.setdefault("warnings", []).append(f"MOT export failed: {exc}")

    # Per-track trajectories as GeoJSON (one feature per track) for GIS/plotting tools.
    if cfg.reports.export_trajectories_geojson and not clean.empty:
        try:
            fc = traj.tracks_to_geojson(clean, track_summary, calibrator)
            traj.write_geojson(fc, out / "trajectories.geojson")
        except Exception as exc:
            qc.setdefault("warnings", []).append(f"trajectories GeoJSON export failed: {exc}")

    full_summary = {
        "video_info": video_info,
        "model_settings": model_settings,
        "qc": qc,
        "behaviour": video_summary,
    }
    rep.write_video_summary_json(full_summary, out / "video_summary.json")

    if cfg.reports.make_markdown:
        md_text = rep.build_markdown(video_info, model_settings, qc, video_summary,
                                     plot_paths, heatmap_paths)
        rep.write_markdown(md_text, out / "report.md")
        if cfg.reports.make_html:
            html_text = rep.build_html(video_info, model_settings, qc, video_summary,
                                       plot_paths, heatmap_paths)
            rep.write_html(html_text, out / "report.html")
            if cfg.reports.make_pdf:
                try:
                    rep.export_report_pdf(out / "report.html", out / "report.pdf")
                except Exception as exc:  # optional deps / render issues must not fail the run
                    qc.setdefault("warnings", []).append(f"PDF report skipped: {exc}")

    # Written last so any PDF-export warning above is captured on disk.
    with (out / "qc_report.json").open("w", encoding="utf-8") as fh:
        json.dump(qc, fh, indent=2, default=rep._json_default)

    # FAIR sidecars: provenance, data dictionary, checksums, RO-Crate (DOI-ready).
    if cfg.fair.enabled:
        _finalize_fair_for_run(out, cfg, raw, clean, video_path, video_info)

    return full_summary


def _fair_input(role: str, path, cfg: Config) -> dict:
    p = Path(path)
    rec: dict = {"role": role, "path": str(p), "filename": p.name}
    if p.exists():
        rec["contentSize"] = p.stat().st_size
        rec["sha256"] = fair.sha256_file(p) if cfg.fair.hash_inputs else None
    else:
        rec["note"] = "input not available at analysis time (hash unknown)"
    return rec


def _finalize_fair_for_run(out: Path, cfg: Config, raw: pd.DataFrame,
                           clean: pd.DataFrame, video_path, video_info: dict) -> None:
    inputs = []
    if video_path:
        inputs.append(_fair_input("source-video", video_path, cfg))
    if "model_path" in raw.columns and len(raw):
        inputs.append(_fair_input("model-weights", raw["model_path"].iloc[0], cfg))
    # Species: union of detected class names and parsed filename metadata.
    species: set[str] = set()
    if not clean.empty and "class_name" in clean.columns:
        species.update(map(str, clean["class_name"].dropna().unique()))
    if video_path:
        species.update(parse_filename(Path(video_path).name).species)
    name = video_info.get("filename") or out.name
    fair.finalize_fair(
        out, cfg,
        dataset_name=f"MOSMON-Larvae tracking & behaviour results: {name}",
        dataset_description=(
            "Per-frame detections, cleaned tracks, behaviour features, heatmaps, "
            "QC, and reports produced by the mosmon-tracking pipeline from a single "
            "mosquito-larvae video."
        ),
        inputs=inputs,
        species_present=sorted(species),
    )


def run_single_video(
    video_path: str | Path,
    model_path: str | Path,
    cfg: Config,
    out_dir: str | Path,
    calibrator: Calibrator | None = None,
) -> dict:
    """Track one video with one model, then run the full pipeline."""
    from .yolo_tracker import track_video  # lazy: avoids importing ultralytics in tests

    info = probe_video(video_path)
    raw = track_video(video_path, model_path, cfg)
    # QC's "% frames with detections" must be relative to the frames we actually
    # processed, not the full video length (frame_stride / max_frames reduce it).
    n_processed = info.frame_count
    if n_processed:
        stride = max(1, cfg.video.frame_stride)
        n_processed = (n_processed + stride - 1) // stride
        if cfg.video.max_frames:
            n_processed = min(n_processed, cfg.video.max_frames)
    return pipeline_from_raw(
        raw, cfg, out_dir, video_path=video_path, calibrator=calibrator,
        n_video_frames=n_processed,
    )


def analyse_raw(
    raw_path: str | Path, cfg: Config, out_dir: str | Path,
    calibrator: Calibrator | None = None,
) -> dict:
    """Recompute features/summaries/reports from an existing raw table (no YOLO)."""
    from .video_io import load_table

    raw_path = Path(raw_path)
    if raw_path.is_dir():
        raw_path = next(
            (p for name in ("tracks_raw.parquet", "tracks_raw.csv",
                            "detections.parquet", "detections.csv")
             for p in [raw_path / name] if p.exists()),
            None,
        )
        if raw_path is None:
            raise FileNotFoundError("no tracks_raw/detections table found in directory")
    raw = load_table(raw_path)
    vp = raw["video_path"].iloc[0] if "video_path" in raw.columns and len(raw) else None
    if vp and not Path(vp).exists():
        vp = None
    return pipeline_from_raw(raw, cfg, out_dir, video_path=vp, calibrator=calibrator)


def batch_status(videos_root: str | Path, models: list[str | Path],
                 cfg: Config, out_dir: str | Path) -> dict:
    """Progress + ETA for a (possibly running) batch, without touching it.

    Enumerates the expected (model, video) pairs, classifies each output folder as
    done / failed / pending, and estimates remaining wall-clock from the pending
    videos' resolutions using the measured calibration (:mod:`runtime`).
    """
    from . import runtime
    from .video_io import find_videos

    out = Path(out_dir)
    videos = find_videos(videos_root, cfg.input)
    done = failed = pending = 0
    remaining_s = 0.0
    pending_names: list[str] = []
    for model_path in models:
        stem = Path(model_path).stem
        for vp in videos:
            run_dir = out / f"{stem}__{vp.stem}"
            if (run_dir / "video_summary.json").exists():
                done += 1
            elif (run_dir / "failure.json").exists():
                failed += 1
            else:
                pending += 1
                pending_names.append(vp.name)
                info = probe_video(vp)
                if info.width and info.height:
                    remaining_s += runtime.estimate_video_seconds(
                        info.width, info.height, info.frame_count, info.fps,
                        info.duration_s, cfg)
    total = done + failed + pending
    return {
        "total_runs": total,
        "done": done,
        "failed": failed,
        "pending": pending,
        "percent_complete": round(100 * (done + failed) / total, 1) if total else 0.0,
        "estimated_remaining_hours": round(remaining_s / 3600, 1),
        "tracker": cfg.tracker.type,
        "pending_videos": pending_names,
    }


def batch_track(
    videos_root: str | Path,
    models: list[str | Path],
    cfg: Config,
    out_dir: str | Path,
    calibrator: Calibrator | None = None,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Run every (model, video) pair into its own output folder.

    When ``skip_existing`` is True, any pair whose output folder already holds a
    finished bundle (a ``video_summary.json``) is left untouched and recorded as
    ``status="skipped"`` — so an interrupted multi-day run can be resumed by
    re-issuing the same command.
    """
    from .video_io import find_videos

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    videos = find_videos(videos_root, cfg.input)
    index_rows = []
    for model_path in models:
        model_path = Path(model_path)
        for vp in videos:
            run_dir = out / f"{model_path.stem}__{vp.stem}"
            row = {"model": model_path.stem, "video": vp.name, "out_dir": str(run_dir)}
            if skip_existing and (run_dir / "video_summary.json").exists():
                row["status"] = "skipped"
                index_rows.append(row)
                continue
            t0 = time.time()
            try:
                summary = run_single_video(vp, model_path, cfg, run_dir, calibrator)
                row["status"] = "ok"
                row["n_tracks"] = summary.get("behaviour", {}).get("total_tracks")
                row["wall_s"] = round(time.time() - t0, 1)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                row["wall_s"] = round(time.time() - t0, 1)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "failure.json").write_text(
                    json.dumps({"video": str(vp), "model": str(model_path),
                                "error": str(exc), "traceback": traceback.format_exc()},
                               indent=2),
                    encoding="utf-8",
                )
            index_rows.append(row)
    index = pd.DataFrame(index_rows)
    index_path = save_table(index, out / "batch_index.parquet")
    if cfg.fair.enabled:
        fair.finalize_fair_file(index_path, cfg,
                                inputs=[{"role": "videos-root", "path": str(videos_root)},
                                        {"role": "models", "path": str(models)}])
    return index


def compare_models(batch_dir: str | Path, out_dir: str | Path,
                   cfg: Config | None = None) -> pd.DataFrame:
    """Compare models on tracking-stability / behaviour-readiness metrics."""
    cfg = cfg or Config()
    batch_dir = Path(batch_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for summ_path in batch_dir.glob("*/video_summary.json"):
        try:
            data = json.loads(summ_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        beh = data.get("behaviour", {})
        qc = data.get("qc", {})
        rows.append({
            # Group by model, not video. Prefer the model_name recorded in the
            # behaviour summary; fall back to the run-folder prefix ("model__video").
            "model": beh.get("model_name") or summ_path.parent.name.split("__")[0],
            "run": summ_path.parent.name,
            "median_track_duration_s": beh.get("median_track_duration_s"),
            "fraction_short_tracks": beh.get("fraction_short_tracks"),
            "large_jump_fraction": beh.get("large_jump_fraction"),
            "interpolation_fraction": beh.get("interpolation_fraction"),
            "mean_active_tracks_per_frame": beh.get("mean_active_tracks_per_frame"),
            "duplicate_track_proxy": qc.get("duplicate_track_proxy"),
            "mean_confidence": beh.get("mean_confidence"),
            "total_tracks": beh.get("total_tracks"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        save_table(df, out / "model_comparison.parquet")
        return df
    # Rank: reward longer tracks & fewer short/jumpy/duplicate tracks.
    agg = df.groupby("model").agg(
        median_track_duration_s=("median_track_duration_s", "median"),
        fraction_short_tracks=("fraction_short_tracks", "median"),
        large_jump_fraction=("large_jump_fraction", "median"),
        duplicate_track_proxy=("duplicate_track_proxy", "median"),
        mean_confidence=("mean_confidence", "median"),
        n_runs=("run", "count"),
    ).reset_index()

    def _rank_score(r):
        dur = r["median_track_duration_s"] if pd.notna(r["median_track_duration_s"]) else 0
        return (
            0.4 * np.tanh(dur / 5.0)
            + 0.2 * (1 - (r["fraction_short_tracks"] or 0))
            + 0.2 * (1 - (r["large_jump_fraction"] or 0))
            + 0.1 * (1 - (r["duplicate_track_proxy"] or 0))
            + 0.1 * (r["mean_confidence"] or 0)
        )

    agg["behaviour_readiness_score"] = agg.apply(_rank_score, axis=1)
    agg = agg.sort_values("behaviour_readiness_score", ascending=False)
    agg_path = save_table(agg, out / "model_comparison.parquet")
    save_table(df, out / "model_comparison_per_run.parquet")
    if cfg.fair.enabled:
        fair.finalize_fair_file(agg_path, cfg,
                                inputs=[{"role": "batch-dir", "path": str(batch_dir)}])
    return agg


# Behaviour metrics rolled up in the aggregate report (those present are used).
_AGG_METRICS = [
    "total_tracks", "mean_active_tracks_per_frame", "mean_track_duration_s",
    "median_track_duration_s", "global_mean_speed_px_s", "population_activity_index",
    "border_occupancy_fraction", "center_occupancy_fraction",
    "mean_nearest_neighbor_distance_px", "aggregation_index",
    "heatmap_entropy_occupancy", "interpolation_fraction", "large_jump_fraction",
    "mean_confidence",
]


def _group_means(df: pd.DataFrame, key: str, metrics: list[str]) -> pd.DataFrame:
    """Mean of present metric columns per ``key``, with an n_runs count."""
    present = [m for m in metrics if m in df.columns]
    grouped = df.groupby(key, dropna=False)
    out = grouped[present].mean(numeric_only=True).reset_index()
    out.insert(1, "n_runs", grouped.size().to_numpy())
    return out


def aggregate_report(batch_dir: str | Path, out_dir: str | Path,
                     cfg: Config | None = None) -> pd.DataFrame:
    """Roll up behaviour metrics across all runs in a batch into one report.

    Reads every ``*/video_summary.json``, builds a per-run table, group means by
    model / species / metadata conditions, a handful of cross-video bar charts,
    and a markdown + HTML report. Complements ``compare_models`` (which ranks
    detectors) by summarising *behaviour* across videos and conditions.
    """
    cfg = cfg or Config()
    batch_dir = Path(batch_dir)
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    rows = []
    for summ_path in sorted(batch_dir.glob("*/video_summary.json")):
        try:
            data = json.loads(summ_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        vi = data.get("video_info", {}) or {}
        ms = data.get("model_settings", {}) or {}
        beh = data.get("behaviour", {}) or {}
        qc = data.get("qc", {}) or {}
        row = {
            "run": summ_path.parent.name,
            "video": vi.get("filename") or summ_path.parent.name,
            "model": beh.get("model_name") or summ_path.parent.name.split("__")[0],
            "tracker": beh.get("tracker") or ms.get("tracker"),
        }
        row.update({k: v for k, v in vi.items() if k.startswith("meta_")})
        row.update({m: beh[m] for m in _AGG_METRICS if m in beh})
        if "duplicate_track_proxy" in qc:
            row["duplicate_track_proxy"] = qc["duplicate_track_proxy"]
        rows.append(row)

    runs = pd.DataFrame(rows)
    runs_path = save_table(runs, out / "aggregate_runs.parquet")

    sections: list[tuple[str, pd.DataFrame]] = [("Per-run behaviour", runs)]
    plot_paths: list[str] = []
    if not runs.empty:
        by_model = _group_means(runs, "model", _AGG_METRICS)
        save_table(by_model, out / "aggregate_by_model.parquet")
        sections.append(("Mean behaviour by model", by_model))

        if "meta_species" in runs.columns and runs["meta_species"].notna().any():
            by_species = _group_means(runs, "meta_species", _AGG_METRICS)
            save_table(by_species, out / "aggregate_by_species.parquet")
            sections.append(("Mean behaviour by species", by_species))

        # Cross-video bar charts (only for metrics/groupings actually present).
        chart_specs = [
            ("model", "median_track_duration_s"),
            ("video", "total_tracks"),
            ("meta_species", "population_activity_index"),
        ]
        for group_col, value_col in chart_specs:
            if group_col in runs.columns and value_col in runs.columns and runs[value_col].notna().any():
                try:
                    from . import visualization as viz
                    p = viz.plot_group_bar(runs, group_col, value_col,
                                           out / "plots" / f"{value_col}_by_{group_col}.png")
                    plot_paths.append(str(Path(p).relative_to(out)))
                except Exception:  # a single bad chart must not kill the report
                    pass

    title = f"{len(runs)} runs from {batch_dir.name}"
    if cfg.reports.make_markdown:
        rep.write_markdown(rep.build_aggregate_markdown(title, sections, plot_paths),
                           out / "aggregate_report.md")
    if cfg.reports.make_html:
        rep.write_html(rep.build_aggregate_html(title, sections, plot_paths),
                       out / "aggregate_report.html")

    if cfg.fair.enabled and not runs.empty:
        fair.finalize_fair_file(runs_path, cfg,
                                inputs=[{"role": "batch-dir", "path": str(batch_dir)}])
    return runs
