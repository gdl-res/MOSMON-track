"""Command-line interface for the MOSMON-Larvae tracking/behaviour pipeline.

Commands
--------
* ``inspect-videos``  - build a video inventory with parsed filename metadata.
* ``track-video``     - run YOLO tracking + full analysis on one video.
* ``batch-track``     - run every (model, video) pair.
* ``analyse-tracks``  - recompute features/reports from existing raw tracks (no YOLO).
* ``compare-models``  - rank models by behaviour-readiness.
* ``aggregate-report``- roll up behaviour metrics across all runs in a batch.
* ``species-analysis``- test whether tracking metrics distinguish species.
* ``detect-videos``   - build the frozen detection cache (detector only, no tracker).
* ``benchmark-track`` - replay that cache through every tracker arm.
* ``validate-config`` - flag unknown/typo'd config keys the loader would ignore.
* ``preflight``       - pre-run checks (readability, model, GPU, disk, ETA).
* ``batch-status``    - progress + ETA for a running/finished batch.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .calibration import Calibrator
from .config import load_config
from .video_io import build_inventory, save_table

app = typer.Typer(add_completion=False, help="MOSMON-Larvae tracking & behaviour analysis.")


def _load(config: str | None):
    return load_config(config)


@app.command("inspect-videos")
def inspect_videos(
    videos: str = typer.Option(..., help="Folder (or single file) of raw videos."),
    out: str = typer.Option("outputs/video_inventory.csv", help="Output inventory path."),
    config: str = typer.Option(None, help="Config YAML (defaults if omitted)."),
):
    """Probe videos (ffprobe/OpenCV) and parse filename metadata into a table."""
    cfg = _load(config)
    inv = build_inventory(videos, cfg.input)
    path = save_table(inv, out)
    if cfg.fair.enabled:
        from .fair import finalize_fair_file

        finalize_fair_file(path, cfg, logical_table="video_inventory",
                           inputs=[{"role": "videos-root", "path": str(videos)}])
    typer.echo(f"Wrote inventory with {len(inv)} videos -> {path} (+ FAIR sidecars)")


@app.command("track-video")
def track_video_cmd(
    video: str = typer.Option(..., help="Path to one video."),
    model: str = typer.Option(..., help="Path to a YOLO11 .pt model."),
    config: str = typer.Option(None, help="Config YAML."),
    out: str = typer.Option(..., help="Output run directory."),
    save_tracking_video: bool = typer.Option(
        None, "--save-tracking-video/--no-save-tracking-video",
        help="Write a full-resolution, full-length annotated tracking_video.mp4 "
             "(overrides config; off by default).",
    ),
):
    """Run YOLO tracking on one video and produce the full analysis bundle."""
    from .batch import run_single_video

    cfg = _load(config)
    if save_tracking_video is not None:
        cfg.video.save_tracking_video = save_tracking_video
    calibrator = Calibrator.from_config(cfg.calibration)
    summary = run_single_video(video, model, cfg, out, calibrator)
    n = summary.get("behaviour", {}).get("total_tracks", 0)
    typer.echo(f"Done: {n} tracks. Outputs in {out}")


@app.command("batch-track")
def batch_track_cmd(
    videos: str = typer.Option(..., help="Folder of videos."),
    models: str = typer.Option(..., help="Folder of .pt models OR comma-separated paths."),
    config: str = typer.Option(None, help="Config YAML."),
    out: str = typer.Option(..., help="Output batch directory."),
    save_tracking_video: bool = typer.Option(
        None, "--save-tracking-video/--no-save-tracking-video",
        help="Write a full-resolution, full-length annotated tracking_video.mp4 per run "
             "(overrides config; off by default).",
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing",
        help="Skip (model, video) pairs whose output folder already has a finished "
             "bundle (video_summary.json). Use to resume an interrupted run.",
    ),
):
    """Run all (model, video) pairs, one output folder each."""
    from .batch import batch_track

    cfg = _load(config)
    if save_tracking_video is not None:
        cfg.video.save_tracking_video = save_tracking_video
    model_paths = _resolve_models(models)
    if not model_paths:
        raise typer.BadParameter(f"no .pt models found in {models}")
    calibrator = Calibrator.from_config(cfg.calibration)
    index = batch_track(videos, model_paths, cfg, out, calibrator, skip_existing=skip_existing)
    ok = (index["status"] == "ok").sum() if not index.empty else 0
    skipped = (index["status"] == "skipped").sum() if not index.empty else 0
    msg = f"Batch done: {ok}/{len(index)} runs ok"
    if skipped:
        msg += f", {skipped} skipped"
    typer.echo(msg + f". Index in {out}/batch_index.parquet")


@app.command("detect-videos")
def detect_videos_cmd(
    videos: str = typer.Option(..., help="Folder (or single file) of videos."),
    model: str = typer.Option(..., help="Path to the YOLO .pt weights."),
    out: str = typer.Option(..., help="Detection cache directory."),
    config: str = typer.Option("configs/tracker_benchmark.yaml", help="Config YAML."),
    skip_existing: bool = typer.Option(
        False, "--skip-existing",
        help="Skip videos whose detection parquet + sidecar already exist.",
    ),
):
    """Detector-only pass: the fixed detections every tracker arm replays.

    Runs no tracker, so nothing here is conditioned on an association algorithm.
    """
    from .benchmark.detect import detect_batch

    cfg = _load(config)
    index = detect_batch(videos, model, cfg, out, skip_existing=skip_existing)
    if index.empty:
        typer.echo("No videos found.")
        raise typer.Exit(code=1)
    ok = int((index["status"] == "ok").sum())
    skipped = int((index["status"] == "skipped").sum())
    failed = int((index["status"] == "failed").sum())
    typer.echo(f"Detection cache: {ok} ok, {skipped} skipped, {failed} failed -> {out}")
    if failed:
        raise typer.Exit(code=1)


@app.command("benchmark-track")
def benchmark_track_cmd(
    detections: str = typer.Option(..., help="Detection cache directory (from detect-videos)."),
    out: str = typer.Option(..., help="Benchmark output directory."),
    videos: str = typer.Option(None, help="Folder of source videos; required for arms "
                                          "using GMC or ReID."),
    config: str = typer.Option("configs/tracker_benchmark.yaml", help="Config YAML."),
    arms: str = typer.Option(None, help="Comma-separated arm names (default: all in config)."),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip videos already replayed."),
    native: bool = typer.Option(
        True, "--native/--no-native",
        help="Also write the tracker-native (Experiment A) tables."),
):
    """Replay the frozen detections through every tracker arm."""
    from .benchmark.batch import benchmark_batch

    cfg = _load(config)
    names = [a.strip() for a in arms.split(",")] if arms else None
    index = benchmark_batch(detections, cfg, out, videos_dir=videos,
                            arm_names=names, skip_existing=skip_existing,
                            write_native=native)
    if index.empty:
        typer.echo("No detection caches found.")
        raise typer.Exit(code=1)
    ok = int((index["status"] == "ok").sum())
    failed = int((index["status"] == "failed").sum())
    typer.echo(f"Benchmark: {ok}/{len(index)} videos ok, {failed} failed -> {out}")
    if failed:
        raise typer.Exit(code=1)


@app.command("benchmark-validate")
def benchmark_validate_cmd(
    detections: str = typer.Option(..., help="Detection cache directory."),
    video: str = typer.Option(..., help="Source video to validate against."),
    reference: str = typer.Option(..., help="Reference tracks_raw.parquet from a previous run."),
    config: str = typer.Option("configs/tracker_benchmark.yaml", help="Config YAML."),
    arm: str = typer.Option("botsort", help="Arm to validate."),
    tracker_yaml: str = typer.Option("configs/tracker_botsort.yaml", help="Arm's tracker YAML."),
    max_frames: int = typer.Option(0, help="Limit processed frames (0 = whole video)."),
):
    """Check that an arm reproduces a previously produced track table (§6.5)."""
    import json as _json

    from .benchmark.validate import validate_arm

    cfg = _load(config)
    result = validate_arm(detections, video, reference, cfg, arm=arm,
                          tracker_yaml=tracker_yaml,
                          max_frames=max_frames or None)
    typer.echo(_json.dumps(result["comparison"], indent=2))
    typer.echo(f"\nVERDICT: {'PASS' if result['passed'] else 'FAIL'} - {result['message']}")
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command("validate-config")
def validate_config_cmd(
    config: str = typer.Option(..., help="Config YAML to validate."),
):
    """Check a config for unknown/typo'd keys that the loader would silently ignore."""
    import yaml

    from .config import find_unknown_keys

    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    unknown = find_unknown_keys(raw)
    if not unknown:
        typer.echo(f"OK: no unknown keys in {config}")
        return
    typer.echo(f"Found {len(unknown)} unknown key(s) (silently ignored by the loader):")
    for k in unknown:
        typer.echo(f"  - {k}")
    raise typer.Exit(code=1)


@app.command("preflight")
def preflight_cmd(
    videos: str = typer.Option(..., help="Folder of videos."),
    models: str = typer.Option(..., help="Folder of .pt models OR comma-separated paths."),
    out: str = typer.Option(..., help="Intended batch output directory."),
    config: str = typer.Option(None, help="Config YAML."),
):
    """Pre-run checks before a long batch: readability, model, GPU, disk, config, ETA."""
    import json

    from .preflight import run_preflight

    cfg = _load(config)
    rep = run_preflight(videos, _resolve_models(models), cfg, out, config_path=config)
    typer.echo(json.dumps(rep, indent=2))
    status = "READY" if rep["ok"] else "BLOCKED"
    typer.echo(f"\n{status}: {rep['n_runs']} runs, ~{rep['estimated_runtime_hours']} h "
               f"({rep['tracker']}), {rep['disk_free_gb']} GB free.")
    if not rep["ok"]:
        for b in rep["blockers"]:
            typer.echo(f"  BLOCKER: {b}")
        raise typer.Exit(code=1)
    for w in rep["warnings"]:
        typer.echo(f"  warning: {w}")


@app.command("batch-status")
def batch_status_cmd(
    batch: str = typer.Option(..., help="Batch output directory being filled."),
    videos: str = typer.Option(..., help="The same videos folder the batch is running on."),
    models: str = typer.Option(..., help="The same models folder/paths the batch is using."),
    config: str = typer.Option(None, help="Config YAML (for stride/tracker in the ETA)."),
):
    """Show progress and an ETA for a running/finished batch."""
    from .batch import batch_status

    cfg = _load(config)
    s = batch_status(videos, _resolve_models(models), cfg, batch)
    typer.echo(f"{s['done']}/{s['total_runs']} done"
               f"  ({s['percent_complete']}%)  failed={s['failed']}  pending={s['pending']}")
    typer.echo(f"ETA: ~{s['estimated_remaining_hours']} h remaining ({s['tracker']})")


@app.command("analyse-tracks")
def analyse_tracks_cmd(
    tracks: str = typer.Option(..., help="raw tracks file OR a run/batch directory."),
    config: str = typer.Option(None, help="Config YAML."),
    out: str = typer.Option(..., help="Output analysis directory."),
):
    """Recompute behaviour features and reports from existing raw tracks."""
    from .batch import analyse_raw

    cfg = _load(config)
    calibrator = Calibrator.from_config(cfg.calibration)
    tracks_path = Path(tracks)
    # If a batch directory of run subfolders, analyse each; else single.
    run_dirs = [p for p in tracks_path.glob("*") if p.is_dir() and
                any((p / n).exists() for n in ("tracks_raw.parquet", "tracks_raw.csv",
                                               "detections.parquet", "detections.csv"))] \
        if tracks_path.is_dir() else []
    if run_dirs:
        for rd in run_dirs:
            analyse_raw(rd, cfg, Path(out) / rd.name, calibrator)
        typer.echo(f"Analysed {len(run_dirs)} runs -> {out}")
    else:
        analyse_raw(tracks, cfg, out, calibrator)
        typer.echo(f"Analysed tracks -> {out}")


@app.command("compare-models")
def compare_models_cmd(
    batch: str = typer.Option(..., help="Batch output directory."),
    out: str = typer.Option(..., help="Output comparison directory."),
    config: str = typer.Option(None, help="Config YAML."),
):
    """Rank models by tracking stability and behaviour-readiness."""
    from .batch import compare_models

    cfg = _load(config)
    table = compare_models(batch, out, cfg)
    typer.echo(f"Compared {len(table)} models -> {out}/model_comparison.parquet")


@app.command("aggregate-report")
def aggregate_report_cmd(
    batch: str = typer.Option(..., help="Batch output directory."),
    out: str = typer.Option(..., help="Output aggregate-report directory."),
    config: str = typer.Option(None, help="Config YAML."),
):
    """Roll up behaviour metrics across all runs into one cross-video report."""
    from .batch import aggregate_report

    cfg = _load(config)
    runs = aggregate_report(batch, out, cfg)
    typer.echo(f"Aggregated {len(runs)} runs -> {out}/aggregate_report.html")


@app.command("species-analysis")
def species_analysis_cmd(
    batch: str = typer.Option(..., help="Batch output directory."),
    out: str = typer.Option(..., help="Output species-analysis directory."),
    config: str = typer.Option(None, help="Config YAML."),
    min_track_duration: float = typer.Option(
        None, help="Override species_analysis.min_track_duration_s."),
    no_gate: bool = typer.Option(False, "--no-gate",
                                help="Skip the duplicate/coverage QC gate (sensitivity run)."),
    permutations: int = typer.Option(None, help="Override the permutation-null count."),
    dual_batch: str = typer.Option(
        None, help="Directory of dual-container runs re-analysed with "
                   "regions.dual_container_split_x_fraction set (for the within-video contrast)."),
):
    """Test whether tracking metrics distinguish species (video-disjoint, confound-controlled)."""
    from .species_analysis import species_analysis

    cfg = _load(config)
    if min_track_duration is not None:
        cfg.species_analysis.min_track_duration_s = min_track_duration
    if no_gate:
        cfg.species_analysis.apply_qc_gate = False
    if permutations is not None:
        cfg.species_analysis.n_permutations = permutations
    dual_runs = None
    if dual_batch:
        dual_runs = {d.name: str(d) for d in Path(dual_batch).iterdir() if d.is_dir()}
    res = species_analysis(batch, out, cfg, dual_run_dirs=dual_runs)
    typer.echo(res["headline"])
    for w in res["warnings"]:
        typer.echo(f"WARNING: {w}")
    typer.echo(f"Wrote {out}/species_report.html")


def _resolve_models(models: str) -> list[str]:
    if "," in models:
        return [m.strip() for m in models.split(",") if m.strip()]
    p = Path(models)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.glob("*.pt"))]
    return []


if __name__ == "__main__":
    app()
