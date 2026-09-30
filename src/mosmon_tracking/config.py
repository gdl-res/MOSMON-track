"""Configuration loading.

The pipeline is driven entirely by a YAML file (see ``configs/default.yaml``).
We parse it into nested dataclasses so that every threshold is typed, documented,
and discoverable, while still tolerating partial YAML files (missing keys fall
back to the dataclass defaults).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ProjectConfig:
    name: str = "mosmon_larvae_behaviour"
    random_seed: int = 42


@dataclass
class InputConfig:
    video_extensions: list[str] = field(
        default_factory=lambda: [".mp4", ".mov", ".m4v", ".avi", ".mkv"]
    )
    recursive: bool = True


@dataclass
class TilingConfig:
    """Tiled-inference backend for native-resolution small-object detection.

    When ``enabled``, each frame is detected tile-by-tile at 1:1 pixel fidelity and the
    per-tile boxes are remapped to full-frame coordinates, de-duplicated across tile
    seams, then handed to the tracker. Off by default (full-frame downscale to imgsz).
    Ported from the overlapping-tiling training pipeline in training/.
    """

    enabled: bool = False
    tile_size: int = 640  # 1:1 pixel tiles; match the training tile size
    overlap: float = 0.10  # fraction of tile overlap (matches the shipped weight's training)
    merge_iou: float | None = None  # class-agnostic NMS IoU across seams (None -> model.iou)
    batch: int = 8  # tiles per predict() call
    min_area_frac: float = 0.0  # drop detections smaller than this fraction of tile area (0 = keep all)


@dataclass
class ModelConfig:
    paths: list[str] = field(default_factory=list)
    imgsz: int = 1920
    conf: float = 0.15
    iou: float = 0.5
    device: str = "auto"
    half: bool = True
    classes: list[int] | None = None
    agnostic_nms: bool = False  # merge overlapping boxes across classes (de-duplicate)
    # Ultralytics defaults to 300 detections/frame. MOSMON scenes reach ~350 larvae,
    # so the default silently truncates the densest frames to the top-300 by score.
    max_det: int = 300
    tiling: TilingConfig = field(default_factory=TilingConfig)


@dataclass
class TrackerConfig:
    type: str = "bytetrack"
    yaml: str = "configs/tracker_bytetrack.yaml"
    persist: bool = True
    max_gap_frames: int = 15
    min_track_length_frames: int = 10
    min_mean_confidence: float = 0.10


@dataclass
class VideoConfig:
    frame_stride: int = 1
    max_frames: int | None = None
    save_debug_video: bool = False
    debug_video_fps: int = 20
    debug_video_max_frames: int | None = 300  # cap debug clip length (None = all)
    debug_video_scale: float = 0.35  # downscale factor for the debug video (5.3K -> manageable)
    debug_video_trail: int = 20  # number of past centers to draw as a trail
    # Full annotated tracking video: every processed frame, all tracks, full length.
    # Distinct from the debug clip, which is downscaled and capped for quick QC.
    save_tracking_video: bool = False
    tracking_video_fps: int | None = None  # None = use source video fps
    tracking_video_scale: float = 1.0  # 1.0 = full resolution
    tracking_video_trail: int = 30  # past centers drawn as a fading trail
    # Heatmap-accumulation video: occupancy heatmap building up over time on a frame.
    save_heatmap_video: bool = False
    heatmap_video_fps: int = 20
    heatmap_video_scale: float = 0.5  # downscale factor for the heatmap video
    heatmap_video_max_frames: int | None = 300  # cap output frames (None = all)
    heatmap_video_alpha: float = 0.6  # heatmap opacity over the background frame


@dataclass
class CalibrationConfig:
    enabled: bool = False
    pixels_per_mm: float | None = None
    container_width_cm: float | None = None
    container_height_cm: float | None = None
    homography_path: str | None = None


@dataclass
class PostprocessConfig:
    interpolate_gaps: bool = True
    interpolation_max_gap_frames: int = 15
    smoothing: str = "savgol"  # one of: savgol, none
    smoothing_window_frames: int = 9
    smoothing_polyorder: int = 2
    remove_outlier_jumps: bool = True
    max_speed_px_s: float | None = None


@dataclass
class RegionsConfig:
    border_margin_fraction: float = 0.10
    center_roi_fraction: float = 0.50
    custom_rois_path: str | None = None
    dual_container_split_x_fraction: float | None = None


@dataclass
class HeatmapsConfig:
    bins_x: int = 128
    bins_y: int = 72
    normalize: str = "probability"  # one of: probability, count, density
    per_track: bool = False
    per_class: bool = True
    per_time_window_s: float | None = None


@dataclass
class FeaturesConfig:
    compute_pairwise: bool = True
    pairwise_max_tracks_per_frame: int = 300
    nearest_neighbor_k: int = 1
    activity_speed_threshold_px_s: float = 5.0
    freezing_speed_threshold_px_s: float = 1.0
    burst_speed_quantile: float = 0.90
    export_population_timeseries: bool = True  # write population_timeseries.parquet


@dataclass
class ReportsConfig:
    make_html: bool = True
    make_markdown: bool = True
    make_pdf: bool = True  # also render report.pdf alongside report.html (needs the [pdf] extra)
    save_plots: bool = True
    export_mot: bool = True  # write tracks_mot.txt (MOTChallenge format) + seqinfo.ini
    mot_include_interpolated: bool = True  # include interpolated rows in the MOT export
    export_trajectories_geojson: bool = True  # write trajectories.geojson (one feature per track)


@dataclass
class SpeciesAnalysisConfig:
    """Cross-video species-discrimination analysis (``mosmon species-analysis``).

    Defaults encode the cohort rules established by the 66-video full eval: drop
    duplicate source videos, gate on the duplicate-track proxy and detection
    coverage, and keep only tracks long enough for shape-of-motion features.
    """

    min_track_duration_s: float = 3.0  # tracks shorter than this carry no reliable shape
    apply_qc_gate: bool = True
    max_duplicate_track_proxy: float = 0.15  # crowding-driven over-counting gate
    min_coverage_pct: float = 95.0  # % of processed frames with detections
    drop_duplicate_inputs: bool = True  # identical sha256 source videos
    drop_conflicting_duplicates: bool = True  # both members of a contradictory-label pair
    exclude_dual_container: bool = True  # analysed separately by the paired contrast
    max_tracks_per_video: int | None = 4000  # subsample per video (seeded) to bound cost
    msd_max_lag: int = 20
    autocorr_max_lag: int = 5
    pause_speed_bl_s: float = 0.25  # body-lengths/s below which a larva counts as paused
    density_match_min: float = 4.0  # larvae/frame band where all species are represented
    density_match_max: float = 30.0
    classifier_max_tracks_per_video: int = 1000  # cap rows per video in the track-level fit
    n_permutations: int = 200
    logistic_l2: float = 0.05  # L2 on standardised features, mean-NLL scale
    logistic_l2_grid: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.2, 1.0])
    logistic_max_iter: int = 200
    fdr_alpha: float = 0.05


@dataclass
class FairConfig:
    """Metadata for FAIR-compliant outputs (RO-Crate + provenance)."""

    enabled: bool = True
    hash_inputs: bool = True  # sha256 input video/model (slow for very large videos)
    license: str = "https://creativecommons.org/licenses/by/4.0/"
    license_name: str = "CC-BY-4.0"
    creator_name: str | None = None
    creator_orcid: str | None = None  # full URL, e.g. https://orcid.org/0000-...
    affiliation: str | None = None
    publisher: str | None = None
    project_url: str | None = None
    dataset_keywords: list[str] = field(
        default_factory=lambda: [
            "mosquito larvae", "MOSMON", "object tracking", "behaviour analysis",
            "YOLO11", "trajectory",
        ]
    )


@dataclass
class BenchmarkConfig:
    """Fixed-detector multi-tracker benchmark (configs/tracker_benchmark.yaml)."""

    #: Confidence floor the detection cache is written at. Kept separate from
    #: ``model.conf`` so re-analysis cannot silently redefine the cached pool.
    detection_conf: float = 0.05
    #: Floor applied to the cached pool before it reaches a tracker. ``None``
    #: hands every arm the full pool and lets each apply its own thresholds,
    #: which is the fair default -- pre-filtering at one arm's threshold would
    #: give that arm's assumptions to all the others.
    replay_conf: float | None = None
    #: Floor at which the reproduction gate reconstructs an earlier run's pool.
    reference_conf: float = 0.15
    #: Arm name -> tracker YAML.
    arms: dict[str, str] = field(default_factory=dict)
    #: Arm used as the anchor when reporting relative differences. It is a
    #: reference point, not ground truth.
    reference_arm: str = "botsort"
    #: Encoder tried if an appearance arm's configured ReID model fails to load.
    reid_fallback: str | None = None


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    regions: RegionsConfig = field(default_factory=RegionsConfig)
    heatmaps: HeatmapsConfig = field(default_factory=HeatmapsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    reports: ReportsConfig = field(default_factory=ReportsConfig)
    fair: FairConfig = field(default_factory=FairConfig)
    species_analysis: SpeciesAnalysisConfig = field(
        default_factory=SpeciesAnalysisConfig
    )
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, data: Any):
    """Recursively construct a dataclass from a (possibly partial) mapping.

    Unknown keys are ignored with no error so configs stay forward-compatible;
    missing keys fall back to dataclass defaults.
    """
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs: dict[str, Any] = {}
    # Resolve string annotations (we use ``from __future__ import annotations``)
    # back to real types so nested dataclasses can be detected.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            continue  # forward-compatible: ignore unknown keys
        field_type = hints.get(key)
        if is_dataclass(field_type):
            kwargs[key] = _build(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def find_unknown_keys(data: Any, cls: type = Config, prefix: str = "") -> list[str]:
    """Return dotted paths of keys in ``data`` not present on the ``cls`` schema.

    The loader silently ignores unknown keys (forward-compatibility), which means
    a typo like ``save_traking_video`` does nothing instead of erroring. This walks
    the same structure to surface those so a long run isn't launched on a config
    that quietly drops a setting.
    """
    unknown: list[str] = []
    if not is_dataclass(cls) or not isinstance(data, dict):
        return unknown
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in known:
            unknown.append(path)
            continue
        field_type = hints.get(key)
        if is_dataclass(field_type) and isinstance(value, dict):
            unknown.extend(find_unknown_keys(value, field_type, prefix=f"{path}."))
    return unknown


def load_config(path: str | Path | None = None) -> Config:
    """Load a :class:`Config` from a YAML file, or return defaults if ``path`` is None."""
    if path is None:
        return Config()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build(Config, raw)


def dump_config(config: Config, path: str | Path) -> None:
    """Write a config back to YAML (useful for recording the exact run settings)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False)
