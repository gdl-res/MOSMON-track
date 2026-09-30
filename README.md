# MOSMON-Larvae Tracking & Behaviour Analysis

A command-line tool that takes **videos of mosquito larvae** and automatically
produces **tracks, movement measurements, heatmaps, and a report** for each video.

You do not need to be a programmer to use it. If you can open a terminal and copy
and paste a few commands, you can run the whole pipeline. This README explains
every step, every command, every option, and every file it produces.

---

## Table of contents

1. [What this tool does](#1-what-this-tool-does)
2. [Words you need to know (plain-language glossary)](#2-words-you-need-to-know)
3. [What you need before you start](#3-what-you-need-before-you-start)
4. [Installation (one time)](#4-installation-one-time)
5. [Where to put your videos and models](#5-where-to-put-your-videos-and-models)
6. [Quick start (3 commands)](#6-quick-start)
7. [The commands, explained one by one](#7-the-commands-explained-one-by-one)
8. [The configuration file, explained](#8-the-configuration-file-explained)
9. [The output files, explained](#9-the-output-files-explained)
10. [Tuning the tracker](#10-tuning-the-tracker)
11. [Calibration: turning pixels into millimetres](#11-calibration-turning-pixels-into-millimetres)
12. [High-resolution videos (8K) and tiling](#12-high-resolution-videos-8k-and-tiling)
13. [FAIR outputs (shareable, citable data)](#13-fair-outputs)
14. [How filenames are read (metadata)](#14-how-filenames-are-read)
15. [How to read the results responsibly](#15-how-to-read-the-results-responsibly)
16. [Troubleshooting](#16-troubleshooting)
17. [For developers (tests, layout)](#17-for-developers)
18. [Manuscript figures](#18-manuscript-figures)
19. [Multi-tracker benchmark](#19-multi-tracker-benchmark-fixed-detector)
20. [License and acknowledgements](#20-license-and-acknowledgements)

---

## 1. What this tool does

You give it:

- a folder of **videos** of mosquito larvae swimming in containers;
- one or more **trained YOLO11 models** (files ending in `.pt`) that know how to
  spot a larva in an image.

It gives you, for every video:

- the position of every larva in every frame (**detections**);
- those positions joined over time into **tracks** (one path per larva);
- cleaned-up trajectories with **speed, acceleration, turning, etc.**;
- **heatmaps** showing where the larvae spent their time;
- **plots** (trajectories, speed over time, distributions…);
- a **quality report** telling you how trustworthy the tracking was;
- a human-readable **report** (`report.md` / `report.html`);
- **FAIR metadata** so the results are documented, checksummed, and citable.

The flow looks like this:

```
                ┌─────────────┐   ┌────────────┐   ┌──────────────┐
 your video ──▶ │ YOLO11      │──▶│ tracker    │──▶│ raw tracks   │
                │ (finds      │   │ (follows   │   │ (positions   │
                │  larvae)    │   │  each one) │   │  per frame)  │
                └─────────────┘   └────────────┘   └──────┬───────┘
                                                          │
            ┌─────────────────────────────────────────────┘
            ▼
   ┌─────────────────┐   ┌───────────────────┐   ┌────────────────────────┐
   │ cleaned tracks  │──▶│ behaviour features │──▶│ heatmaps, plots,       │
   │ (gaps filled,   │   │ (speed, tortuosity,│   │ summaries, QC, report, │
   │  smoothed)      │   │  heatmap entropy…) │   │ FAIR metadata          │
   └─────────────────┘   └───────────────────┘   └────────────────────────┘
```

---

## 2. Words you need to know

| Word | Plain meaning |
| --- | --- |
| **Frame** | One still image from the video. A 30 fps video has 30 frames per second. |
| **Detection** | The tool finding a larva in one frame and drawing a box around it. |
| **Bounding box** | The rectangle around a detected larva (left, top, right, bottom). |
| **Track** | One larva followed across many frames. Each track has an **ID** (a number). |
| **Trajectory** | The path a track traces over time. |
| **Tracker** | The algorithm that decides "the larva here now is the same one I saw before." Two options: **ByteTrack** and **BoT-SORT**. |
| **Confidence** | How sure the model is that a box really is a larva (0 to 1). |
| **Kinematics** | Movement numbers: speed, acceleration, turning angle. |
| **Heatmap** | A coloured image showing which areas of the container were used most. |
| **Calibration** | Converting pixels into real units (millimetres). Off by default. |
| **FPS** | Frames per second — the video's speed. Needed to compute real-time speeds. |
| **Parquet / CSV** | Table file formats. Parquet is compact; CSV opens in Excel. |
| **YOLO11** | The AI model type that detects larvae. The `.pt` file is the trained model. |
| **FAIR** | A standard for making data **F**indable, **A**ccessible, **I**nteroperable, **R**eusable. |

---

## 3. What you need before you start

- A computer with **Linux** (macOS and Windows should work but are untested).
- **Conda** (Miniconda/Anaconda) or any Python >= 3.10 environment.
- An **NVIDIA GPU** is strongly recommended for speed (development used an
  RTX 4060). It also runs on CPU, just much slower.
- Your **videos** and your **trained `.pt` model(s)**.

---

## 4. Installation (one time)

**Step 1 — create the environment:**

```bash
conda create -n mosmon_tracker python=3.12 -y
```

**Step 2 — activate it (do this every time you open a new terminal):**

```bash
conda activate mosmon_tracker
```

**Step 3 — install PyTorch with GPU support.** Pick the build that matches your
CUDA driver from <https://pytorch.org/get-started/locally/>; for example, for CUDA 13:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Step 4 — install this tool and all its dependencies:**

```bash
git clone https://github.com/gdl-res/MOSMON-track.git
cd MOSMON-track
pip install -e .
```

That last command installs everything (Ultralytics/YOLO, OpenCV, pandas, etc.)
and creates the `mosmon` command you will use.

**Check it worked:**

```bash
mosmon --help
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

You should see a list of commands and `GPU available: True`.

> **Always run `conda activate mosmon_tracker` first** in any new terminal,
> otherwise the `mosmon` command will not be found.

---

## 5. Where to put your videos and models

The project has ready-made folders:

```
data/
  raw_videos/     ← put your videos here (or point --videos anywhere)
  models/         ← put your YOLO11 .pt files here (or point --model anywhere)
  metadata/       ← optional extra metadata
outputs/          ← everything the tool produces goes here
configs/          ← settings files (you edit these to change behaviour)
```

**Model weights are not included in this repository** (they are too large for git).
Download the trained MOSMON-Larvae YOLO11 weight
(`mosmon_yolov11x_overlap_tiling_best_v1.pt`) from the location given in the
associated publication and place it in `weights/` (the path `configs/models.yaml` and
the benchmark scripts expect) or in `data/models/`. Any other YOLO11 detection `.pt`
works too.

You do **not** have to copy huge videos into `data/raw_videos/`. Every command
lets you point at videos and models wherever they already live on disk, using the
`--videos` / `--video` / `--models` / `--model` options.

---

## 6. Quick start

Three commands take you from raw videos to a full analysis. Run them with the
environment activated (`conda activate mosmon_tracker`).

```bash
# (1) Look at your videos and read their filenames into a table
mosmon inspect-videos --videos data/raw_videos --out outputs/video_inventory.csv

# (2) Track ONE video with ONE model and analyse it fully
mosmon track-video \
  --video data/raw_videos/example.mp4 \
  --model data/models/yolo11_best.pt \
  --config configs/default.yaml \
  --out outputs/example_run

# (3) Open the report
xdg-open outputs/example_run/report.html
```

When you are happy with the settings, run everything in one go with
`batch-track` (section 7.3).

---

## 7. The commands, explained one by one

There are **five** commands. Each one is `mosmon <command> --options`.
Run `mosmon <command> --help` at any time to see its options.

### 7.1 `inspect-videos` — make an inventory of your videos

**What it does:** looks at every video in a folder, reads its width, height, fps,
duration and codec, **reads the metadata hidden in the filename** (species, camera,
lighting, container, etc.), and writes one big table.

**When to use:** first thing, to check your videos are readable and your filenames
are understood.

```bash
mosmon inspect-videos \
  --videos data/raw_videos \
  --out outputs/video_inventory.csv
```

| Option | Meaning |
| --- | --- |
| `--videos` | Folder of videos (searched, including subfolders) **or** a single video file. |
| `--out` | Where to write the inventory table (`.csv` or `.parquet`). |
| `--config` | (Optional) settings file; defaults are used if omitted. |

**Produces:** `video_inventory.csv` plus FAIR sidecars
(`*.provenance.json`, `*.sha256`, `*.data_dictionary.json`).
Open the CSV in Excel/LibreOffice to check everything looks right.

### 7.2 `track-video` — analyse one video

**What it does:** runs the YOLO model on one video, tracks each larva, cleans the
tracks, computes all the behaviour measurements, and writes the full output bundle
(tables + plots + heatmaps + report + FAIR metadata).

**When to use:** to process a single video, or to test your settings before a big
batch run.

```bash
mosmon track-video \
  --video data/raw_videos/example.mp4 \
  --model data/models/yolo11_best.pt \
  --config configs/default.yaml \
  --out outputs/example_run
```

| Option | Meaning |
| --- | --- |
| `--video` | The one video to analyse. |
| `--model` | The YOLO11 `.pt` model file to use. |
| `--config` | Settings file (use `configs/default.yaml` to start). |
| `--out` | A folder to create for this run's outputs. |

**Produces:** a folder with everything (see [section 9](#9-the-output-files-explained)).

### 7.3 `batch-track` — analyse many videos with one or more models

**What it does:** the same as `track-video`, but for **every combination** of model
and video. If you have 2 models and 50 videos, you get 100 runs, each in its own
sub-folder.

**When to use:** the real job, once your settings are good.

```bash
mosmon batch-track \
  --videos data/raw_videos \
  --models data/models \
  --config configs/default.yaml \
  --out outputs/batch_run
```

| Option | Meaning |
| --- | --- |
| `--videos` | Folder of videos. |
| `--models` | Folder of `.pt` files, **or** a comma-separated list of model paths. |
| `--config` | Settings file. |
| `--out` | Folder to hold all the runs. |
| `--skip-existing` | Skip pairs whose output folder already has a finished `video_summary.json`. Re-issue the same command to **resume** an interrupted (e.g. multi-day) run. |

**Produces:** one sub-folder per `(model, video)` pair named `modelname__videoname/`,
plus `batch_index.parquet` listing every run and whether it succeeded, failed, or was skipped.
If a video fails, a `failure.json` is written for that run and the batch keeps going.

### 7.4 `analyse-tracks` — recompute results without re-running the AI

**What it does:** takes tracks that were already produced (the `tracks_raw` table)
and recomputes all the behaviour features, plots, heatmaps and reports — **without**
running YOLO again. This is fast.

**When to use:** when you change a setting that only affects analysis (e.g. speed
thresholds, heatmap bins, calibration) and don't want to wait for YOLO again.

```bash
# From a single run's raw tracks:
mosmon analyse-tracks \
  --tracks outputs/example_run/tracks_raw.parquet \
  --config configs/default.yaml \
  --out outputs/example_analysis

# Or point at a whole run folder, or a whole batch folder:
mosmon analyse-tracks --tracks outputs/batch_run --config configs/default.yaml --out outputs/reanalysis
```

| Option | Meaning |
| --- | --- |
| `--tracks` | A `tracks_raw.parquet`/`.csv` file, **or** a run folder, **or** a batch folder (each run is re-analysed). |
| `--config` | Settings file (change this to change the analysis). |
| `--out` | Output folder. |

### 7.5 `compare-models` — rank models by tracking quality

**What it does:** after a batch run with several models, compares them — not just on
"how many larvae did it find," but on **how stable and usable the tracks are**
(long tracks, few fragments, few impossible jumps, few duplicates). Produces a
ranked table.

**When to use:** to decide which trained model is best for behaviour analysis.

```bash
mosmon compare-models \
  --batch outputs/batch_run \
  --out outputs/model_comparison
```

| Option | Meaning |
| --- | --- |
| `--batch` | The batch folder produced by `batch-track`. |
| `--out` | Output folder for the comparison. |
| `--config` | (Optional) settings file. |

**Produces:** `model_comparison.parquet` (ranked summary per model) and
`model_comparison_per_run.parquet` (the per-run numbers behind it). A higher
`behaviour_readiness_score` is better. Note: the model that detects the *most*
larvae is not always the best — longer, more stable tracks matter more for
behaviour.

### 7.6 `aggregate-report` — roll up behaviour across all videos

**What it does:** after a batch run, reads every run's `video_summary.json` and
builds one **cross-video report**: a per-run table plus mean behaviour grouped by
model and by species, a few cross-video bar charts, and a rendered HTML/Markdown
page. Where `compare-models` ranks *detectors*, this summarises *behaviour* across
videos and conditions.

**When to use:** to see population-level patterns across your whole dataset
(e.g. activity by species, tracks per video) in one place.

```bash
mosmon aggregate-report \
  --batch outputs/batch_run \
  --out outputs/aggregate
```

| Option | Meaning |
| --- | --- |
| `--batch` | The batch folder produced by `batch-track`. |
| `--out` | Output folder for the aggregate report. |
| `--config` | (Optional) settings file. |

**Produces:** `aggregate_runs.parquet` (one row per run), `aggregate_by_model.parquet`,
`aggregate_by_species.parquet` (when species metadata is present), `plots/` with
cross-video bar charts, and `aggregate_report.html` / `aggregate_report.md`.

### 7.7 `species-analysis` — can tracking tell the species apart?

**What it does:** takes a finished batch and asks whether any tracking-derived
metric distinguishes the species — properly, not by eyeballing a per-species mean.
It builds a clean cohort (drops sha256-duplicate source videos, drops both members
of a duplicate pair whose labels disagree, sets dual-container videos aside, and
applies a QC gate on `duplicate_track_proxy` and detection coverage), recomputes
**scale-free** per-track features (speeds and distances divided by the animal's own
body length, so a 1080p and an 8K video are comparable), aggregates them per video,
and then tests them with the video — never the track — as the unit of analysis.

**Why the extra machinery:** larval density spans ~200x across videos and is
confounded with species, so every test is run three ways (raw, density-adjusted,
density-matched), and every classification score is printed beside a permutation
null, a density-only baseline, and the same features asked to predict the *camera*
instead of the species. Cross-validation is video-disjoint (and camera- and
session-disjoint), because tracks from one video are not independent samples.

```bash
mosmon species-analysis \
  --batch outputs/batch_run \
  --out outputs/species_analysis
```

| Option | Meaning |
| --- | --- |
| `--batch` | The batch folder produced by `batch-track`. |
| `--out` | Output folder for the analysis. |
| `--config` | (Optional) settings file; see `species_analysis:` in section 8. |
| `--min-track-duration` | Keep only tracks at least this long (seconds, default 3). |
| `--no-gate` | Skip the QC gate — useful as a sensitivity run, not as the default. |
| `--permutations` | Number of label permutations for the null (default 200). |
| `--dual-batch` | Folder of dual-container runs re-analysed with a compartment split (section 11), enabling the within-video species contrast. |

**Produces:** `cohort.parquet` (every run with an `excluded_reason`),
`track_features.parquet`, `video_features.parquet`, `univariate_tests.parquet`,
`pairwise_tests.parquet`, `morphometric_tests.parquet`, `confound_audit.parquet`,
`classification_results.json`, `plots/`, and `species_report.html` / `.md`.

**Read it in this order:** the confound audit first (what species is entangled
with in *your* data), then the classification table *with* its baselines, then the
density-adjusted and density-matched univariate tables. The raw table is included
only to show what the confounded version would have told you.

### 7.8 Operational utilities — `validate-config`, `preflight`, `batch-status`

For long batch runs (the 66-video job is multi-day), three helpers de-risk and
monitor the run. They read-only; none of them touch your data.

```bash
# 1. Catch typo'd config keys the loader would SILENTLY ignore (e.g. save_traking_video).
mosmon validate-config --config configs/eval.yaml

# 2. Pre-run checks before committing the machine for days: every video readable,
#    model present, GPU available, config clean, enough disk, and a runtime/disk ETA.
mosmon preflight --videos data/raw_videos --models data/models \
  --config configs/eval.yaml --out outputs/full_eval

# 3. Progress + ETA for a running (or finished) batch, any time.
mosmon batch-status --batch outputs/full_eval \
  --videos data/raw_videos --models data/models --config configs/eval.yaml
```

`preflight` exits non-zero (and lists **BLOCKER**s) on any hard problem — unreadable
video, missing model, CUDA requested but unavailable, unknown config key, or
insufficient disk — and warns when estimated output exceeds 80% of free space. Its
runtime and disk estimates come from an on-machine calibration (`src/mosmon_tracking/runtime.py`).
`batch_index.parquet` now also records `wall_s` per run, so ETAs become data-driven over time.

---

## 8. The configuration file, explained

All behaviour is controlled by a YAML settings file: **`configs/default.yaml`**.
You can copy it, edit your copy, and pass it with `--config`. You never edit code.

YAML is just `key: value`. Indentation matters (use spaces). Here is every section
in plain language.

```yaml
project:
  name: mosmon_larvae_behaviour   # a label for your project
  random_seed: 42                 # keeps results repeatable

input:
  video_extensions: [".mp4", ".mov", ".m4v", ".avi", ".mkv"]  # which files count as videos
  recursive: true                 # also look in sub-folders

model:
  paths: []                       # (optional) list of model files
  imgsz: 1920                     # size the video is resized to before detection
  conf: 0.15                      # minimum confidence to keep a detection (lower = more, noisier)
  iou: 0.5                        # overlap threshold for merging duplicate boxes
  device: "auto"                  # "auto" picks GPU if available; or "cpu", "cuda:0"
  half: true                      # use faster 16-bit maths on GPU
  classes: null                   # null = all species; or e.g. [0,2] to keep only some
  agnostic_nms: false             # true = merge overlapping boxes ACROSS classes (de-duplicate)
  max_det: 3000                   # max detections kept per frame

tracker:
  type: bytetrack                 # "bytetrack" or "botsort"
  yaml: configs/tracker_bytetrack.yaml   # the tracker's own settings file
  persist: true                   # keep IDs across frames (always true for video)
  max_gap_frames: 15              # how many missing frames a track may bridge
  min_track_length_frames: 10     # drop tracks shorter than this (too short to trust)
  min_mean_confidence: 0.10       # drop tracks whose average confidence is too low

video:
  frame_stride: 1                 # 1 = every frame; 2 = every other frame (faster, less precise)
  max_frames: null                # null = whole video; or a number to stop early (for testing)
  save_debug_video: false         # true = write an annotated MP4 (boxes + IDs + trails)
  debug_video_fps: 20             # playback fps of the debug video
  debug_video_max_frames: 300     # cap debug clip length (null = all processed frames)
  debug_video_scale: 0.35         # downscale factor (5.3K -> manageable file size)
  debug_video_trail: 20           # number of past centres drawn as a fading trail
  save_tracking_video: false      # true = write the full annotated video (all frames, all tracks)
  tracking_video_fps: null        # null = use the source video fps
  tracking_video_scale: 1.0       # 1.0 = full resolution (large files on high-res sources)
  tracking_video_trail: 30        # number of past centres drawn as a fading trail
  save_heatmap_video: false       # true = write the heatmap-accumulation video
  heatmap_video_fps: 20           # playback fps of the heatmap video
  heatmap_video_scale: 0.5        # downscale factor for the heatmap video
  heatmap_video_max_frames: 300   # cap output frames (null = all)
  heatmap_video_alpha: 0.6        # heatmap opacity over the background frame

calibration:                      # see section 11
  enabled: false                  # false = results are in PIXELS
  pixels_per_mm: null             # set this to convert to millimetres
  container_width_cm: null
  container_height_cm: null
  homography_path: null

postprocess:                      # cleaning the tracks
  interpolate_gaps: true          # fill short gaps by drawing a straight line
  interpolation_max_gap_frames: 15  # only fill gaps up to this length
  smoothing: savgol               # smooth the path ("savgol") or "none"
  smoothing_window_frames: 9      # how much smoothing (must be odd)
  smoothing_polyorder: 2
  remove_outlier_jumps: true      # flag impossible jumps
  max_speed_px_s: null            # null = auto-detect jumps; or a hard speed limit

regions:                          # dividing the container into zones
  border_margin_fraction: 0.10    # outer 10% counts as "border"
  center_roi_fraction: 0.50       # central 50% box counts as "center"
  custom_rois_path: null
  dual_container_split_x_fraction: null  # for left/right two-species videos

heatmaps:
  bins_x: 128                     # heatmap grid width
  bins_y: 72                      # heatmap grid height
  normalize: probability          # "probability", "count", or "density"
  per_track: false
  per_class: true                 # make separate heatmaps per species
  per_time_window_s: null

features:                         # thresholds that define behaviour states
  compute_pairwise: true
  pairwise_max_tracks_per_frame: 300
  nearest_neighbor_k: 1           # distance to the 1st nearest neighbour
  activity_speed_threshold_px_s: 5.0   # faster than this = "active"
  freezing_speed_threshold_px_s: 1.0   # slower than this = "freezing"
  burst_speed_quantile: 0.90      # top 10% of speeds = "burst"
  export_population_timeseries: true  # write population_timeseries.parquet (per-frame metrics)

reports:
  make_html: true                 # write report.html
  make_markdown: true             # write report.md
  make_pdf: true                  # also write report.pdf alongside report.html (needs the [pdf] extra)
  save_plots: true
  export_mot: true                # write tracks_mot.txt (MOTChallenge) + seqinfo.ini
  mot_include_interpolated: true  # include interpolated rows in the MOT export
  export_trajectories_geojson: true  # write trajectories.geojson (one feature per track)

fair:                             # see section 13
  enabled: true
  hash_inputs: true               # checksum the input video/model (slow for huge videos)
  license: "https://creativecommons.org/licenses/by/4.0/"
  license_name: "CC-BY-4.0"
  creator_name: null              # ← fill in your name to credit yourself
  creator_orcid: null             # ← e.g. "https://orcid.org/0000-0000-0000-0000"
  affiliation: null
  publisher: null
  project_url: null
  dataset_keywords: [mosquito larvae, MOSMON, object tracking, behaviour analysis, YOLO11, trajectory]
```

**The settings you will change most often:**

- `model.max_det` — how many detections are kept per frame. Keep it high (3000): dense
  videos produce ~800 candidate boxes per frame, and Ultralytics' own default of 300 silently
  drops the rest. `configs/eval.yaml` pins 300 only to reproduce the published evaluation.
- `model.agnostic_nms` — set **true** for this dataset; it removes duplicate boxes where the same larva is detected as two species at once (big quality win, see section 10).
- `model.conf` — lower it if larvae are missed; raise it if you get false detections.
- `model.imgsz` — raise toward native resolution for tiny larvae (uses more GPU memory).
- `tracker.type` — try `botsort` if the camera/water moves a lot (handheld GoPro footage).
- `video.frame_stride` / `video.max_frames` — make quick test runs.
- `video.save_debug_video` — set **true** to get a short annotated MP4 to eyeball the tracking.
- `video.save_tracking_video` (or `--save-tracking-video`) — set **true** for the full-length, full-resolution annotated video.
- `features.*_threshold_px_s` — define what "active" / "freezing" mean for your larvae.
- `calibration.pixels_per_mm` — to get millimetres instead of pixels.

> **Ready-made evaluation configs.** `configs/eval.yaml` is a tuned profile
> (BoT-SORT + agnostic-NMS + iou 0.7 + stride 2 + debug video on). For a long
> batch on tight disk, `configs/eval_lean.yaml` uses the same detector/tracker but
> drops the frame-heavy *extra* outputs (debug video, MOT, trajectory GeoJSON) and
> input hashing — ~31 GB output instead of ~43 GB across the 66-video set, same
> core tables/heatmaps/reports. Any dropped output can be regenerated later from
> `tracks_clean.parquet` via `analyse-tracks`.

---

## 9. The output files, explained

Each run folder (from `track-video` / `analyse-tracks`) contains:

### Tables (the data)

| File | What's inside |
| --- | --- |
| `tracks_raw.parquet` | Every tracked larva box in every frame, with its track ID — the raw tracking result. Note these are the tracker's *output* boxes, not the detector's: a detection the tracker declined to associate never appears. For true detector output, build a detection cache with `mosmon detect-videos`. |
| `tracks_clean.parquet` | Cleaned tracks: gaps filled/flagged, smoothed, with speed/acceleration/heading/turn, region labels, neighbour distances, and movement state. |
| `track_summary.parquet` | **One row per larva (track)** — its duration, path length, average speed, tortuosity, quality score, etc. This is usually the table you analyse. |
| `population_timeseries.parquet` | **One row per frame** — active-larvae count, mean/median/max speed, mean acceleration, mean nearest-neighbour distance, density, and the fraction of larvae active/freezing/bursting and in border/center. This is the data behind the "…over time" plots; re-plot it or correlate it with metadata conditions without re-running. Controlled by `features.export_population_timeseries` (on by default). |
| `trajectories.geojson` | **One feature per larva (track)** — a GeoJSON `LineString` of its centre path over time (a `Point` for single-frame tracks), with the track's summary stats attached as `properties`. Drops straight into GIS/plotting tools (QGIS, kepler.gl, geopandas). Coordinates are image pixels (`y` points down) unless calibration is on, then millimetres. Controlled by `reports.export_trajectories_geojson` (on by default). |
| `tracks_mot.txt` | Tracks in **MOTChallenge format** (`frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1`; frames 1-based) for interoperable evaluation with TrackEval / py-motmetrics (MOTA, IDF1, HOTA) and other MOT tooling. A `seqinfo.ini` is written alongside so the folder drops straight into a TrackEval sequence. Controlled by `reports.export_mot` (on by default) and `reports.mot_include_interpolated`. |

> Tables are **Parquet** by default (compact, fast). If Parquet isn't available the
> tool automatically writes **CSV** instead (opens in Excel).

**Key columns in `tracks_clean`:**

| Column | Meaning |
| --- | --- |
| `frame_idx`, `time_s` | Frame number and time in seconds. |
| `track_id` | Which larva. |
| `cx`, `cy` | Centre of the larva (pixels). `cx_norm`/`cy_norm` are 0–1. |
| `speed_px_s` | Speed in pixels per second. |
| `acceleration_px_s2` | Acceleration. |
| `heading_rad`, `turn_angle_rad` | Direction of travel and how sharply it turned. |
| `roi_label` | `border`, `center`, or `intermediate`. |
| `nearest_neighbor_distance_px` | Distance to the closest other larva. |
| `movement_state` | `freezing`, `slow`, `active`, `burst`, or `unknown`. |
| `is_interpolated` | This row was filled in across a gap (not a real detection). |
| `jump_flag` | This step looked like an impossible jump (treat with caution). |

**Key columns in `track_summary`:**

| Column | Meaning |
| --- | --- |
| `duration_s`, `n_frames` | How long the larva was tracked. |
| `path_length_px` | Total distance travelled. |
| `net_displacement_px` | Straight-line distance from start to end. |
| `tortuosity` | Path length ÷ net displacement (1 = straight line; higher = wandering). |
| `straightness` | The inverse idea (1 = perfectly straight). |
| `mean_speed_px_s`, `max_speed_px_s` | Speeds. |
| `radius_of_gyration_px` | How spread out the path was. |
| `border_fraction`, `center_fraction` | Fraction of time spent in each zone. |
| `freezing_fraction`, `burst_fraction` | Fraction of time frozen / bursting. |
| `quality_score` | **0–1 score of how trustworthy this track is** (length, confidence, gaps, jumps). Use it to filter — e.g. keep only tracks with `quality_score > 0.5`. Not a biological number. |

### Summaries and quality

| File | What's inside |
| --- | --- |
| `video_summary.json` | One-page summary of the whole video: number of larvae, average activity, occupancy, etc., plus the settings and metadata used. |
| `qc_report.json` | Quality control: % of frames with detections, track-length distribution, gap frequency, interpolation fraction, confidence distribution, duplicate-track warning, and a list of **warnings** if something looks off. |

### Pictures

| Folder | Contents |
| --- | --- |
| `plots/` | `trajectories.png`, `speed_over_time.png`, `active_over_time.png`, `track_duration_hist.png`, `speed_distribution.png`, `turn_angle_distribution.png`, `nn_distance_distribution.png`, `qc_dashboard.png`. |
| `heatmaps/` | For occupancy, dwell-time, speed, activity (and per species): a `.png` to look at, a `.npy` (raw numbers) and a `.parquet`/`.csv` (table) for further analysis. |

### Annotated video (optional)

| File | What's inside |
| --- | --- |
| `debug_video.mp4` | A short annotated clip (boxes + track IDs + fading motion trails), written when `video.save_debug_video: true`. The fastest way to *see* whether tracking is sane. Downscaled and length-capped (see the `debug_video_*` config options). |
| `tracking_video.mp4` | The **full** annotated video — every processed frame, all tracks, full length, at the source FPS. Written when `video.save_tracking_video: true` or via the `--save-tracking-video` flag on `track-video` / `batch-track`. Full resolution by default (`tracking_video_scale: 1.0`); expect large files and slow encoding on high-res sources — lower `tracking_video_scale` to trade quality for size. |
| `heatmap_video.mp4` | The occupancy heatmap **accumulating over time**, coloured (`inferno`) and overlaid on a representative frame — shows where larvae have been, building up. Written when `video.save_heatmap_video: true`. Downscaled (`heatmap_video_scale: 0.5`) and output-frame-capped (`heatmap_video_max_frames`) by default; the histogram still accumulates every frame regardless of the cap. |

### The report

| File | What's inside |
| --- | --- |
| `report.md` | A readable summary in **Markdown** — open on GitHub or in a Markdown viewer to see formatted tables and embedded images. |
| `report.html` | A **standalone web page**: double-click to open in a browser. Real HTML tables (video info, settings, QC, behaviour) plus an image gallery embedding every plot and heatmap (click to enlarge), QC warnings in red, and the interpretation caveats. |
| `report.pdf` | A **PDF that matches `report.html`** (same tables and figure gallery, laid out for A4). Written on every run (`reports.make_pdf: true` by default); needs the optional extra `pip install '.[pdf]'` (WeasyPrint + pypdf) — if that isn't installed the run still succeeds and notes a QC warning. You can also generate it after a run with `python scripts/report_to_pdf.py --run-dir <folder>`. |

### FAIR metadata (see section 13)

`ro-crate-metadata.json`, `provenance.json`, `data_dictionary.json`,
`checksums.sha256`, `LICENSE.txt`.

> **Where results actually go.** Each run is written to its **own folder**
> (`outputs/<run>/` for `track-video`, `outputs/<batch>/<model>__<video>/` for
> `batch-track`). The top-level `outputs/detections/`, `outputs/plots/`, etc.
> are **empty placeholders** (they contain only a hidden `.gitkeep`) — your data
> is in the per-run folders, not there.

---

## 10. Tuning the tracker

If the results are not good, here is what to change, in order:

1. **Same larva detected as two species / duplicate overlapping boxes.** Set
   `model.agnostic_nms: true`. On the test data this dropped the duplicate-box rate
   from 0.74 to ~0.02 — the single most effective setting for this model. (Residual
   overlap in very dense containers is *real crowding*, not a duplicate.)
2. **Larvae are missed (not detected).** Lower `model.conf` (e.g. 0.15 → 0.08) and/or
   raise `model.imgsz` (e.g. 1920 → 2560) so small larvae are bigger to the model.
3. **Too many false detections (junk boxes).** Raise `model.conf`.
4. **IDs keep switching / tracks break into pieces.** Increase `tracker.max_gap_frames`
   and the tracker's `track_buffer` (in `configs/tracker_botsort.yaml`). Use
   `tracker.type: botsort`, which compensates for camera/water motion (best for GoPro).
5. **Lots of short useless tracks.** Raise `tracker.min_track_length_frames`.
6. **Impossible jumps in trajectories.** Set `postprocess.max_speed_px_s` to a sensible
   limit, or leave `null` to auto-flag them.
7. **Want a quick test first.** Set `video.max_frames: 300` and `video.frame_stride: 2`
   to process just the start, quickly. Turn on `video.save_debug_video` and watch the
   resulting `debug_video.mp4`.

After any change that is *analysis only* (thresholds, smoothing, heatmaps,
calibration), use `analyse-tracks` instead of re-running `track-video` — it's much
faster because it reuses the existing tracks.

---

## 11. Calibration: turning pixels into millimetres

By default everything is in **pixels** and **pixels per second**. This is honest but
not biological. To get millimetres, turn on calibration in your config.

**Easiest method — a single scale factor.** If you know how many pixels equal one
millimetre (e.g. measure a ruler in the video):

```yaml
calibration:
  enabled: true
  pixels_per_mm: 12.5     # ← your measured value
```

Now speeds/distances also get `_mm` columns.

**Reference dimensions** (from the dataset, to help you estimate):
standard container ≈ 38 × 16 cm; large container ≈ 52 × 17 cm; water depth ≈ 4.5 cm.
If you know the container width in cm and can measure it in pixels, pixels-per-mm =
(width in pixels) ÷ (width in cm × 10).

**Advanced — perspective correction (homography).** If the camera was at an angle,
provide a 3×3 matrix file:

```yaml
calibration:
  enabled: true
  homography_path: data/metadata/homography.json   # {"homography": [[...],[...],[...]]}
```

If calibration is **off**, the tool clearly labels everything as pixels and never
pretends otherwise.

---

## 12. High-resolution videos (8K) and tiling

Some videos are very high resolution (8K, 5.7K). By default the model resizes each
frame down to `model.imgsz` (default 1920) before detecting. For huge frames this
shrinks the already-tiny larvae and can hurt detection.

Two options:

- **Full-frame (default):** raise `model.imgsz` (e.g. 2560 or 3200) for better detection
  at the cost of GPU memory and speed.
- **Tiled inference:** split each frame into overlapping tiles, detect in each at 1:1
  pixel resolution, remap the boxes back to the full frame, de-duplicate the seams, then
  track. This preserves the microscopic detail the detector was trained on and recovers
  small-larva recall, at the cost of running the detector many times per frame. It is
  the runtime counterpart of the training-time tiling in `training/` (see section 17.1).

Turn on tiled inference in the config:

```yaml
model:
  tiling:
    enabled: true      # off by default
    tile_size: 640     # match the tile size the model was trained on
    overlap: 0.10      # overlap between adjacent tiles (avoids splitting a larva at a seam)
    merge_iou: null    # NMS IoU for de-duplicating seam boxes (null = use model.iou)
    batch: 8           # tiles per detector forward pass (lower this if you hit OOM)
    min_area_frac: 0.0 # drop detections smaller than this fraction of a tile (0 = keep all)
```

**When to use which.** Tiling helps most on native-resolution, densely-clustered, small
larvae — the report behind `training/` measured roughly +14% strict localization
(mAP50-95) from pixel-preserving tiling versus a single downscale. The trade-off is
speed: full-frame runs one forward pass per frame; tiled runs one per *tile* (dozens on a
5.7K frame). For a fast pass, prefer full-frame with a higher `imgsz`; for the most
faithful counts on high-res clips, enable tiling. The output tables are identical either
way, so every downstream step (features, heatmaps, reports) is unchanged.

On an 8 GB GPU, start full-frame with `imgsz: 1920` or `2560`; if you hit out-of-memory
errors, lower `imgsz`/`tiling.batch` or set `video.frame_stride: 2`.

---

## 13. FAIR outputs

Every result folder is packaged so the data is **F**indable, **A**ccessible,
**I**nteroperable, and **R**eusable — ready to share or publish.

| File | What it is |
| --- | --- |
| `ro-crate-metadata.json` | A standard **RO-Crate** description of the dataset: a unique ID, the license, keywords, the species (linked to NCBI Taxonomy), and a list of every file with its size, format, and checksum. You can upload the whole folder to **Zenodo** to get a citable **DOI**. |
| `provenance.json` | Exactly how the results were made: software version, all package versions, the full settings used, the command you ran, the input video/model checksums, and a timestamp. This is what makes results reproducible. |
| `data_dictionary.json` | What every column in every table means, **with units** (UCUM where applicable; pixels are flagged as non-physical). |
| `checksums.sha256` | A fingerprint of every file, to prove nothing was altered. |
| `LICENSE.txt` | The license (default **CC-BY-4.0**: others may reuse with credit). |

**To credit yourself**, fill in the `fair:` block in your config:

```yaml
fair:
  creator_name: "Your Name"
  creator_orcid: "https://orcid.org/0000-0000-0000-0000"
  affiliation: "Your institute"
```

These are embedded as author/creator metadata in every RO-Crate.

---

## 14. How filenames are read

The tool extracts metadata directly from video filenames, which look like:

```
20251009_alboSX_culexDX_stage3and4_djiosmoaction5_video_4k30fps_rocksteadywide_lightnatartcold_cp4_big2boxcontainer_depth45mm_larvaeadded.MP4
```

From that one name it reads: **date** (2025-10-09), **two species** in a split
container (`alboSX` = *Aedes albopictus* on the left, `culexDX` = *Culex pipiens* on
the right), **stage** (3–4), **camera** (DJI Osmo Action 5), **resolution/fps**
(4K, 30 fps), **stabilisation/lens**, **lighting** (natural + artificial cold),
**camera position** (4), **container** (big two-box), **water depth** (45 mm), and
**modifiers** (larvae added).

It is robust to typos, abbreviations, and missing pieces — anything it cannot read
is recorded as a warning rather than crashing. Check `inspect-videos` output to see
what was understood for your files.

The four species and their NCBI Taxonomy IDs:

| Species | NCBI taxon |
| --- | --- |
| *Aedes aegypti* | txid7159 |
| *Aedes albopictus* | txid7160 |
| *Anopheles stephensi* | txid30069 |
| *Culex pipiens* | txid7175 |

---

## 15. How to read the results responsibly

These caveats are included in every report — please respect them:

- The tool tracks the **centre of the bounding box**, not the larva's body posture or
  head/tail direction.
- **Heading/turning** numbers are unreliable when a larva is nearly still (it jitters).
- Apparent motion can be caused by **camera movement, water ripples, turbidity,
  reflections, and lens distortion**, not just the larva.
- Without **calibration**, speeds and distances are in **pixels**, not millimetres.
- Track IDs can **fragment** under crowding/occlusion. Filter by `quality_score` and
  avoid over-interpreting very short tracks.
- In **two-species (left/right) containers**, species come from the filename's SX/DX
  labels — not from looking at the larva.
- If you use `frame_stride > 1`, you skip frames; the tool uses the true FPS so times
  stay correct, but very fast events may be missed.

A good habit: always open `qc_report.json` (or the QC dashboard plot) first and read
its `warnings` before trusting the behaviour numbers.

---

## 16. Troubleshooting

| Problem | Likely cause / fix |
| --- | --- |
| `mosmon: command not found` | You forgot `conda activate mosmon_tracker`, or didn't run `pip install -e .`. |
| `GPU available: False` | PyTorch CPU-only installed. Reinstall torch with the CUDA index URL (step 3). |
| `CUDA out of memory` | Lower `model.imgsz`, set `video.frame_stride: 2`, or set `model.half: true`. |
| Very few or no detections | Lower `model.conf`; raise `model.imgsz`; check you used the right `.pt` model. |
| Lots of broken/short tracks | Increase `tracker.max_gap_frames` and `track_buffer`; try `botsort`. |
| `no fps` warning / weird speeds | The video's FPS couldn't be read; install `ffmpeg`/`ffprobe`, or re-export the video. |
| Output tables are `.csv` not `.parquet` | `pyarrow` isn't installed; CSV works fine, or `pip install pyarrow`. |
| A video failed during `batch-track` | Read its `failure.json`; the batch continued with the others. |
| Plots look empty | The video produced too few/no tracks; check QC and detection settings. |

To see ffprobe (better video info), install ffmpeg: `conda install -c conda-forge ffmpeg`.

---

## 17. For developers

```bash
conda activate mosmon_tracker
pip install -e ".[dev]"     # adds pytest, ruff, mypy
pytest                      # run the test suite (uses tiny synthetic data, no GPU/weights needed)
ruff check src              # lint
```

Project layout:

```
configs/                 settings (YAML)
src/mosmon_tracking/      the code
  cli.py                  the `mosmon` command-line interface
  config.py               loads/validates the YAML config
  metadata.py             reads metadata from filenames
  video_io.py             probes videos, reads/writes tables
  yolo_tracker.py         runs YOLO11 + tracker (the only GPU part; full-frame + tiled backends)
  tiling.py               tiled-inference geometry (windows, box remap, seam NMS)
  track_postprocess.py    cleans tracks, computes per-frame kinematics
  features.py             track- and video-level behaviour measurements
  heatmaps.py             spatial heatmaps
  visualization.py        plots
  quality_control.py      QC metrics
  reports.py              JSON/Markdown/HTML reports
  fair.py                 FAIR metadata (RO-Crate, provenance, checksums)
  batch.py                ties it all together (single run, batch, compare)
tests/                   unit tests
outputs/                 results (created as you run)
```

The behaviour analysis is fully tested **without** needing the AI model or a GPU —
only the `yolo_tracker.py` step needs the `.pt` weights and a video.

### 17.1 Model training (upstream) — `training/`

This repo also includes the **upstream** half of the project: the dataset-preparation and
YOLO11 training pipeline that *produces* the detection weights the tracker consumes.

```
training/
  tiling/     build tiled datasets (naive grid, overlapping, smart-ROI, oversampled)
  train/      split_n_verify.py (temporal split with buffer zones), train.py (one mode),
              train_auto.py (all modes), test.py (inference / validation)
  compare/    tabulate and plot metrics across trained modes
  configs/    one Ultralytics dataset YAML per mode — edit `path:` to your dataset root
```

It covers ten "Modes" (0-9) of pixel-preserving tiling, class balancing, and loss
corrections; the released weight `mosmon_yolov11x_overlap_tiling_best_v1.pt` comes from
the overlapping-tiling lineage. The **tiled-inference** backend described in section 12 is
the runtime mirror of that training-time tiling (same stride math and box remapping).

```bash
python training/train/train.py --mode 6 --help
```

You only need `training/` if you want to retrain or reproduce the weights; day-to-day
tracking uses the pre-trained `.pt` and never touches it.

---

## 18. Manuscript figures

`scripts/figures/` regenerates the MOSMON-Track manuscript figures (3-7) from the stored
pipeline outputs. Every plotted value is traced back to a file on disk, every
selected video, image and frame is recorded, and the scripts refuse to invent a missing
input.

### Running them

Point `paths.mosmon_outputs` in `configs/figures.yaml` at your tracking output tree
(everything else defaults to the standard layout underneath it), then, from the
repository root:

```bash
make figures                                          # all of them
python scripts/figures/fig03_tracking_crowding.py     # or one at a time
python scripts/figures/fig04_population_analysis.py
python scripts/figures/fig05_dual_container.py
python scripts/figures/fig06_tracker_benchmark.py
python scripts/figures/fig07_tracker_downstream.py
```

Each script writes, into `figures/generated/`:

| File | Contents |
|---|---|
| `<figure>.pdf` | The figure, with **vector** (selectable) text for print |
| `<figure>.png` | 300 dpi raster preview |
| `<figure>_data.csv` | Exactly the values plotted, one tidy row per point |
| `<figure>_metadata.json` | Provenance sidecar (see below) |
| `panels/<figure>_<p>.pdf/.png` | Each panel exported standalone, plus supplementary panels |

### What each figure needs

| Figure | Inputs |
|---|---|
| **3** — tracking quality across crowding | `full_eval_summary_table.csv`, the 66 per-run `video_summary.json` and `population_timeseries.parquet`, and the source videos (panel **a** seeks real frames) |
| **4** — population-level trajectory information | `species_analysis/{classification_results.json, video_features.parquet, univariate_tests.parquet}` |
| **5** — controlled dual-container analysis | `species_analysis/dual_container_contrast.parquet`, `dual_reanalysis/{splits.json, <run>/}`, and one source video |
| **6** — multi-tracker benchmark | `tracker_benchmark/evaluation/{metrics_per_video.parquet, summary_by_arm.parquet}` (see §19) |
| **7** — downstream effect of tracker choice | `tracker_benchmark/downstream/{descriptor_stability.parquet, classification_by_arm.parquet, cohort_summary.parquet}` |

Figure 3 reads several hundred MB and seeks three videos, so it takes a
couple of minutes. Figure 4 regenerates 200 permutation draws for each of three
validation schemes (about 100 s). Figure 5 is fast.

### The provenance sidecar

`<figure>_metadata.json` records every input path with its size and sha256 (source
videos carry the hash from the run's `provenance.json` rather than being re-read), the
selected run/video/frame identifiers, the plotting parameters, the library versions and
the git commit — and a `validation` block comparing every manuscript value with the value
recomputed from source data.

**A disagreement never gets papered over.** The script prints it, records it, and plots
the *computed* value. Three are known and expected:

- Figure 3 finds **nine** recordings below 97 % detection coverage, not the eight stated
  in the manuscript; the extra one is at 96.84 %.
- Figures 4 and 5 rank features by effect size and show the true top *N*, which includes
  one or two features absent from the manuscript's hand-picked tables.
- Figure 3 panel (b) plots all **66 processed runs**, the universe that reproduces
  `r = 0.601`; deduplicating to the 63 unique source videos gives `r = 0.571`, and both
  are recorded. Duplicated inputs are drawn as open markers.

Two quantities the pipeline summarises but does not persist — the per-video
leave-one-video-out predictions and the individual permutation draws — are regenerated in
Figure 4 from `video_features.parquet` with the analysis's own seed, then checked against
the stored aggregates before anything is drawn.

### Reading these figures responsibly

The panels show **tracking diagnostics, not ground-truth MOT metrics**. In particular
`duplicate_track_proxy` is not IDF1, HOTA, fragmentation or an identity-switch rate;
"frame detection coverage" is the fraction of processed frames holding at least one
detection, not instance-level recall; and "mean active tracks per frame" is
tracker-derived, not a true larval count. Species labels come from acquisition metadata,
never from detector class predictions. Movement variables are body-length normalised
(apparent body lengths), not physical units. See section 15.

### Tests

```bash
make test-figures        # or: pytest tests/test_figures.py -q
```

These cover the pure helpers — species naming, deterministic selection, the
manuscript-value checker, the dual-container sign convention and the export contract — and
need neither model weights nor the outputs volume.

---

*Outputs are licensed CC-BY-4.0 by default (configurable). When you publish, fill in
the `fair:` creator fields so the datasets are properly attributed to you.*

---

## 19. Multi-tracker benchmark (fixed detector)

Compares association algorithms with the **detector held fixed**: one detector pass per
video, replayed by every tracker. Because the detections are byte-identical across arms,
detection quality is not merely controlled — it is *invariant*, so every difference
between arms is association and nothing else.

### Read this before interpreting anything

There is **no manually annotated identity ground truth**, so the benchmark reports
**no HOTA, DetA, AssA, IDF1 or ID-switch counts**. Nothing in it says which tracker is
*correct*. What it does measure:

- **Output statistics** — track duration, fragments per object, duplicate proxies.
  A tracker that merges two larvae into one identity scores *well* on all of these.
- **Cross-arm identity agreement** — how differently two arms group the same
  observations. Two arms can agree perfectly and both be wrong.

So "which tracker is best" is not answerable here. "How much do the arms differ, where,
and does the difference reach the biology?" is.

### The arms

| Arm | Mechanism | Needs video frames? |
|---|---|---|
| `bytetrack_iou` | ByteTrack with its low-confidence stage switched off — single-stage Kalman + IoU. The motion-only baseline. **Not SORT**: it is Ultralytics' implementation and Kalman model, so do not report it as Bewley et al. | no |
| `bytetrack` | Two-stage, recovers low-confidence detections | no |
| `ocsort` | Observation-centric motion, `use_byte: false` | no |
| `botsort` | Hybrid + `sparseOptFlow` global motion compensation. **The published MOSMON reference.** | yes |
| `deepocsort` | Motion + a generic appearance (ReID) embedding | yes |

The frame-free arms replay from the detection table alone with no video decode, which is
roughly two orders of magnitude cheaper. All arms run from a **single** decode pass.

`deepocsort` needs an explicit ReID encoder. `model: auto` silently does nothing when
replaying a detection table — it reuses detector backbone features that only exist inside
Ultralytics' predict loop — so the adapter refuses it rather than reporting a non-appearance
arm as an appearance arm. The encoder is generic and off-the-shelf; no larva-specific ReID
is trained.

### Running it

```bash
# 1. Detect once (the expensive stage; conf 0.05, max_det 3000) and replay every arm.
python scripts/benchmark/run_all_trackers.py \
  --videos data/raw_videos \
  --model  data/models/mosmon_yolov11x_overlap_tiling_best_v1.pt \
  --out    outputs/tracker_benchmark \
  --skip-existing

# 2. Roll up tables, re-run the species analysis per arm, write the report.
python scripts/benchmark/evaluate_all_trackers.py \
  --benchmark outputs/tracker_benchmark \
  --downstream

# 3. Figures.
make fig06 fig07
```

Both stages resume with `--skip-existing`. The individual CLI commands
(`mosmon detect-videos`, `mosmon benchmark-track`, `mosmon benchmark-validate`) are also
available if you want to run a stage on its own.

### Why the detection cache is re-built rather than reused

The per-run `detections.parquet` written by earlier versions of the pipeline was **a
byte-identical copy of `tracks_raw.parquet`** — BoT-SORT's *output*, not the detector's.
Its minimum confidence is 0.2502, exactly BoT-SORT's `track_high_thresh`: the whole
0.15–0.25 detector band is absent and unassociated boxes were never written. Reusing it
would bias the comparison toward BoT-SORT and leave ByteTrack's low-confidence stage
nothing to associate. That duplicate file is no longer written.

The cache is built at `conf 0.05` so each tracker applies its own internal thresholds.
Filtering it at a higher confidence reproduces a native pass at that confidence *exactly* —
NMS is greedy over scores, so a low-scoring box can never suppress a higher-scoring one.
Verified on a real video: 6890 boxes, identical coordinates. The comparison must match
Ultralytics' semantics (strict `>` against an **fp16** threshold), which
`benchmark.detections_io.apply_conf_floor` does.

`max_det` is raised to 3000. Ultralytics defaults to **300**, and earlier versions of this pipeline never passed it to the tracker, so
an earlier full evaluation truncated any frame with more than 300 candidate detections to
the top 300 by score — precisely in the dense regime a crowding analysis is about.
`check_detection_table` probes for the signature (a spike at exactly `max_det`).

### Validating an arm against a previous run

```bash
mosmon benchmark-validate \
  --detections outputs/tracker_benchmark/detections_canonical \
  --video      data/raw_videos/<video>.MP4 \
  --reference  outputs/eval_batch/<run>/tracks_raw.parquet \
  --arm botsort
```

Boxes are matched per frame by nearest neighbour within 1 px, not by exact equality: the
trackers emit Kalman-filtered states rather than the detections they consumed (~0.8 px
from the input box), so two runs that made every identical decision still differ in the
last bits once global motion compensation reads independently decoded pixels. The residual
distribution is reported alongside the match rate, so a real divergence cannot hide behind
the tolerance.

Measured on the 5.3K albopictus reference video: **99.7 % of boxes matched, ARI 0.998,
identity-F1 0.996, residual 0.089 px median / 0.99 px max.** `exact_id_match` is lower
(0.91) only because track *numbering* depends on allocation order — which is why ARI and
identity-F1, not raw ID equality, are the measures used.

### Outputs

```
tracker_benchmark/
  detections_canonical/<video>.parquet     the fixed detector output + .meta.json sidecar
  <arm>/<video>/                           a normal run folder: tracks_raw/clean, summaries, QC
                 assignments.parquet       shared detection row -> this arm's track id
                 tracks_native.parquet     Experiment A: minimal filtering, no smoothing
                 arm_metadata.json         resolved parameters, provenance, timing, checks
  _agreement/<video>.parquet               pairwise cross-arm identity agreement
  _manifests/<video>.json                  per-video fairness + validity checks
  evaluation/                              per-video metrics, per-arm summaries, trends
  downstream/                              cohorts, descriptor stability, per-arm classification
  reports/TRACKER_BENCHMARK_REPORT.md      written by evaluate_all_trackers.py
```

Two reconstruction variants are written per arm, and must never be mixed:
**Experiment A** (`tracks_native.parquet`) is the arm's own output with only the minimum
filtering descriptors require; **Experiment B** (`tracks_clean.parquet`) is the standard
MOSMON cleaning applied identically to every arm. A is what the tracker produces; B asks
whether the reconstruction pipeline normalises the differences away.

### The duplicate-track proxy

`quality_control` now reports two:

- `duplicate_track_proxy` — the published metric. Despite its docstring it tests **box
  geometry only** and never required the overlapping boxes to carry different track IDs,
  so it also fires on a correctly tracked crowded pair. Left unchanged, because it gates
  the published 45-video species cohort.
- `duplicate_id_proxy` — the identity-aware version, which is the one to use when asking
  whether *association* duplicated an individual.

The benchmark reports both and correlates them against fragmentation, crowding and
cross-arm disagreement. That is a **coherence** check, not the validation the metric
really needs: with no ground truth there is no measured association error to correlate
against, so whether the proxy predicts real identity failure remains open.

---

## 20. License and acknowledgements

The code in this repository is released under the **Apache License 2.0** — see
[`LICENSE`](LICENSE). Outputs generated by the pipeline carry the license set in the
`fair:` section of the config (CC-BY-4.0 by default).

The training pipeline in `training/` was originally written by **Francesca Sanasi** and
adapted for this repository.

