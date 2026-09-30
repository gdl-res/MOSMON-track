"""Shared style, provenance and export machinery for the manuscript figures.

Every figure script in this directory imports from here so that typography,
species naming/ordering, panel labelling and the export contract stay identical
across figures. Nothing in this module reads pipeline data: it only knows how a
figure should look and what must be recorded about it.

The export contract for one figure ``<name>`` is:

    figures/generated/<name>.pdf              vector, selectable text
    figures/generated/<name>.png              300 dpi raster preview
    figures/generated/<name>_metadata.json    provenance sidecar
    figures/generated/<name>_data.csv         exactly the values plotted
    figures/generated/panels/<name>_<p>.pdf   per-panel exports (+ .png)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "figures.yaml"


# --------------------------------------------------------------------------- #
# 1. Species naming — one order, one set of abbreviations, everywhere
# --------------------------------------------------------------------------- #
SPECIES_ORDER: list[str] = [
    "Ae. aegypti",
    "Ae. albopictus",
    "An. stephensi",
    "Cx. pipiens",
]

# The pipeline stores full binomials (``meta_species``); the manuscript uses the
# abbreviations above. Keys are lowercased so any capitalisation variant maps.
_SPECIES_ALIASES: dict[str, str] = {
    "aedes aegypti": "Ae. aegypti",
    "aedes albopictus": "Ae. albopictus",
    "anopheles stephensi": "An. stephensi",
    "culex pipiens": "Cx. pipiens",
    "aedes_aegypti": "Ae. aegypti",
    "aedes_albopictus": "Ae. albopictus",
    "anopheles_stephensi": "An. stephensi",
    "culex_pipiens": "Cx. pipiens",
    "ae. aegypti": "Ae. aegypti",
    "ae. albopictus": "Ae. albopictus",
    "an. stephensi": "An. stephensi",
    "cx. pipiens": "Cx. pipiens",
}


def species_label(name: Any) -> str:
    """Canonical manuscript abbreviation for a species name.

    Unknown values are returned stripped rather than silently dropped, so a
    label the pipeline produced but this module does not know about shows up in
    the figure instead of disappearing.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "unknown"
    key = str(name).strip().lower()
    return _SPECIES_ALIASES.get(key, str(name).strip())


def species_sort_key(name: Any) -> tuple[int, str]:
    """Sort key placing the four known species in manuscript order, others last."""
    label = species_label(name)
    if label in SPECIES_ORDER:
        return (SPECIES_ORDER.index(label), label)
    return (len(SPECIES_ORDER), label)


def order_species(values) -> list[str]:
    """Unique canonical species labels from ``values``, in manuscript order."""
    seen = {species_label(v) for v in values}
    return sorted(seen, key=species_sort_key)


# Colour *and* marker differ per species: the figures must stay readable when
# the journal prints them in grayscale, so colour never carries meaning alone.
_SPECIES_STYLE: dict[str, dict[str, Any]] = {
    "Ae. aegypti":    {"color": "#1f6fb4", "marker": "o", "hatch": ""},
    "Ae. albopictus": {"color": "#d1701c", "marker": "s", "hatch": "//"},
    "An. stephensi":  {"color": "#2e7d4f", "marker": "^", "hatch": ".."},
    "Cx. pipiens":    {"color": "#7b4ea3", "marker": "D", "hatch": "xx"},
}
_UNKNOWN_STYLE = {"color": "#7a7a7a", "marker": "v", "hatch": "\\\\"}


def species_style(name: Any) -> dict[str, Any]:
    """Colour / marker / hatch for a species, consistent across all figures."""
    return dict(_SPECIES_STYLE.get(species_label(name), _UNKNOWN_STYLE))


# --------------------------------------------------------------------------- #
# 1b. Tracker naming and palette - the benchmark figures (fig06, fig07)
# --------------------------------------------------------------------------- #
# Arms are ordered by how much machinery their association uses, so a reader
# moves left-to-right from pure geometry to appearance. Colour follows the arm,
# never its rank, so a figure that drops an arm never repaints the survivors.
TRACKER_ORDER: list[str] = [
    "bytetrack_iou", "bytetrack", "ocsort", "botsort", "deepocsort",
]

TRACKER_LABELS: dict[str, str] = {
    # "ByteTrack (1-stage)" rather than "SORT": this is Ultralytics' ByteTrack
    # with its low-confidence stage switched off, not the Bewley implementation,
    # and labelling it SORT would misattribute the Kalman model.
    "bytetrack_iou": "ByteTrack (1-stage)",
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "botsort": "BoT-SORT",
    "deepocsort": "Deep OC-SORT",
}

# Five hues drawn from the reference categorical ramp and validated as a set
# under the all-pairs pairlist (these are scatter/dot forms, so every pair must
# separate, not just adjacent ones): worst CVD Delta E 13.0, worst normal-vision
# Delta E 16.3, both above their floors. Yellow and magenta fall below the
# contrast target against a white surface, which obligates the visible labels,
# legend and exported data table these figures already carry; a thin ink edge on
# every marker keeps them legible in print without changing their identity.
_TRACKER_STYLE: dict[str, dict[str, Any]] = {
    "bytetrack_iou": {"color": "#2a78d6", "marker": "o"},
    "bytetrack":     {"color": "#eda100", "marker": "s"},
    "ocsort":        {"color": "#e87ba4", "marker": "^"},
    "botsort":       {"color": "#008300", "marker": "D"},
    "deepocsort":    {"color": "#4a3aa7", "marker": "v"},
}
_TRACKER_FALLBACK = {"color": "#5c5c5c", "marker": "P"}

#: Edge drawn on every tracker marker. Ink, not the series colour - text and
#: strokes wear text tokens so identity stays with the fill and the shape.
TRACKER_MARKER_EDGE = "#1a1a1a"


def tracker_label(name: Any) -> str:
    """Display name for an arm."""
    key = str(name).strip().lower()
    return TRACKER_LABELS.get(key, str(name))


def tracker_sort_key(name: Any) -> tuple[int, str]:
    key = str(name).strip().lower()
    return ((TRACKER_ORDER.index(key), "") if key in TRACKER_ORDER
            else (len(TRACKER_ORDER), key))


def order_trackers(values) -> list[str]:
    """Unique arm names in the canonical order, unknown ones alphabetically last."""
    return sorted({str(v) for v in values if v is not None and str(v) != "nan"},
                  key=tracker_sort_key)


def tracker_style(name: Any) -> dict[str, Any]:
    """Colour + marker for one arm. Identity is never colour alone."""
    key = str(name).strip().lower()
    style = dict(_TRACKER_STYLE.get(key, _TRACKER_FALLBACK))
    style["markeredgecolor"] = TRACKER_MARKER_EDGE
    style["markeredgewidth"] = 0.5
    return style


# Neutral ramp for non-species categories (validation schemes, QC variables).
NEUTRAL = {
    "ink": "#1a1a1a",
    "mid": "#5c5c5c",
    "light": "#9e9e9e",
    "faint": "#d8d8d8",
    "grid": "#e8e8e8",
    "accent": "#1f6fb4",
    "warn": "#b3341f",
}


# --------------------------------------------------------------------------- #
# 2. Style
# --------------------------------------------------------------------------- #
def apply_style() -> None:
    """Restrained IJCV-style defaults: white ground, thin rules, vector text.

    ``pdf.fonttype``/``ps.fonttype`` 42 embeds TrueType rather than rasterising
    or subsetting to Type 3, which is what keeps PDF text selectable and
    searchable — one of the figure checklist's requirements.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 120,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "axes.titleweight": "regular",
        "axes.facecolor": "white",
        "axes.edgecolor": NEUTRAL["mid"],
        "axes.linewidth": 0.6,
        "axes.labelcolor": NEUTRAL["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": NEUTRAL["grid"],
        "grid.linewidth": 0.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.color": NEUTRAL["ink"],
        "ytick.color": NEUTRAL["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.3,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "text.color": NEUTRAL["ink"],
        "image.cmap": "viridis",
    })


def panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.04,
                fontsize: float = 9.0) -> None:
    """Draw the ``(a)`` / ``(b)`` panel label in axes-fraction coordinates."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="left",
            color=NEUTRAL["ink"])


def hide_axes(ax) -> None:
    """Turn an Axes into a bare image canvas (used for the frame panels)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NEUTRAL["light"])
        spine.set_linewidth(0.6)


def thousands(ax, axis: str = "y") -> None:
    """Thousands separators, so large counts stay readable at print size."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# --------------------------------------------------------------------------- #
# 3. Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FigurePaths:
    """Resolved input/output locations for the figure scripts."""

    mosmon_outputs: Path
    full_eval: Path
    summary_table: Path
    species_analysis: Path
    dual_reanalysis: Path
    tracker_benchmark: Path
    output_dir: Path

    def require(self, *attrs: str) -> None:
        """Fail loudly, naming the exact missing input.

        The figure instructions forbid silently substituting or reconstructing a
        missing input, so an absent path is a hard stop with its role named.
        """
        missing = [(a, getattr(self, a)) for a in attrs if not getattr(self, a).exists()]
        if missing:
            lines = "\n".join(f"  - {a}: {p}" for a, p in missing)
            raise SystemExit(
                "Missing required input(s). Nothing was plotted.\n"
                f"{lines}\n"
                f"Check `paths` in {DEFAULT_CONFIG} and that the outputs volume is mounted."
            )


def load_config(path: str | Path | None = None) -> dict:
    """Read ``configs/figures.yaml`` (or an override path)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise SystemExit(f"Figure config not found: {cfg_path}")
    with open(cfg_path) as fh:
        return yaml.safe_load(fh) or {}


def resolve_paths(cfg: dict) -> FigurePaths:
    """Derive the standard output-tree layout, honouring explicit overrides."""
    p = dict(cfg.get("paths") or {})
    root = Path(str(p.get("mosmon_outputs", "")))

    def pick(key: str, default: Path) -> Path:
        value = p.get(key)
        return Path(str(value)) if value else default

    out = pick("output_dir", REPO_ROOT / "figures" / "generated")
    if not out.is_absolute():
        out = REPO_ROOT / out
    return FigurePaths(
        mosmon_outputs=root,
        full_eval=pick("full_eval", root / "full_eval"),
        summary_table=pick("summary_table", root / "full_eval_summary_table.csv"),
        species_analysis=pick("species_analysis", root / "species_analysis"),
        dual_reanalysis=pick("dual_reanalysis", root / "dual_reanalysis"),
        tracker_benchmark=pick("tracker_benchmark", root / "tracker_benchmark"),
        output_dir=out,
    )


# --------------------------------------------------------------------------- #
# 4. Provenance sidecar
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path, max_bytes: int | None = None) -> str | None:
    """sha256 of a file, or ``None`` when it exceeds ``max_bytes``.

    Multi-GB source videos are deliberately not hashed here: the tracking run's
    ``provenance.json`` already records their sha256, and re-reading hundreds of
    GB on every figure rebuild would make the scripts unusable.
    """
    path = Path(path)
    if max_bytes is not None and path.stat().st_size > max_bytes:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class CheckFailure(RuntimeError):
    """A value recomputed from source data disagrees with the manuscript."""


@dataclass
class Sidecar:
    """Accumulates everything the metadata JSON must record for one figure.

    Beyond bookkeeping this enforces the instructions' central rule: a value that
    fails :meth:`check` is reported as a discrepancy and the manuscript number is
    never substituted for the computed one.
    """

    figure: str
    description: str = ""
    hash_max_bytes: int | None = None
    inputs: list[dict] = field(default_factory=list)
    selections: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- inputs ------------------------------------------------------------ #
    def add_input(self, path: str | Path, role: str,
                  sha256: str | None = None) -> Path:
        """Record an input file. Fails loudly (naming it) if absent."""
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"Missing required input [{role}]: {path}\nNothing was plotted.")
        stat = path.stat()
        digest = sha256 if sha256 is not None else sha256_file(path, self.hash_max_bytes)
        self.inputs.append({
            "role": role,
            "path": str(path),
            "bytes": stat.st_size,
            "modified": _dt.datetime.fromtimestamp(stat.st_mtime, _dt.UTC).isoformat(),
            "sha256": digest,
            "sha256_source": ("provenance" if sha256 is not None
                              else ("computed" if digest else "skipped (over hash_max_bytes)")),
        })
        return path

    # -- context ----------------------------------------------------------- #
    def select(self, key: str, value: Any) -> None:
        """Record a selected video / image / frame id."""
        self.selections[key] = value

    def param(self, key: str, value: Any) -> None:
        """Record a plotting parameter."""
        self.params[key] = value

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  note: {text}")

    # -- validation -------------------------------------------------------- #
    def check(self, name: str, computed: Any, expected: Any,
              tol: float = 5e-4, unit: str = "") -> bool:
        """Compare a recomputed value with the manuscript's, and record it.

        Returns whether they agree. A disagreement is printed and stored; the
        caller keeps plotting the *computed* value, per the instructions.
        """
        try:
            ok = bool(abs(float(computed) - float(expected)) <= tol)
            delta: Any = float(computed) - float(expected)
        except (TypeError, ValueError):
            ok = computed == expected
            delta = None
        self.checks.append({
            "name": name, "computed": _jsonable(computed),
            "manuscript": _jsonable(expected), "tolerance": tol,
            "delta": _jsonable(delta), "unit": unit, "agrees": ok,
        })
        mark = "ok  " if ok else "DIFF"
        extra = "" if delta is None else f"  (delta {delta:+.4g})"
        print(f"  [{mark}] {name}: computed={computed!r} manuscript={expected!r}{extra}")
        return ok

    def failed_checks(self) -> list[dict]:
        return [c for c in self.checks if not c["agrees"]]

    # -- output ------------------------------------------------------------ #
    def to_dict(self, outputs: dict[str, str] | None = None) -> dict:
        return {
            "figure": self.figure,
            "description": self.description,
            "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "generated_by": Path(sys.argv[0]).name or "figstyle",
            "git_commit": _git_commit(),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "inputs": self.inputs,
            "selections": _jsonable(self.selections),
            "plot_parameters": _jsonable(self.params),
            "validation": {
                "n_checks": len(self.checks),
                "n_disagreements": len(self.failed_checks()),
                "checks": self.checks,
            },
            "notes": self.notes,
            "outputs": outputs or {},
        }


def _jsonable(obj: Any) -> Any:
    """Convert numpy/pandas scalars and containers into JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# 5. Export
# --------------------------------------------------------------------------- #
def save_figure(fig, name: str, sidecar: Sidecar, table: pd.DataFrame,
                out_dir: str | Path, panels: dict[str, Callable[[Any], None]] | None = None,
                panel_size: dict[str, tuple[float, float]] | None = None) -> dict[str, str]:
    """Write the full export set for one figure and return the paths.

    ``panels`` maps a panel key to a draw function taking a single Axes, so each
    panel can be re-rendered standalone from exactly the same code that produced
    it inside the composite figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    outputs["pdf"] = str(pdf_path)
    outputs["png"] = str(png_path)

    csv_path = out_dir / f"{name}_data.csv"
    table.to_csv(csv_path, index=False)
    outputs["data_csv"] = str(csv_path)

    for key, draw in (panels or {}).items():
        size = (panel_size or {}).get(key, (3.4, 2.6))
        pfig, pax = plt.subplots(figsize=size)
        draw(pax)
        p_pdf = panel_dir / f"{name}_{key}.pdf"
        pfig.savefig(p_pdf)
        pfig.savefig(panel_dir / f"{name}_{key}.png", dpi=300)
        plt.close(pfig)
        outputs[f"panel_{key}"] = str(p_pdf)

    meta_path = out_dir / f"{name}_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(sidecar.to_dict(outputs), fh, indent=2, sort_keys=False)
        fh.write("\n")
    outputs["metadata"] = str(meta_path)

    print(f"\nWrote {len(outputs)} files for {name}:")
    for key, value in outputs.items():
        print(f"  {key:14s} {os.path.relpath(value, REPO_ROOT)}")
    return outputs


def report_checks(sidecar: Sidecar) -> None:
    """Print the closing summary of manuscript-vs-data comparisons."""
    bad = sidecar.failed_checks()
    total = len(sidecar.checks)
    if not bad:
        print(f"\nAll {total} manuscript values reproduced from source data.")
        return
    print(f"\n{len(bad)} of {total} manuscript values DISAGREE with the source data.")
    print("The computed values were plotted; no manuscript number was substituted.")
    for c in bad:
        print(f"  - {c['name']}: computed {c['computed']!r} vs manuscript {c['manuscript']!r}")


# --------------------------------------------------------------------------- #
# 6. Small statistics helpers shared by more than one figure
# --------------------------------------------------------------------------- #
def pearson_r(x, y) -> float:
    """Pearson r over the pairwise-finite subset."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def linfit_with_band(x, y, n_grid: int = 100, alpha: float = 0.05):
    """Least-squares line plus its ``1 - alpha`` mean-response confidence band.

    The band is the textbook confidence interval for the fitted mean, computed
    from the data only — the figure instructions allow a band solely when it is
    genuinely computed rather than drawn for decoration.
    """
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 finite points to fit a line")
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), n_grid)
    fit = slope * grid + intercept
    resid = y - (slope * x + intercept)
    dof = n - 2
    s_err = np.sqrt(np.sum(resid ** 2) / dof)
    sxx = np.sum((x - x.mean()) ** 2)
    se_mean = s_err * np.sqrt(1.0 / n + (grid - x.mean()) ** 2 / sxx)
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, dof)
    return grid, fit, fit - tcrit * se_mean, fit + tcrit * se_mean, slope, intercept


# --------------------------------------------------------------------------- #
# 7. Frame selection and reading, shared by the figures that show video frames
# --------------------------------------------------------------------------- #
def median_active_frame(run_dir: str | Path) -> dict:
    """The processed frame whose active-track count is closest to the video median.

    Prefers ``population_timeseries.parquet`` (one row per processed frame),
    which is far cheaper than scanning the full track table, and falls back to
    counting rows per frame in ``tracks_clean.parquet`` for runs written with a
    reduced output profile. Ties break on the lowest frame index, so the choice
    is reproducible either way.
    """
    ts_path = Path(run_dir) / "population_timeseries.parquet"
    if ts_path.exists():
        ts = pd.read_parquet(ts_path, columns=["frame_idx", "n_larvae"])
    else:
        tracks_path = Path(run_dir) / "tracks_clean.parquet"
        if not tracks_path.exists():
            raise SystemExit(
                "Missing required input [per-frame active-track counts]: neither "
                f"{ts_path} nor {tracks_path} exists."
            )
        counts = pd.read_parquet(tracks_path, columns=["frame_idx"])["frame_idx"]
        ts = (counts.value_counts().sort_index().rename("n_larvae")
              .rename_axis("frame_idx").reset_index())
    median_active = float(ts["n_larvae"].median())
    ranked = ts.assign(_d=(ts["n_larvae"] - median_active).abs()).sort_values(
        ["_d", "frame_idx"])
    frames = np.sort(ts["frame_idx"].to_numpy())
    stride = int(np.diff(frames[:3]).min()) if frames.size > 2 else 1
    return {
        "frame_idx": int(ranked["frame_idx"].iloc[0]),
        "n_active_in_frame": int(ranked["n_larvae"].iloc[0]),
        "median_active_tracks": median_active,
        "frame_stride": max(1, stride),
    }


def read_video_frame(video_path: str | Path, frame_idx: int,
                     max_display_width: int = 1500) -> tuple[np.ndarray, float]:
    """Read one RGB frame, downscaled for display, with the scale it was reduced by.

    The scale is returned so overlaid coordinates can be reduced by the same
    factor and stay registered to the pixels. Downscaling is display-only; it
    never changes a plotted value.
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise SystemExit(f"Missing required input [source video]: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Could not read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = min(1.0, max_display_width / frame.shape[1])
    if scale < 1.0:
        frame = cv2.resize(frame, (int(round(frame.shape[1] * scale)),
                                   int(round(frame.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    return frame, scale


def _source_video_record(run_dir: str | Path) -> dict:
    """The run's source-video entry from its provenance manifest."""
    prov = Path(run_dir) / "provenance.json"
    if not prov.exists():
        return {}
    with open(prov) as fh:
        for item in json.load(fh).get("inputs", []):
            if item.get("role") == "source-video":
                return item
    return {}


def source_video_sha256(run_dir: str | Path) -> str | None:
    """sha256 of a run's source video, as recorded in its provenance manifest."""
    return _source_video_record(run_dir).get("sha256")


def source_video_path(run_dir: str | Path) -> Path:
    """Absolute path of a run's source video, from its provenance manifest."""
    path = _source_video_record(run_dir).get("path")
    if not path:
        raise SystemExit(f"No source-video entry in {Path(run_dir) / 'provenance.json'}")
    return Path(path)
