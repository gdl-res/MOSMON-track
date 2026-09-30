"""Parse MOSMON-Larvae metadata encoded in video filenames.

Filenames are messy and human-authored: token order varies, abbreviations are
inconsistent (``camerapositionA`` / ``camposA`` / ``cpA`` / ``posA``), there are
typos (``lighartcold`` vs ``lightartcold``), and many optional modifier tokens
(``turbolence``, ``leavesadded``, ``thermometer`` ...). We therefore classify
each underscore-delimited token independently instead of relying on position,
and we always record:

- the recognised fields (``None`` when absent),
- a list of unrecognised tokens (for QC / future parsing rules),
- a list of human-readable warnings.

Example::

    20251009_alboSX_culexDX_stage3and4_djiosmoaction5_video_4k30fps_
    rocksteadywide_lightnatartcold_cp4_big2boxcontainer_depth45mm_larvaeadded.MP4
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Canonical class names (match model.names where possible).
SPECIES_AEGYPTI = "Aedes aegypti"
SPECIES_ALBOPICTUS = "Aedes albopictus"
SPECIES_STEPHENSI = "Anopheles stephensi"
SPECIES_PIPIENS = "Culex pipiens"
SPECIES_UNIDENTIFIED = "unidentified"

# Known modifier tokens (substring match on the raw token).
_MODIFIER_KEYS = (
    "larvaeadded",
    "larvaeadd",
    "larvaadded",
    "larvadd",
    "larvaeinsert",
    "turbolence",
    "leavesadded",
    "leaveadded",
    "thermometer",
    "boxmoved",
    "whitecilinder",
    "xiaomivideo",
    "outsideinsettario",
    "take2",
    "take3",
    "2cameras",
    "2cams",
    "30cm",
)


@dataclass
class VideoMetadata:
    """Structured metadata parsed from a single filename."""

    filename: str
    stem: str
    recording_date: str | None = None  # ISO yyyy-mm-dd
    species: list[str] = field(default_factory=list)
    species_left: str | None = None  # SX compartment in dual-container videos
    species_right: str | None = None  # DX compartment in dual-container videos
    is_dual_container: bool = False
    stage: str | None = None
    camera_family: str | None = None
    camera_model: str | None = None
    modality: str | None = None  # video / photo
    resolution: str | None = None  # e.g. "4K", "8K", "5.7K"
    fps: int | None = None
    lens_tokens: list[str] = field(default_factory=list)
    lighting: list[str] = field(default_factory=list)  # natural / artificial_cold / artificial_warm
    camera_position: str | None = None
    container_type: str | None = None
    water_depth_mm: float | None = None
    modifiers: list[str] = field(default_factory=list)
    unrecognized_tokens: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Flatten list fields for tabular (CSV/Parquet) friendliness.
        for key in ("species", "lens_tokens", "lighting", "modifiers", "unrecognized_tokens", "warnings"):
            d[key] = "|".join(d[key]) if d[key] else ""
        return d


def resolve_species_fragment(fragment: str) -> str | None:
    """Map a raw species fragment (possibly abbreviated) to a canonical name."""
    f = fragment.lower()
    if "aegypti" in f:
        return SPECIES_AEGYPTI
    if "albo" in f:  # albopictus / albo
        return SPECIES_ALBOPICTUS
    if "culex" in f or "pipiens" in f:
        return SPECIES_PIPIENS
    if "anopheles" in f or "stephensi" in f:  # files use "stephensis"
        return SPECIES_STEPHENSI
    if "notidentified" in f or "notident" in f:
        return SPECIES_UNIDENTIFIED
    return None


def _parse_date(token: str) -> tuple[str | None, str | None]:
    """Parse a leading yyyymmdd token. Returns (iso_date, warning)."""
    if not re.fullmatch(r"\d{8}", token):
        return None, None
    try:
        d: date = datetime.strptime(token, "%Y%m%d").date()
        return d.isoformat(), None
    except ValueError:
        return None, f"unparseable date token '{token}'"


def _classify_camera(token: str) -> tuple[str, str] | None:
    """Return (family, model) if the token is a camera token, else None."""
    if token.startswith("gopro"):
        return "gopro", token
    if token.startswith("dji"):
        return "dji", token
    if token.startswith("insta360"):
        return "insta360", token
    return None


def _classify_lighting(token: str) -> list[str]:
    """Classify a lighting token into natural / artificial_cold / artificial_warm."""
    if not (token.startswith("light") or token.startswith("ligh") or token in {"artcold", "artwarm", "natural", "artcoldnat"}):
        return []
    out: list[str] = []
    if "nat" in token:
        out.append("natural")
    if "cold" in token:
        out.append("artificial_cold")
    if "warm" in token:
        out.append("artificial_warm")
    return out


def _classify_position(token: str) -> str | None:
    """Parse a camera-position token to a normalized label (A-Z, digit, or FREE)."""
    m = re.fullmatch(r"(?:cameraposition|campos|cp|pos)([a-z0-9]+)", token)
    if not m:
        return None
    return m.group(1).upper()


def _classify_container(token: str) -> str | None:
    if "big2box" in token:
        return "big2box"
    if "big1box" in token:
        return "big1box"
    if "bigcontainer" in token or token == "big":
        return "big"
    if "stdclean" in token:
        return "std_clean"
    if "stdcontainer" in token:
        return "std"
    if "bicontainer" in token:
        return "bi"
    if "fountain" in token or "fontana" in token:
        return "fountain"
    return None


def _classify_lens(token: str) -> bool:
    lens_keys = ("rocksteady", "lenslinear", "distortionlinear", "distlinear",
                 "ultrawide", "widescreen", "linear", "wide", "fullframe", "cinematic")
    return any(k in token for k in lens_keys)


def parse_filename(filename: str) -> VideoMetadata:
    """Parse a single video filename into :class:`VideoMetadata`."""
    name = Path(filename).name
    stem = Path(filename).stem
    low = stem.lower()
    meta = VideoMetadata(filename=name, stem=stem)

    tokens = low.split("_")

    # --- whole-stem extractions (robust to surrounding text) ---
    fps_m = re.search(r"(\d+)fps", low)
    if fps_m:
        meta.fps = int(fps_m.group(1))
    res_m = re.search(r"(\d)dot(\d)k?", low)
    if res_m:
        meta.resolution = f"{res_m.group(1)}.{res_m.group(2)}K"
    else:
        # e.g. "4k30fps", "8k30fps" -> k followed by a digit/'fps', not a letter
        # (so we don't match the 'k' in words like "black").
        res_k = re.search(r"(\d+)k(?![a-z])", low)
        if res_k:
            meta.resolution = f"{res_k.group(1)}K"
    depth_m = re.search(r"depth(\d+)mm", low)
    if depth_m:
        meta.water_depth_mm = float(depth_m.group(1))
    elif re.search(r"depth\d+(?!mm)", low):
        meta.warnings.append("water-depth token looks truncated (no 'mm' suffix)")

    # --- dual-container species (SX = left, DX = right) ---
    sx = next((t for t in tokens if t.endswith("sx")), None)
    dx = next((t for t in tokens if t.endswith("dx")), None)
    if sx and dx:
        meta.is_dual_container = True
        meta.species_left = resolve_species_fragment(sx[:-2])
        meta.species_right = resolve_species_fragment(dx[:-2])
        for sp in (meta.species_left, meta.species_right):
            if sp and sp not in meta.species:
                meta.species.append(sp)
        if meta.species_left is None or meta.species_right is None:
            meta.warnings.append(f"could not fully resolve dual species from '{sx}'/'{dx}'")

    # --- per-token classification ---
    consumed: set[int] = set()
    for i, tok in enumerate(tokens):
        if not tok:
            consumed.add(i)
            continue
        # date (only meaningful as first token, but classify anywhere)
        if i == 0:
            iso, warn = _parse_date(tok)
            if iso:
                meta.recording_date = iso
                consumed.add(i)
                continue
            if warn:
                meta.warnings.append(warn)

        if tok in ("mosmon", "iss"):  # project/site markers, not informative metadata
            consumed.add(i)
            continue
        if tok in ("video", "photo"):
            meta.modality = tok
            consumed.add(i)
            continue
        if tok.endswith("sx") or tok.endswith("dx"):
            consumed.add(i)  # already handled by dual logic
            continue

        # single species fragments
        sp = resolve_species_fragment(tok)
        if sp is not None and not meta.is_dual_container:
            if sp not in meta.species:
                meta.species.append(sp)
            consumed.add(i)
            continue
        if tok in ("aedes", "anopheles"):  # bare genus tokens accompany an epithet
            consumed.add(i)
            continue

        m = re.fullmatch(r"stage([0-9and]+)", tok)
        if m:
            meta.stage = m.group(1).replace("and", "-")
            consumed.add(i)
            continue

        cam = _classify_camera(tok)
        if cam:
            meta.camera_family, meta.camera_model = cam
            consumed.add(i)
            continue

        lit = _classify_lighting(tok)
        if lit:
            for x in lit:
                if x not in meta.lighting:
                    meta.lighting.append(x)
            consumed.add(i)
            continue

        pos = _classify_position(tok)
        if pos is not None:
            meta.camera_position = pos
            consumed.add(i)
            continue

        cont = _classify_container(tok)
        if cont is not None:
            meta.container_type = cont
            consumed.add(i)
            continue

        # resolution/fps tokens already captured at stem level
        if re.search(r"\d+fps", tok) or re.fullmatch(r"\d+k\w*", tok) or re.search(r"\ddot\d", tok) \
                or tok in ("slowmotion", "fullframe", "widescreen"):
            consumed.add(i)
            continue
        if tok.startswith("depth"):
            consumed.add(i)
            continue

        if _classify_lens(tok):
            meta.lens_tokens.append(tok)
            consumed.add(i)
            continue

        if any(k in tok for k in _MODIFIER_KEYS):
            meta.modifiers.append(tok)
            consumed.add(i)
            continue

    meta.unrecognized_tokens = [tokens[i] for i in range(len(tokens)) if i not in consumed and tokens[i]]
    if not meta.species:
        meta.warnings.append("no species could be parsed from filename")
    if meta.recording_date is None:
        meta.warnings.append("no recording date could be parsed from filename")
    return meta
