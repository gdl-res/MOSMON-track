"""FAIR-compliant packaging of pipeline outputs.

Every run directory is made **F**indable, **A**ccessible, **I**nteroperable, and
**R**eusable:

* **Findable**  - a persistent run UUID, rich machine-readable metadata, and an
  RO-Crate that can be uploaded to Zenodo to mint a DOI.
* **Accessible** - open formats only (Parquet/CSV/JSON/PNG/NPY); metadata lives in
  plain JSON retrievable independently of the data.
* **Interoperable** - schema.org / RO-Crate 1.1 vocabulary, NCBI Taxonomy IDs for
  species, UCUM codes for units, and an explicit per-table data dictionary.
* **Reusable** - a clear license (default CC-BY-4.0), full provenance (software
  versions, config, input checksums, command, timestamp), and SHA-256 checksums.

The public entry point is :func:`finalize_fair`, called at the end of every run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config

RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
SCHEMA = "http://schema.org/"

# Controlled vocabulary: canonical species name -> NCBI Taxonomy ID.
SPECIES_TAXON_ID = {
    "Aedes aegypti": "NCBI:txid7159",
    "Aedes albopictus": "NCBI:txid7160",
    "Anopheles stephensi": "NCBI:txid30069",
    "Culex pipiens": "NCBI:txid7175",
}


def species_iri(name: str) -> str | None:
    """Return a resolvable NCBI Taxonomy IRI for a canonical species name."""
    tax = SPECIES_TAXON_ID.get(name)
    if not tax:
        return None
    return f"http://purl.obolibrary.org/obo/NCBITaxon_{tax.split('txid')[-1]}"


# Per-table data dictionaries: column -> (UCUM-ish unit, description).
# Units: 'px'/'px/s'/'px2' are pixel-based (non-physical); 'mm','mm/s','s','rad','1'
# follow UCUM. Pixels have no UCUM code, so we label them explicitly as such.
_UNIT_PIXEL_NOTE = "pixel (non-physical image unit; not UCUM)"

TABLE_SCHEMAS: dict[str, dict[str, tuple[str, str]]] = {
    "tracks_raw": {
        "video_path": ("1", "Source video file path"),
        "video_name": ("1", "Source video filename"),
        "model_path": ("1", "YOLO11 model weights path"),
        "model_name": ("1", "Model identifier (weights stem)"),
        "tracker": ("1", "Tracker algorithm (bytetrack/botsort)"),
        "frame_idx": ("{frame}", "Original-video frame index"),
        "time_s": ("s", "Timestamp from frame_idx / fps"),
        "track_id": ("1", "Persistent track identifier within the video"),
        "class_id": ("1", "Model class index (see model.names)"),
        "class_name": ("1", "Class name; map to NCBI taxon via data dictionary"),
        "confidence": ("1", "Detection confidence [0,1]"),
        "x1": (_UNIT_PIXEL_NOTE, "Bounding-box left"),
        "y1": (_UNIT_PIXEL_NOTE, "Bounding-box top"),
        "x2": (_UNIT_PIXEL_NOTE, "Bounding-box right"),
        "y2": (_UNIT_PIXEL_NOTE, "Bounding-box bottom"),
        "w": (_UNIT_PIXEL_NOTE, "Bounding-box width"),
        "h": (_UNIT_PIXEL_NOTE, "Bounding-box height"),
        "cx": (_UNIT_PIXEL_NOTE, "Bounding-box center x"),
        "cy": (_UNIT_PIXEL_NOTE, "Bounding-box center y"),
        "cx_norm": ("1", "Center x normalized by frame width [0,1]"),
        "cy_norm": ("1", "Center y normalized by frame height [0,1]"),
        "frame_width": (_UNIT_PIXEL_NOTE, "Frame width"),
        "frame_height": (_UNIT_PIXEL_NOTE, "Frame height"),
    },
    "tracks_clean": {
        "is_interpolated": ("1", "True if row was filled across a short gap"),
        "is_smoothed": ("1", "True if center path was Savitzky-Golay smoothed"),
        "jump_flag": ("1", "True if speed is an implausible outlier"),
        "border_flag": ("1", "True if within the border margin"),
        "low_confidence_flag": ("1", "True if confidence below threshold"),
        "dx": (_UNIT_PIXEL_NOTE, "Displacement x from previous frame"),
        "dy": (_UNIT_PIXEL_NOTE, "Displacement y from previous frame"),
        "dt": ("s", "Validated time delta from previous frame"),
        "speed_px_s": ("px/s", "Instantaneous speed (pixel units)"),
        "acceleration_px_s2": ("px/s2", "Instantaneous acceleration (pixel units)"),
        "heading_rad": ("rad", "Heading from displacement vector"),
        "turn_angle_rad": ("rad", "Change in heading, wrapped to (-pi,pi]"),
        "bbox_area_px": ("px2", "Bounding-box area"),
        "bbox_aspect_ratio": ("1", "Bounding-box width/height"),
        "roi_label": ("1", "Region: border/center/intermediate"),
        "dist_to_border_px": (_UNIT_PIXEL_NOTE, "Distance to nearest frame edge"),
        "nearest_neighbor_distance_px": (_UNIT_PIXEL_NOTE, "Distance to k-th nearest larva"),
        "local_density": ("1", "Neighbour count within 5% of frame diagonal"),
        "movement_state": ("1", "freezing/slow/active/burst/unknown"),
    },
    "track_summary": {
        "duration_s": ("s", "Track duration"),
        "n_frames": ("{frame}", "Number of observed/filled frames"),
        "interpolated_fraction": ("1", "Fraction of interpolated frames"),
        "path_length_px": (_UNIT_PIXEL_NOTE, "Total path length"),
        "net_displacement_px": (_UNIT_PIXEL_NOTE, "Start-to-end displacement"),
        "mean_speed_px_s": ("px/s", "Mean speed"),
        "median_speed_px_s": ("px/s", "Median speed"),
        "max_speed_px_s": ("px/s", "Max speed"),
        "mean_acceleration_px_s2": ("px/s2", "Mean absolute acceleration"),
        "mean_abs_turn_angle_rad": ("rad", "Mean absolute turn angle"),
        "tortuosity": ("1", "Path length / net displacement"),
        "straightness": ("1", "Net displacement / path length"),
        "radius_of_gyration_px": (_UNIT_PIXEL_NOTE, "RMS distance from centroid"),
        "spatial_entropy": ("nat", "Occupancy entropy over spatial bins"),
        "quality_score": ("1", "Interpretable QC score [0,1]; NOT biological"),
    },
    "video_inventory": {
        "fps": ("{frame}/s", "Frames per second"),
        "duration_s": ("s", "Video duration"),
        "width": (_UNIT_PIXEL_NOTE, "Frame width"),
        "height": (_UNIT_PIXEL_NOTE, "Frame height"),
    },
}

_ENCODING = {
    ".parquet": "application/vnd.apache.parquet",
    ".csv": "text/csv",
    ".json": "application/json",
    ".png": "image/png",
    ".npy": "application/x-numpy-data",
    ".md": "text/markdown",
    ".html": "text/html",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_provenance(
    cfg: Config,
    run_id: str,
    inputs: list[dict[str, Any]] | None = None,
    command: str | None = None,
) -> dict:
    """Capture software/environment/config provenance for a run."""
    return {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "name": "mosmon-tracking",
            "version": __version__,
            "repository": cfg.fair.project_url,
        },
        "command": command or " ".join(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "packages": {
                p: _pkg_version(p)
                for p in ("ultralytics", "torch", "torchvision", "numpy", "pandas",
                          "scipy", "opencv-python", "pyarrow", "matplotlib")
            },
        },
        "config": cfg.to_dict(),
        "inputs": inputs or [],
    }


def build_data_dictionary(run_dir: Path, species_present: list[str] | None = None) -> dict:
    """Build a data dictionary for the tables actually present in ``run_dir``."""
    tables: dict[str, Any] = {}
    for logical, schema in TABLE_SCHEMAS.items():
        for ext in (".parquet", ".csv"):
            f = run_dir / f"{logical}{ext}"
            if f.exists():
                tables[f.name] = {
                    "fields": [
                        {"name": col, "unit": unit, "description": desc}
                        for col, (unit, desc) in schema.items()
                    ]
                }
                break
    dd: dict[str, Any] = {
        "description": "Column-level data dictionary with units (UCUM where applicable).",
        "unit_notes": {
            "pixel": "Pixel quantities are non-physical image units (no UCUM code). "
                     "Use calibration to obtain mm.",
            "nat": "Entropy in nats (natural-log base).",
        },
        "tables": tables,
    }
    if species_present:
        dd["controlled_vocabularies"] = {
            "species": {
                name: {
                    "ncbi_taxon": SPECIES_TAXON_ID.get(name),
                    "iri": species_iri(name),
                }
                for name in species_present
            }
        }
    return dd


def _file_entity(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    return {
        "@id": rel,
        "@type": "File",
        "name": path.name,
        "encodingFormat": _ENCODING.get(path.suffix.lower(), "application/octet-stream"),
        "contentSize": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_checksums(run_dir: Path, exclude: set[str]) -> Path:
    """Write a sha256sum-format checksums file for all files in the run dir."""
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in exclude:
            lines.append(f"{sha256_file(p)}  {p.relative_to(run_dir).as_posix()}")
    out = run_dir / "checksums.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _person_or_org(cfg: Config) -> tuple[list[dict], dict | None]:
    """Build schema.org author + publisher entities from FAIR config."""
    authors: list[dict] = []
    if cfg.fair.creator_name:
        person = {
            "@id": cfg.fair.creator_orcid or f"#person-{uuid.uuid4().hex[:8]}",
            "@type": "Person",
            "name": cfg.fair.creator_name,
        }
        if cfg.fair.affiliation:
            person["affiliation"] = cfg.fair.affiliation
        authors.append(person)
    publisher = None
    if cfg.fair.publisher:
        publisher = {"@id": f"#org-{uuid.uuid4().hex[:8]}", "@type": "Organization",
                     "name": cfg.fair.publisher}
    return authors, publisher


def build_ro_crate(
    run_dir: Path,
    cfg: Config,
    run_id: str,
    dataset_name: str,
    dataset_description: str,
    species_present: list[str] | None = None,
) -> Path:
    """Write an RO-Crate 1.1 ``ro-crate-metadata.json`` describing the run.

    The crate is Zenodo-uploadable: it carries license, authors, keywords,
    per-file checksums/sizes/formats, and provenance, so a DOI can be minted.
    """
    root = run_dir
    crate_files = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "ro-crate-metadata.json"
    ]
    file_entities = [_file_entity(p, root) for p in crate_files]
    has_part = [{"@id": e["@id"]} for e in file_entities]

    authors, publisher = _person_or_org(cfg)
    now = datetime.now(timezone.utc).date().isoformat()

    root_dataset = {
        "@id": "./",
        "@type": "Dataset",
        "name": dataset_name,
        "description": dataset_description,
        "identifier": f"urn:uuid:{run_id}",
        "datePublished": now,
        "license": {"@id": cfg.fair.license},
        "keywords": cfg.fair.dataset_keywords,
        "hasPart": has_part,
    }
    if authors:
        root_dataset["author"] = [{"@id": a["@id"]} for a in authors]
        root_dataset["creator"] = [{"@id": a["@id"]} for a in authors]
    if publisher:
        root_dataset["publisher"] = {"@id": publisher["@id"]}
    if species_present:
        about = [{"@id": species_iri(s)} for s in species_present if species_iri(s)]
        if about:
            root_dataset["about"] = about

    graph: list[dict] = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
            "about": {"@id": "./"},
        },
        root_dataset,
        {
            "@id": cfg.fair.license,
            "@type": "CreativeWork",
            "name": cfg.fair.license_name,
            "url": cfg.fair.license,
        },
        {
            "@id": "#mosmon-tracking",
            "@type": "SoftwareApplication",
            "name": "mosmon-tracking",
            "version": __version__,
            "url": cfg.fair.project_url,
        },
    ]
    graph.extend(authors)
    if publisher:
        graph.append(publisher)
    # Species as defined terms (Interoperable: link to NCBI Taxonomy).
    for s in species_present or []:
        iri = species_iri(s)
        if iri:
            graph.append({
                "@id": iri, "@type": "Taxon", "name": s,
                "identifier": SPECIES_TAXON_ID.get(s),
            })
    graph.extend(file_entities)

    crate = {"@context": RO_CRATE_CONTEXT, "@graph": graph}
    out = root / "ro-crate-metadata.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(crate, fh, indent=2)
    return out


def finalize_fair_file(
    file_path: str | Path,
    cfg: Config,
    logical_table: str | None = None,
    inputs: list[dict[str, Any]] | None = None,
    command: str | None = None,
) -> dict:
    """FAIR sidecars for a single output file (e.g. the video inventory CSV).

    Writes ``<file>.provenance.json``, ``<file>.sha256``, and (if a schema is
    known) ``<file>.data_dictionary.json`` next to the file.
    """
    if not cfg.fair.enabled:
        return {}
    file_path = Path(file_path)
    run_id = str(uuid.uuid4())
    prov = collect_provenance(cfg, run_id, inputs=inputs, command=command)
    prov["output_file"] = {
        "name": file_path.name,
        "sha256": sha256_file(file_path),
        "contentSize": file_path.stat().st_size,
        "license": cfg.fair.license,
    }
    file_path.with_suffix(file_path.suffix + ".provenance.json").write_text(
        json.dumps(prov, indent=2, default=str), encoding="utf-8"
    )
    file_path.with_suffix(file_path.suffix + ".sha256").write_text(
        f"{prov['output_file']['sha256']}  {file_path.name}\n", encoding="utf-8"
    )
    if logical_table and logical_table in TABLE_SCHEMAS:
        schema = TABLE_SCHEMAS[logical_table]
        dd = {
            "table": file_path.name,
            "license": cfg.fair.license,
            "fields": [
                {"name": col, "unit": unit, "description": desc}
                for col, (unit, desc) in schema.items()
            ],
        }
        file_path.with_suffix(file_path.suffix + ".data_dictionary.json").write_text(
            json.dumps(dd, indent=2), encoding="utf-8"
        )
    return prov


def finalize_fair(
    run_dir: str | Path,
    cfg: Config,
    dataset_name: str,
    dataset_description: str,
    inputs: list[dict[str, Any]] | None = None,
    species_present: list[str] | None = None,
    command: str | None = None,
) -> dict:
    """Write all FAIR sidecars into ``run_dir`` and return the provenance dict.

    Order matters: provenance + data dictionary first, then checksums over the
    data, then the RO-Crate (which references checksums and provenance too).
    """
    run_dir = Path(run_dir)
    if not cfg.fair.enabled:
        return {}
    run_id = str(uuid.uuid4())

    prov = collect_provenance(cfg, run_id, inputs=inputs, command=command)
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str),
                                             encoding="utf-8")
    dd = build_data_dictionary(run_dir, species_present)
    (run_dir / "data_dictionary.json").write_text(json.dumps(dd, indent=2),
                                                  encoding="utf-8")
    # LICENSE pointer file
    (run_dir / "LICENSE.txt").write_text(
        f"{cfg.fair.license_name}\n{cfg.fair.license}\n", encoding="utf-8"
    )
    write_checksums(run_dir, exclude={"checksums.sha256", "ro-crate-metadata.json"})
    build_ro_crate(run_dir, cfg, run_id, dataset_name, dataset_description,
                   species_present=species_present)
    return prov
