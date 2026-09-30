import json

import pandas as pd

from mosmon_tracking.config import Config
from mosmon_tracking.fair import (
    SPECIES_TAXON_ID,
    build_data_dictionary,
    finalize_fair,
    finalize_fair_file,
    species_iri,
)
from tests.test_track_postprocess import _cfg, _make_track


def test_species_controlled_vocabulary():
    assert SPECIES_TAXON_ID["Aedes aegypti"] == "NCBI:txid7159"
    assert "NCBITaxon_7159" in species_iri("Aedes aegypti")
    assert species_iri("not a species") is None


def test_finalize_fair_file(tmp_path):
    cfg = Config()
    inv = tmp_path / "video_inventory.csv"
    pd.DataFrame({"filename": ["a.mp4"], "fps": [30.0]}).to_csv(inv, index=False)
    finalize_fair_file(inv, cfg, logical_table="video_inventory")
    assert (tmp_path / "video_inventory.csv.provenance.json").exists()
    assert (tmp_path / "video_inventory.csv.sha256").exists()
    dd = json.loads((tmp_path / "video_inventory.csv.data_dictionary.json").read_text())
    assert any(f["name"] == "fps" for f in dd["fields"])
    prov = json.loads((tmp_path / "video_inventory.csv.provenance.json").read_text())
    assert prov["software"]["name"] == "mosmon-tracking"
    assert "config" in prov and "environment" in prov


def test_finalize_fair_run_dir(tmp_path):
    cfg = Config()
    (tmp_path / "tracks_raw.csv").write_text("track_id,cx\n1,10\n", encoding="utf-8")
    (tmp_path / "track_summary.csv").write_text("track_id,duration_s\n1,1.0\n", encoding="utf-8")
    prov = finalize_fair(
        tmp_path, cfg,
        dataset_name="test run", dataset_description="desc",
        species_present=["Aedes aegypti", "Culex pipiens"],
    )
    # All FAIR sidecars present
    for f in ("ro-crate-metadata.json", "provenance.json", "data_dictionary.json",
              "checksums.sha256", "LICENSE.txt"):
        assert (tmp_path / f).exists(), f
    crate = json.loads((tmp_path / "ro-crate-metadata.json").read_text())
    assert crate["@context"].endswith("context")
    graph = {e["@id"]: e for e in crate["@graph"]}
    root = graph["./"]
    assert root["@type"] == "Dataset"
    assert root["license"]["@id"] == cfg.fair.license
    assert root["identifier"].startswith("urn:uuid:")
    # files referenced with checksums
    assert any(e.get("@type") == "File" and "sha256" in e for e in crate["@graph"])
    # species linked to NCBI taxonomy
    assert any(e.get("@type") == "Taxon" for e in crate["@graph"])
    assert prov["run_id"]


def test_data_dictionary_only_lists_present_tables(tmp_path):
    (tmp_path / "tracks_raw.csv").write_text("track_id\n1\n", encoding="utf-8")
    dd = build_data_dictionary(tmp_path, species_present=["Aedes aegypti"])
    assert "tracks_raw.csv" in dd["tables"]
    assert "track_summary.csv" not in dd["tables"]
    assert dd["controlled_vocabularies"]["species"]["Aedes aegypti"]["ncbi_taxon"]


def test_pipeline_emits_fair(tmp_path):
    from mosmon_tracking.batch import pipeline_from_raw

    a = _make_track(track_id=1, n=20, x0=100, y0=100, vx=3, vy=0)
    b = _make_track(track_id=2, n=20, x0=300, y0=200, vx=0, vy=2)
    raw = pd.concat([a, b], ignore_index=True)
    cfg = _cfg()
    pipeline_from_raw(raw, cfg, tmp_path, video_path=None)
    assert (tmp_path / "ro-crate-metadata.json").exists()
    assert (tmp_path / "provenance.json").exists()
    assert (tmp_path / "checksums.sha256").exists()
    crate = json.loads((tmp_path / "ro-crate-metadata.json").read_text())
    ids = {e["@id"] for e in crate["@graph"]}
    # cleaned tracks table is part of the crate
    assert any(i.startswith("tracks_clean") for i in ids)
