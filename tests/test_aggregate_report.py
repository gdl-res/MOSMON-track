import json

import pandas as pd

from mosmon_tracking.batch import aggregate_report
from mosmon_tracking.config import Config


def _cfg():
    cfg = Config()
    cfg.fair.enabled = False  # keep the test hermetic / fast
    return cfg


def _write_run(batch_dir, run_name, model, species, total_tracks, activity, duration):
    d = batch_dir / run_name
    d.mkdir(parents=True)
    summary = {
        "video_info": {"filename": run_name.split("__")[-1] + ".mp4", "meta_species": species},
        "model_settings": {"tracker": "bytetrack"},
        "behaviour": {
            "model_name": model, "tracker": "bytetrack",
            "total_tracks": total_tracks,
            "population_activity_index": activity,
            "median_track_duration_s": duration,
        },
        "qc": {"duplicate_track_proxy": 0.1},
    }
    (d / "video_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _make_batch(tmp_path):
    b = tmp_path / "batch"
    _write_run(b, "modelA__vid1", "modelA", "Aedes aegypti", 10, 0.5, 2.0)
    _write_run(b, "modelA__vid2", "modelA", "Culex pipiens", 20, 0.7, 3.0)
    _write_run(b, "modelB__vid1", "modelB", "Aedes aegypti", 8, 0.3, 1.0)
    return b


def test_per_run_table_has_all_runs(tmp_path):
    runs = aggregate_report(_make_batch(tmp_path), tmp_path / "agg", _cfg())
    assert len(runs) == 3
    assert set(runs["model"]) == {"modelA", "modelB"}
    assert (tmp_path / "agg" / "aggregate_runs.parquet").exists()


def test_group_by_model_means(tmp_path):
    aggregate_report(_make_batch(tmp_path), tmp_path / "agg", _cfg())
    by_model = pd.read_parquet(tmp_path / "agg" / "aggregate_by_model.parquet")
    a = by_model[by_model["model"] == "modelA"].iloc[0]
    assert a["n_runs"] == 2
    assert a["total_tracks"] == 15  # mean of 10 and 20


def test_group_by_species_written(tmp_path):
    aggregate_report(_make_batch(tmp_path), tmp_path / "agg", _cfg())
    by_species = pd.read_parquet(tmp_path / "agg" / "aggregate_by_species.parquet")
    assert "meta_species" in by_species.columns
    assert set(by_species["meta_species"]) == {"Aedes aegypti", "Culex pipiens"}


def test_report_files_written(tmp_path):
    aggregate_report(_make_batch(tmp_path), tmp_path / "agg", _cfg())
    out = tmp_path / "agg"
    assert (out / "aggregate_report.html").exists()
    assert (out / "aggregate_report.md").exists()
    html = (out / "aggregate_report.html").read_text()
    assert "Aggregate behaviour report" in html
    # at least one cross-video bar chart was produced
    assert any((out / "plots").glob("*.png"))


def test_empty_batch_is_graceful(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    runs = aggregate_report(empty, tmp_path / "agg", _cfg())
    assert runs.empty
    assert (tmp_path / "agg" / "aggregate_report.html").exists()
