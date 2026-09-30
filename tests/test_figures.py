"""Unit tests for the manuscript figure scripts (scripts/figures/).

These cover the pure helpers only: species naming, deterministic selection, the
manuscript-value checker, the sign convention of the dual-container contrast and
the export contract. Nothing here needs model weights, source videos, or the
outputs volume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "scripts" / "figures"
sys.path.insert(0, str(FIGURE_DIR))

import figstyle as fs  # noqa: E402


# --------------------------------------------------------------------------- #
# Species naming
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("Aedes aegypti", "Ae. aegypti"),
    ("aedes albopictus", "Ae. albopictus"),
    ("ANOPHELES STEPHENSI", "An. stephensi"),
    ("culex_pipiens", "Cx. pipiens"),
    ("Cx. pipiens", "Cx. pipiens"),
])
def test_species_label_canonicalises(raw, expected):
    assert fs.species_label(raw) == expected


def test_species_label_keeps_unknown_values_visible():
    # An unrecognised label must survive into the figure rather than vanish.
    assert fs.species_label("Aedes koreicus") == "Aedes koreicus"
    assert fs.species_label(None) == "unknown"
    assert fs.species_label(float("nan")) == "unknown"


def test_order_species_uses_manuscript_order_and_puts_unknowns_last():
    messy = ["culex_pipiens", "Aedes albopictus", "Aedes koreicus", "aedes aegypti",
             "Anopheles stephensi", "Aedes albopictus"]
    assert fs.order_species(messy) == [*fs.SPECIES_ORDER, "Aedes koreicus"]


def test_every_species_has_a_distinct_colour_and_marker():
    # Colour alone must never carry meaning: the figures have to survive
    # grayscale printing, so markers must differ too.
    styles = [fs.species_style(s) for s in fs.SPECIES_ORDER]
    assert len({s["color"] for s in styles}) == len(fs.SPECIES_ORDER)
    assert len({s["marker"] for s in styles}) == len(fs.SPECIES_ORDER)


# --------------------------------------------------------------------------- #
# Provenance sidecar
# --------------------------------------------------------------------------- #
def test_check_records_agreement_without_substituting():
    side = fs.Sidecar(figure="t")
    # The deduplicated correlation really does disagree with the manuscript's.
    assert side.check("r deduplicated", round(0.5709, 3), 0.601, tol=0.0005) is False
    assert side.check("r all runs", round(0.6008, 3), 0.601, tol=0.0005) is True
    assert len(side.failed_checks()) == 1
    # The recorded value is what was computed, never the manuscript's.
    assert side.checks[0]["computed"] == pytest.approx(0.571)
    assert side.checks[0]["manuscript"] == pytest.approx(0.601)
    assert side.checks[0]["delta"] == pytest.approx(-0.030, abs=1e-6)


def test_check_handles_non_numeric_and_missing_values():
    side = fs.Sidecar(figure="t")
    assert side.check("missing feature", None, 0.5) is False
    assert side.check("label", "abc", "abc") is True


def test_add_input_names_the_missing_file(tmp_path):
    side = fs.Sidecar(figure="t")
    with pytest.raises(SystemExit) as excinfo:
        side.add_input(tmp_path / "absent.parquet", "clean tracks")
    message = str(excinfo.value)
    assert "clean tracks" in message and "absent.parquet" in message


def test_add_input_hashes_small_files_and_skips_large_ones(tmp_path):
    small = tmp_path / "small.txt"
    small.write_bytes(b"x" * 32)
    side = fs.Sidecar(figure="t", hash_max_bytes=16)
    side.add_input(small, "small", sha256=None)
    assert side.inputs[0]["sha256"] is None
    assert "skipped" in side.inputs[0]["sha256_source"]

    side2 = fs.Sidecar(figure="t", hash_max_bytes=1024)
    side2.add_input(small, "small")
    assert side2.inputs[0]["sha256_source"] == "computed"
    assert len(side2.inputs[0]["sha256"]) == 64


def test_sidecar_serialises_numpy_scalars():
    side = fs.Sidecar(figure="t")
    side.param("n", np.int64(7))
    side.select("frac", np.float64(0.5))
    side.check("v", np.float64(1.0), 1.0)
    json.dumps(side.to_dict())  # must not raise


def test_require_lists_every_missing_path(tmp_path):
    paths = fs.FigurePaths(
        mosmon_outputs=tmp_path, full_eval=tmp_path / "nope",
        summary_table=tmp_path / "also_nope.csv", species_analysis=tmp_path,
        dual_reanalysis=tmp_path, tracker_benchmark=tmp_path, output_dir=tmp_path,
    )
    with pytest.raises(SystemExit) as excinfo:
        paths.require("full_eval", "summary_table", "species_analysis")
    message = str(excinfo.value)
    assert "nope" in message and "also_nope.csv" in message


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_resolve_paths_derives_the_standard_layout():
    paths = fs.resolve_paths({"paths": {"mosmon_outputs": "/data/out"}})
    assert paths.full_eval == Path("/data/out/full_eval")
    assert paths.summary_table == Path("/data/out/full_eval_summary_table.csv")
    assert paths.species_analysis == Path("/data/out/species_analysis")
    assert paths.dual_reanalysis == Path("/data/out/dual_reanalysis")
    # A relative output dir resolves from the repository root, not the cwd.
    assert paths.output_dir.is_absolute()


def test_resolve_paths_honours_explicit_overrides():
    paths = fs.resolve_paths({"paths": {"mosmon_outputs": "/data/out",
                                        "species_analysis": "/elsewhere/sa"}})
    assert paths.species_analysis == Path("/elsewhere/sa")
    assert paths.full_eval == Path("/data/out/full_eval")


def test_shipped_config_parses_and_resolves():
    paths = fs.resolve_paths(fs.load_config())
    assert paths.summary_table.name == "full_eval_summary_table.csv"


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def test_pearson_r_ignores_non_finite_pairs():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([2.0, 4.0, 6.0, 8.0, 1.0])
    assert fs.pearson_r(x, y) == pytest.approx(1.0)
    assert np.isnan(fs.pearson_r([1.0, np.nan], [2.0, 3.0]))


def test_linfit_recovers_a_known_line_and_brackets_it():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 60)
    y = 2.5 * x - 1.0 + rng.normal(0, 0.2, x.size)
    grid, fit, lo, hi, slope, intercept = fs.linfit_with_band(x, y)
    assert slope == pytest.approx(2.5, abs=0.05)
    assert intercept == pytest.approx(-1.0, abs=0.15)
    assert np.all(lo <= fit) and np.all(fit <= hi)
    assert grid.min() == pytest.approx(x.min())


# --------------------------------------------------------------------------- #
# Export contract
# --------------------------------------------------------------------------- #
def test_save_figure_writes_the_full_export_set(tmp_path):
    import matplotlib.pyplot as plt

    fs.apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    side = fs.Sidecar(figure="figXX")
    side.check("demo", 1.0, 1.0)
    table = pd.DataFrame({"panel": ["a"], "value": [1.0]})

    outputs = fs.save_figure(fig, "figXX", side, table, tmp_path,
                             panels={"a": lambda a: a.plot([0, 1], [1, 0])})

    for name in ("figXX.pdf", "figXX.png", "figXX_data.csv", "figXX_metadata.json"):
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "panels" / "figXX_a.pdf").exists()
    assert (tmp_path / "panels" / "figXX_a.png").exists()

    meta = json.loads((tmp_path / "figXX_metadata.json").read_text())
    assert meta["figure"] == "figXX"
    assert meta["validation"]["n_checks"] == 1
    assert meta["validation"]["n_disagreements"] == 0
    assert meta["outputs"]["pdf"] == outputs["pdf"]
    assert pd.read_csv(tmp_path / "figXX_data.csv").equals(table)


def test_pdf_export_keeps_text_as_vector():
    # fonttype 42 embeds TrueType instead of rasterising, which is what keeps
    # PDF text selectable at journal print size.
    import matplotlib

    fs.apply_style()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


# --------------------------------------------------------------------------- #
# Figure 3 - deterministic video selection
# --------------------------------------------------------------------------- #
def _fig03():
    import fig03_tracking_crowding as mod
    return mod


def test_select_example_runs_is_deterministic_and_percentile_based():
    mod = _fig03()
    df = pd.DataFrame({
        "run": [f"run{i}" for i in range(11)],
        "video": [f"v{i}.MP4" for i in range(11)],
        "out_dir": [f"/tmp/run{i}" for i in range(11)],
        "dens": np.linspace(2.0, 300.0, 11),
        "dup": np.linspace(0.0, 0.3, 11),
        "cov": np.linspace(95.0, 100.0, 11),
        "w": 3840, "h": 2160, "fps": 30.0,
    })
    side = fs.Sidecar(figure="t")
    first = mod.select_example_runs(df, [10, 50, 90], {}, side)
    second = mod.select_example_runs(df.sample(frac=1, random_state=3), [10, 50, 90], {},
                                     fs.Sidecar(figure="t"))
    assert [p["run"] for p in first] == [p["run"] for p in second]
    assert [p["regime"] for p in first] == ["low", "median", "high"]
    assert first[0]["density"] < first[1]["density"] < first[2]["density"]
    # The median pick must be the true middle video, not the extremes.
    assert first[1]["run"] == "run5"


def test_select_example_runs_rejects_an_unknown_override():
    mod = _fig03()
    df = pd.DataFrame({"run": ["a"], "video": ["a.MP4"], "out_dir": ["/tmp/a"],
                       "dens": [5.0], "dup": [0.0], "cov": [99.0],
                       "w": [3840], "h": [2160], "fps": [30.0]})
    with pytest.raises(SystemExit):
        mod.select_example_runs(df, [10, 50, 90], {"low": {"run": "missing"}},
                                fs.Sidecar(figure="t"))


# --------------------------------------------------------------------------- #
# Figure 4 / 5 - labelling and sign conventions
# --------------------------------------------------------------------------- #
def test_feature_labels_carry_the_statistic_suffix():
    import fig04_population_analysis as mod
    assert mod.feature_label("path_length_bl_med") == "Path length (BL)"
    assert mod.feature_label("path_length_bl_iqr") == "Path length (BL) IQR"
    assert mod.feature_label("aspect_ratio_cv_iqr") == "Aspect-ratio variability IQR"
    # An unmapped name degrades to something readable rather than raising.
    assert mod.feature_label("some_new_metric") == "some new metric"


def test_pair_effects_orients_the_sign_to_the_named_species():
    import fig05_dual_container as mod

    # The stored table orders the pair alphabetically, so rows where species_a is
    # not the reference species must have their sign flipped before averaging.
    dual = pd.DataFrame({
        "run": ["r1", "r2"],
        "species_a": ["Aedes albopictus", "Culex pipiens"],
        "species_b": ["Culex pipiens", "Aedes albopictus"],
        "feature": ["mean_speed_bl_s", "mean_speed_bl_s"],
        "rank_biserial": [0.60, -0.40],
        "n_a": [10, 10], "n_b": [10, 10],
    })
    out = mod.pair_effects(dual, ("Ae. albopictus", "Cx. pipiens"))
    assert len(out) == 1
    assert out["rank_biserial"].iloc[0] == pytest.approx(0.50)
    assert int(out["n_videos"].iloc[0]) == 2

    flipped = mod.pair_effects(dual, ("Cx. pipiens", "Ae. albopictus"))
    assert flipped["rank_biserial"].iloc[0] == pytest.approx(-0.50)


def test_pair_effects_fails_loudly_for_an_absent_pair():
    import fig05_dual_container as mod
    dual = pd.DataFrame({
        "run": ["r1"], "species_a": ["Aedes albopictus"], "species_b": ["Culex pipiens"],
        "feature": ["mean_speed_bl_s"], "rank_biserial": [0.6], "n_a": [10], "n_b": [10],
    })
    with pytest.raises(SystemExit):
        mod.pair_effects(dual, ("Ae. aegypti", "An. stephensi"))


