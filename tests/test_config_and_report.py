from pathlib import Path

from mosmon_tracking.calibration import Calibrator
from mosmon_tracking.config import Config, load_config
from mosmon_tracking.reports import build_html, build_markdown, write_html, write_markdown


def test_load_default_yaml():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_config(cfg_path)
    assert cfg.model.imgsz == 1920
    assert cfg.tracker.type == "bytetrack"
    assert cfg.heatmaps.bins_x == 128
    assert cfg.features.activity_speed_threshold_px_s == 5.0


def test_partial_yaml_falls_back(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("model:\n  imgsz: 640\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.model.imgsz == 640        # overridden
    assert cfg.model.conf == 0.15        # default preserved
    assert cfg.tracker.type == "bytetrack"


def test_calibrator_modes():
    cfg = Config()
    cal = Calibrator.from_config(cfg.calibration)
    assert cal.mode == "pixel" and cal.units == "px"
    cfg.calibration.enabled = True
    cfg.calibration.pixels_per_mm = 10.0
    cal = Calibrator.from_config(cfg.calibration)
    assert cal.mode == "scalar" and cal.units == "mm"
    assert cal.distance_mm(100.0) == 10.0


def test_report_generation(tmp_path):
    md = build_markdown(
        video_info={"filename": "x.mp4", "fps": 30.0},
        model_settings={"imgsz": 1920, "tracker": "bytetrack"},
        qc={"n_detections": 100, "warnings": ["test warning"]},
        video_summary={"total_tracks": 5, "global_mean_speed_px_s": 12.3},
        plot_paths=["plots/trajectories.png"],
        heatmap_paths=["heatmaps/occupancy.png"],
    )
    assert "Behaviour report" in md
    assert "test warning" in md
    assert "Interpretation caveats" in md
    out = write_markdown(md, tmp_path / "report.md")
    assert out.exists()


def test_html_report_renders_tables_and_images(tmp_path):
    html = build_html(
        video_info={"filename": "x.mp4", "fps": 30.0},
        model_settings={"imgsz": 1920, "tracker": "botsort"},
        qc={"n_detections": 100, "warnings": ["test warning"]},
        video_summary={"total_tracks": 5},
        plot_paths=["plots/trajectories.png"],
        heatmap_paths=["heatmaps/occupancy_aedes albopictus.png"],
    )
    # Real HTML, not a <pre> dump
    assert "<table>" in html and "<th>" in html
    assert "<img" in html
    # spaces in heatmap filenames are URL-encoded
    assert "occupancy_aedes%20albopictus.png" in html
    # warnings and caveats present
    assert "test warning" in html
    assert "Interpretation caveats" in html
    assert "<pre>" not in html
    out = write_html(html, tmp_path / "report.html")
    assert out.exists()
