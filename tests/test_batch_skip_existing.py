"""skip_existing must resume a batch without touching finished bundles or YOLO.

These tests pre-create finished output folders so batch_track skips every pair
and never imports ultralytics — keeping them weight-free and fast.
"""
import json

from mosmon_tracking.batch import batch_track
from mosmon_tracking.config import Config


def _cfg():
    cfg = Config()
    cfg.fair.enabled = False
    return cfg


def _videos(tmp_path, n=3):
    vroot = tmp_path / "videos"
    vroot.mkdir()
    for i in range(n):
        (vroot / f"vid{i}.mp4").write_bytes(b"")  # find_videos only globs by suffix
    return vroot


def _finished(out, model_stem, video_stem):
    d = out / f"{model_stem}__{video_stem}"
    d.mkdir(parents=True)
    (d / "video_summary.json").write_text(json.dumps({"behaviour": {}}), encoding="utf-8")


def test_all_finished_are_skipped(tmp_path):
    vroot = _videos(tmp_path, 3)
    out = tmp_path / "batch"
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    for i in range(3):
        _finished(out, model.stem, f"vid{i}")

    index = batch_track(vroot, [model], _cfg(), out, skip_existing=True)
    assert len(index) == 3
    assert (index["status"] == "skipped").all()


def test_skip_existing_false_does_not_skip(tmp_path):
    # With skip disabled, a finished bundle would be reprocessed -> YOLO import,
    # which fails on the empty .pt: that surfaces as status="failed", not "skipped".
    vroot = _videos(tmp_path, 1)
    out = tmp_path / "batch"
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    _finished(out, model.stem, "vid0")

    index = batch_track(vroot, [model], _cfg(), out, skip_existing=False)
    assert (index["status"] == "skipped").sum() == 0
