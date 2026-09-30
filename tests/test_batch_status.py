"""batch_status classifies expected runs without touching them (no YOLO/probe deps
for the counting logic — empty .mp4 files are enough for find_videos)."""
import json

from mosmon_tracking.batch import batch_status
from mosmon_tracking.config import Config


def _setup(tmp_path, n=4):
    vroot = tmp_path / "videos"
    vroot.mkdir()
    for i in range(n):
        (vroot / f"vid{i}.mp4").write_bytes(b"")
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    out = tmp_path / "batch"
    out.mkdir()
    return vroot, model, out


def _done(out, video_stem, model_stem="m"):
    d = out / f"{model_stem}__{video_stem}"
    d.mkdir(parents=True)
    (d / "video_summary.json").write_text(json.dumps({"behaviour": {}}), encoding="utf-8")


def _failed(out, video_stem, model_stem="m"):
    d = out / f"{model_stem}__{video_stem}"
    d.mkdir(parents=True)
    (d / "failure.json").write_text("{}", encoding="utf-8")


def test_counts_done_failed_pending(tmp_path):
    vroot, model, out = _setup(tmp_path, 4)
    _done(out, "vid0")
    _done(out, "vid1")
    _failed(out, "vid2")
    # vid3 left pending
    s = batch_status(vroot, [model], Config(), out)
    assert s["total_runs"] == 4
    assert s["done"] == 2
    assert s["failed"] == 1
    assert s["pending"] == 1
    assert s["percent_complete"] == 75.0  # done+failed over total
    assert "vid3.mp4" in s["pending_videos"]


def test_all_pending_when_empty(tmp_path):
    vroot, model, out = _setup(tmp_path, 3)
    s = batch_status(vroot, [model], Config(), out)
    assert s["done"] == 0 and s["pending"] == 3
