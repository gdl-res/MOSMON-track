"""model.max_det must reach Ultralytics' track() call.

It used to be read only by the detection-only path, so every tracking run fell back
to Ultralytics' default of 300 and silently truncated the densest frames.
"""
import sys
import types

from mosmon_tracking import yolo_tracker
from mosmon_tracking.config import Config
from mosmon_tracking.video_io import VideoInfo


def test_track_call_passes_max_det(monkeypatch, tmp_path):
    captured = {}

    class FakeYOLO:
        names = {0: "aedes aegypti"}

        def __init__(self, path):
            pass

        def track(self, **kwargs):
            captured.update(kwargs)
            return iter(())

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(yolo_tracker, "probe_video",
                        lambda p: VideoInfo(path=str(p), filename=p.name, width=64, height=48, fps=30.0))
    cfg = Config()
    cfg.model.device = "cpu"
    cfg.model.max_det = 1234

    yolo_tracker.run_full_frame_tracking(tmp_path / "v.mp4", tmp_path / "m.pt", cfg)

    assert captured["max_det"] == 1234


def test_default_max_det_is_above_ultralytics_default():
    assert Config().model.max_det > 300
