import numpy as np
import pandas as pd
import pytest

from mosmon_tracking.visualization import render_heatmap_accumulation_video

cv2 = pytest.importorskip("cv2")


def _clean(n_frames=20):
    rows = []
    rng = np.random.default_rng(0)
    for f in range(n_frames):
        for _ in range(5):
            rows.append({
                "frame_idx": f,
                "cx_norm": float(rng.uniform(0, 1)),
                "cy_norm": float(rng.uniform(0, 1)),
            })
    return pd.DataFrame(rows)


def test_returns_none_on_empty():
    assert render_heatmap_accumulation_video(pd.DataFrame(), "/tmp/x.mp4") is None


def test_returns_none_on_missing_columns():
    df = pd.DataFrame({"frame_idx": [0, 1], "cx": [1.0, 2.0]})  # no cx_norm/cy_norm
    assert render_heatmap_accumulation_video(df, "/tmp/x.mp4") is None


def test_writes_a_nonempty_file(tmp_path):
    out = tmp_path / "heatmap_video.mp4"
    path = render_heatmap_accumulation_video(
        _clean(20), out, background=None, bins_x=32, bins_y=18, max_frames=None
    )
    assert path is not None and path.exists()
    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 1
    cap.release()


def test_max_frames_caps_output(tmp_path):
    out = tmp_path / "capped.mp4"
    render_heatmap_accumulation_video(
        _clean(50), out, background=None, bins_x=16, bins_y=9, max_frames=10
    )
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # 50 frames strided to ~<=10 outputs (+ forced last frame); allow slack for codec count.
    assert 1 <= n <= 12


def test_background_controls_output_size(tmp_path):
    out = tmp_path / "bg.mp4"
    bg = np.zeros((200, 400, 3), dtype=np.uint8)  # H=200, W=400
    render_heatmap_accumulation_video(
        _clean(5), out, background=bg, bins_x=16, bins_y=9, scale=0.5, max_frames=None
    )
    cap = cv2.VideoCapture(str(out))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert (w, h) == (200, 100)  # 400*0.5, 200*0.5
