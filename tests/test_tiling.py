import numpy as np

from mosmon_tracking.tiling import agnostic_nms, remap_boxes, tile_windows


def test_tile_windows_covers_non_divisible_frame():
    # 5312 is not a multiple of the 640 tile / 576 stride -> needs an edge-snapped window.
    W, H, tile, overlap = 5312, 2988, 640, 0.10
    wins = tile_windows(W, H, tile, overlap)

    xs = sorted({x0 for (x0, _y0, _x1, _y1) in wins})
    ys = sorted({y0 for (_x0, y0, _x1, _y1) in wins})
    # stride = int(640 * 0.9) = 576
    assert xs[1] - xs[0] == 576
    # every window is exactly one tile wide/tall (edge windows are snapped, not shrunk)
    assert all((x1 - x0) == tile and (y1 - y0) == tile for (x0, y0, x1, y1) in wins)
    # full coverage: the last window reaches the right/bottom edge
    assert max(x1 for (_x0, _y0, x1, _y1) in wins) == W
    assert max(y1 for (_x0, _y0, _x1, y1) in wins) == H
    # grid is the product of unique starts (deduplicated)
    assert len(wins) == len(xs) * len(ys)


def test_tile_windows_tile_larger_than_frame():
    wins = tile_windows(300, 200, 640, 0.10)
    assert wins == [(0, 0, 300, 200)]


def test_tile_windows_overlap_width_matches_setting():
    wins = tile_windows(2000, 640, 640, 0.25)
    xs = sorted({x0 for (x0, _y0, _x1, _y1) in wins})
    # 25% overlap -> stride 480 -> interior overlap of 640-480 = 160 px
    assert xs[1] - xs[0] == 480


def test_remap_boxes_roundtrip():
    local = np.array([[10.0, 20.0, 50.0, 80.0]])
    out = remap_boxes(local, x0=100, y0=200)
    assert out.tolist() == [[110.0, 220.0, 150.0, 280.0]]
    # empty input stays a well-shaped (0, 4) array
    assert remap_boxes(np.empty((0, 4)), 5, 5).shape == (0, 4)


def test_agnostic_nms_collapses_seam_duplicate_keeps_distinct():
    # Two near-identical boxes (same larva seen in adjacent tiles) + one far away.
    boxes = np.array([
        [100, 100, 140, 140],   # larva A (tile 1)
        [101, 101, 141, 141],   # larva A again (tile 2 overlap) -> should be suppressed
        [500, 500, 540, 540],   # larva B, distinct
    ], dtype=float)
    scores = np.array([0.9, 0.85, 0.8])
    keep = sorted(agnostic_nms(boxes, scores, iou_thr=0.5).tolist())
    assert keep == [0, 2]  # one of the duplicate pair + the distinct box


def test_agnostic_nms_empty():
    assert agnostic_nms(np.empty((0, 4)), np.empty((0,)), 0.5).shape == (0,)
