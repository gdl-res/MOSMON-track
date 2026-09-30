"""Audit a detection cache before spending days replaying it.

Two questions, both of which cost far more to answer after the fact:

1. **Did `max_det` truncate anything the trackers would actually have used?**
   Ultralytics keeps the top-N detections by score, so a cap removes the weakest
   boxes first. That is harmless while the discarded boxes sit below the floor
   every tracker applies internally -- and a silent corruption of the crowding
   analysis as soon as it does not. The 2026-08 full evaluation ran at the
   default `max_det=300` and lost a median of 409 detections/frame on 99.7 % of
   the frames of its densest video; this check exists so that cannot recur.

2. **How much work is the replay stage actually going to be?** Cost scales with
   detections above the replay floor, not with frames, and the corpus spans a
   50-fold range in density.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True, help="Detection cache directory.")
    ap.add_argument("--replay-conf", type=float, default=0.10,
                    help="Floor the trackers will actually see (benchmark.replay_conf).")
    ap.add_argument("--headroom", type=float, default=0.9,
                    help="Flag a video whose usable peak exceeds this fraction of max_det.")
    ap.add_argument("--out", default=None, help="Optional CSV to write.")
    args = ap.parse_args(argv)

    from mosmon_tracking.benchmark.detections_io import apply_conf_floor

    d = Path(args.detections)
    metas = sorted(d.glob("*.meta.json"))
    if not metas:
        print(f"no detection caches under {d}")
        return 1

    rows = []
    for m in metas:
        meta = json.loads(m.read_text())
        parquet = d / f"{meta['video_id']}.parquet"
        if not parquet.exists():
            continue
        det = pd.read_parquet(parquet, columns=["processed_frame", "confidence"])
        max_det = meta.get("detector", {}).get("max_det")
        usable = apply_conf_floor(det, args.replay_conf, half=True)
        per_all = det.groupby("processed_frame").size()
        per_use = usable.groupby("processed_frame").size()
        peak_use = int(per_use.max()) if len(per_use) else 0
        rows.append({
            "video_id": meta["video_id"],
            "frames": meta["n_frames"],
            "max_det": max_det,
            "all_per_frame_median": float(per_all.median()),
            "all_per_frame_max": int(per_all.max()),
            "frames_at_cap": int((per_all >= max_det).sum()) if max_det else 0,
            "usable_per_frame_median": float(per_use.median()) if len(per_use) else 0.0,
            "usable_per_frame_max": peak_use,
            "usable_total": int(len(usable)),
            # The cap only corrupts the experiment if it reached into the pool
            # the trackers use. Below the floor, a truncated box is a box no arm
            # would have looked at.
            "cap_touched_usable": bool(max_det and peak_use >= max_det),
            "headroom_frac": (peak_use / max_det) if max_det else np.nan,
        })

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{len(t)} cached videos, replay floor {args.replay_conf}\n")
    cols = ["video_id", "frames", "all_per_frame_median", "usable_per_frame_median",
            "usable_per_frame_max", "frames_at_cap", "headroom_frac"]
    show = t[cols].copy()
    show["video_id"] = show["video_id"].str.slice(0, 44)
    print(show.sort_values("usable_per_frame_median", ascending=False).round(2).to_string(index=False))

    bad = t[t["cap_touched_usable"]]
    tight = t[(~t["cap_touched_usable"]) & (t["headroom_frac"] > args.headroom)]
    print(f"\ntotal usable detections: {t['usable_total'].sum():,} "
          f"over {t['frames'].sum():,} frames "
          f"({t['usable_total'].sum() / max(t['frames'].sum(), 1):.0f}/frame)")
    print(f"videos where max_det reached the usable pool: {len(bad)}")
    for _, r in bad.iterrows():
        print(f"  ! {r.video_id[:60]}  usable peak {r.usable_per_frame_max} >= max_det {r.max_det}")
    print(f"videos within {100*(1-args.headroom):.0f}% of the cap: {len(tight)}")
    for _, r in tight.iterrows():
        print(f"  ~ {r.video_id[:60]}  usable peak {r.usable_per_frame_max} of {r.max_det}")

    if args.out:
        t.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
