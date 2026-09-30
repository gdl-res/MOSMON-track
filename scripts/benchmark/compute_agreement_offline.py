#!/usr/bin/env python3
"""Recover the cross-arm identity agreement the benchmark never computed.

``benchmark/run.py`` calls :func:`pairwise_agreement` over the arms present in
*that* invocation. The sequential driver runs one arm at a time, so ``results``
always held a single arm, there were no pairs, and every
``_agreement/<video>.parquet`` was written as an empty table -- 66 of them.

Nothing needs re-running: each arm's ``assignments.parquet`` maps ``det_row`` of
the shared detection cache to that arm's ``track_id``, and those files are
intact. This driver loads them, calls the **existing** ``pairwise_agreement``
unchanged, and writes the tables that were always meant to be there.

What the numbers mean, and do not mean
--------------------------------------
Every arm consumed an identical detection pool, so two arms' outputs are two
labellings of one observation set. Comparing them is a clustering-comparison
problem needing no ground truth:

* **ARI** -- chance-corrected agreement on which pairs of detections share an id.
* **V-measure** -- entropy-based agreement between the two partitions.
* **identity_f1** -- IDF1-style score under the optimal one-to-one id matching.
* **coverage_*** -- the share of each arm's detections in the compared
  intersection, so an arm that simply emits fewer detections cannot look
  "agreeable" by abstaining.

These measure **disagreement between two trackers, not correctness**. Neither
side is ground truth; both arms can agree perfectly and both be wrong. They are
not HOTA, DetA, AssA, IDF1-against-truth, or identity accuracy, and must never
be reported as such.

Example
-------
    python scripts/benchmark/compute_agreement_offline.py \
        --benchmark outputs/tracker_benchmark \
        --arms bytetrack,ocsort,botsort
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--arms", default="bytetrack,ocsort,botsort",
                    help="Comma-separated arms to compare (bytetrack_iou is "
                         "bit-identical to bytetrack and is excluded by default).")
    ap.add_argument("--out", default=None, help="Default: <benchmark>/_agreement")
    args = ap.parse_args(argv)

    import pandas as pd

    from mosmon_tracking.benchmark.identity_agreement import pairwise_agreement
    from mosmon_tracking.video_io import save_table

    bench = Path(args.benchmark)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out = Path(args.out) if args.out else bench / "_agreement"
    out.mkdir(parents=True, exist_ok=True)

    videos = sorted(set.intersection(*[
        {p.name for p in (bench / a).iterdir() if (p / "assignments.parquet").exists()}
        for a in arms
    ]))
    if not videos:
        print("no videos carry assignments.parquet for every requested arm")
        return 1
    print(f"{len(videos)} videos x {len(arms)} arms -> "
          f"{len(arms) * (len(arms) - 1) // 2} pairs each")

    stacked = []
    for i, vid in enumerate(videos, 1):
        assign = {a: pd.read_parquet(bench / a / vid / "assignments.parquet",
                                     columns=["det_row", "track_id"]) for a in arms}
        table = pairwise_agreement(assign)
        table.insert(0, "video_id", vid)
        save_table(table, out / f"{vid}.parquet")
        stacked.append(table)
        print(f"[{i}/{len(videos)}] {vid[:56]}  "
              + "  ".join(f"{r.arm_a[:4]}/{r.arm_b[:4]} ARI={r.ari:.3f}"
                          for r in table.itertuples()), flush=True)

    allv = pd.concat(stacked, ignore_index=True)
    save_table(allv, bench / "evaluation" / "identity_agreement_per_video.parquet")

    print("\n=== median across videos, per arm pair ===")
    summary = (allv.groupby(["arm_a", "arm_b"])[
        ["ari", "v_measure", "identity_f1", "coverage_left", "coverage_right", "n_common"]]
        .median().reset_index())
    print(summary.to_string(index=False))
    save_table(summary, bench / "evaluation" / "identity_agreement_summary.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
