#!/usr/bin/env python3
"""Is the population-level species result specific to BoT-SORT?

Assembles every table for the tracker-robustness deliverable from products that
already exist on disk. Nothing here re-runs a tracker or the detector; the one
upstream step this depends on is the per-arm species analysis, produced by
``scripts/benchmark/run_downstream_arms.sh``.

Three DISTINCT arms only. ``bytetrack_iou`` is excluded throughout: it is
bit-identical to ``bytetrack`` on the frozen detection cache and is
not a fourth tracker.

Outputs land in ``outputs/tracker_robustness/`` by default.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

ARMS = ["bytetrack", "ocsort", "botsort"]
LABEL = {"bytetrack": "ByteTrack", "ocsort": "OC-SORT", "botsort": "BoT-SORT"}


def _sha(path: Path, cap: int = 64 * 1024 * 1024) -> str | None:
    import hashlib
    if not path.exists() or path.stat().st_size > cap:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def per_video_metrics(bench: Path) -> pd.DataFrame:
    """One row per (arm, video): timing, QC and track-summary quantities."""
    import numpy as np
    import pandas as pd

    rows = []
    for arm in ARMS:
        for mp in sorted(glob.glob(str(bench / f"_manifests_{arm}" / "*.json"))):
            m = json.loads(Path(mp).read_text())
            x = list(m["arms"].values())[0]
            vid = m["video_id"]
            run = bench / arm / vid
            vs = json.loads((run / "video_summary.json").read_text())
            qc = json.loads((run / "qc_report.json").read_text())
            beh, vi, ms = vs.get("behaviour", {}), vs.get("video_info", {}), vs.get("model_settings", {})
            ts = pd.read_parquet(run / "track_summary.parquet",
                                 columns=["duration_s", "quality_score"])
            nwd = qc.get("n_frames_with_detections")
            fc, stride = vi.get("frame_count"), (ms.get("frame_stride") or 1)
            cov = 100.0 * nwd / math.ceil(fc / stride) if (nwd and fc) else np.nan
            rows.append({
                "arm": arm, "video_id": vid,
                "frames_replayed": m["n_frames_replayed"],
                "n_detections": m["detections"]["n_detections"],
                "box_hash": m["detections"]["box_hash"],
                "association_seconds": x["timing"]["association_seconds"],
                "association_fps": x["timing"]["association_fps"],
                "n_tracks_clean": qc.get("n_unique_track_ids_clean"),
                "median_track_length_frames": qc.get("median_track_length"),
                "median_track_duration_s": float(ts.duration_s.median()),
                "mean_track_quality_score": float(ts.quality_score.mean()),
                "duplicate_track_proxy": qc.get("duplicate_track_proxy"),
                "duplicate_id_proxy": qc.get("duplicate_id_proxy"),
                "interpolation_fraction": qc.get("interpolation_fraction"),
                "large_jump_fraction": qc.get("large_jump_fraction"),
                "mean_active_tracks_per_frame": beh.get("mean_active_tracks_per_frame"),
                "coverage_pct": cov,
            })
    return pd.DataFrame(rows)


def paired_correlations(pv: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Across-video Spearman between arms, preserving the video pairing."""
    import pandas as pd
    from scipy import stats

    rows = []
    for met in metrics:
        w = pv.pivot_table(index="video_id", columns="arm", values=met, aggfunc="first")
        for i, a in enumerate(ARMS):
            for b in ARMS[i + 1:]:
                d = w[[a, b]].dropna()
                if len(d) < 3 or d[a].nunique() < 2 or d[b].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(d[a], d[b])
                rows.append({"metric": met, "arm_a": a, "arm_b": b,
                             "n_videos": len(d), "spearman_rho": float(rho),
                             "p_value": float(p)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "tracker_robustness"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "tracker_benchmark.yaml"))
    args = ap.parse_args(argv)

    from mosmon_tracking.benchmark.analyze_downstream_stability import (
        descriptor_stability,
        load_video_features,
    )
    from mosmon_tracking.config import load_config
    from mosmon_tracking.species_analysis import (
        _feature_matrix,
        classify_species,
        cv_predict,
        video_feature_columns,
    )

    bench = Path(args.benchmark)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    down = bench / "downstream"
    cfg = load_config(args.config)

    # ---- §1/§2 tracking outputs ------------------------------------------ #
    pv = per_video_metrics(bench)
    pv.to_csv(out / "tracker_per_video_metrics.csv", index=False)
    fp_ok = int(pv.groupby("video_id").box_hash.nunique().eq(1).sum())
    print(f"frozen-replay check: {fp_ok}/{pv.video_id.nunique()} videos share one box_hash across arms")

    MET = ["duplicate_track_proxy", "interpolation_fraction", "large_jump_fraction",
           "mean_active_tracks_per_frame", "median_track_duration_s",
           "mean_track_quality_score", "coverage_pct", "n_tracks_clean"]
    q = pv.groupby("arm")[MET].quantile([0.25, 0.5, 0.75]).unstack()
    tracking = pd.DataFrame({"arm": ARMS}).set_index("arm")
    for m in MET:
        tracking[f"{m}_median"] = [q.loc[a, (m, 0.5)] for a in ARMS]
        tracking[f"{m}_iqr_lo"] = [q.loc[a, (m, 0.25)] for a in ARMS]
        tracking[f"{m}_iqr_hi"] = [q.loc[a, (m, 0.75)] for a in ARMS]
    tracking.reset_index().to_csv(out / "tracker_qc_median_iqr.csv", index=False)
    paired_correlations(pv, MET).to_csv(out / "tracker_qc_paired_correlations.csv", index=False)

    # ---- §3 agreement ----------------------------------------------------- #
    agree = pd.read_parquet(bench / "evaluation" / "identity_agreement_per_video.parquet")
    ag = (agree.groupby(["arm_a", "arm_b"])[["ari", "v_measure", "identity_f1",
                                             "coverage_left", "coverage_right"]]
          .agg(["median", "min", "max"]).reset_index())
    ag.columns = ["_".join(c).strip("_") for c in ag.columns]
    ag.to_csv(out / "tracker_identity_agreement.csv", index=False)

    # ---- §4 downstream ---------------------------------------------------- #
    vt = {a: pd.read_parquet(down / a / "video_features.parquet") for a in ARMS}
    common = sorted(set.intersection(*[set(v["run"]) for v in vt.values()]))

    rows, preds = [], {}
    for a in ARMS:
        own = json.loads((down / a / "classification_results.json").read_text())
        t = vt[a][vt[a]["run"].isin(common)].sort_values("run").reset_index(drop=True)
        cols = video_feature_columns(t)
        res = classify_species(t, cols, cfg=cfg)
        X = _feature_matrix(t, cols)
        y = t["meta_species"].to_numpy()
        p, _, _ = cv_predict(X, y, t["run"].to_numpy(),
                             cfg.species_analysis.logistic_l2,
                             cfg.species_analysis.logistic_max_iter)
        preds[a] = pd.Series(p, index=t["run"])
        sub = pv[pv.arm == a]
        rows.append({
            "tracker": LABEL[a], "arm": a, "videos_total": int(sub.video_id.nunique()),
            "association_fps": float(sub.frames_replayed.sum() / sub.association_seconds.sum()),
            "clean_tracks_total": int(sub.n_tracks_clean.sum()),
            "median_track_duration_s": float(sub.median_track_duration_s.median()),
            "median_duplicate_track_proxy": float(sub.duplicate_track_proxy.median()),
            "median_interpolation_fraction": float(sub.interpolation_fraction.median()),
            "median_large_jump_fraction": float(sub.large_jump_fraction.median()),
            "own_cohort_videos": int(own.get("n_videos")),
            "own_cohort_lovo_ba": float(own["video_level"]["balanced_accuracy_lovo"]),
            "common_cohort_videos": len(t),
            "common_cohort_lovo_ba": float(res["video_level"]["balanced_accuracy_lovo"]),
            "common_cohort_eligible_tracks": int(t["n_tracks_used"].sum()),
            "density_only_ba": (res.get("density_only_baseline") or {}).get("balanced_accuracy_lovo"),
            "permutation_p": (res.get("permutation_null") or {}).get("p_value"),
            "chance_ba": res.get("chance_balanced_accuracy"),
        })
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(out / "tracker_robustness_results.csv", index=False)

    P = pd.DataFrame(preds)
    truth = vt["botsort"].set_index("run").loc[common, "meta_species"]
    P["truth"] = truth
    P.to_csv(out / "tracker_common_cohort_predictions.csv")
    agree_pred = {f"{a}_vs_{b}": int((P[a] == P[b]).sum())
                  for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}
    all_same = int((P[ARMS].nunique(axis=1) == 1).sum())

    # ---- §5 representation stability -------------------------------------- #
    vf = load_video_features({a: down / a for a in ARMS})
    st = descriptor_stability(vf, video_feature_columns(vt["botsort"]), key="run")
    st.to_csv(out / "tracker_representation_agreement.csv", index=False)

    # ---- LaTeX ------------------------------------------------------------ #
    tex = [r"% Auto-generated by scripts/benchmark/tracker_robustness_analysis.py",
           r"\begin{tabular}{lrrrrrrrr}", r"\hline",
           r"Tracker & Assoc.\ FPS & Clean tracks & Med.\ dur.\ (s) & Dup.\ proxy & "
           r"Interp.\ frac. & Jump frac. & Cohort $n$ & LOVO bal.\ acc. \\", r"\hline"]
    for r in main_tbl.itertuples():
        tex.append(f"{r.tracker} & {r.association_fps:.1f} & {r.clean_tracks_total:,} & "
                   f"{r.median_track_duration_s:.2f} & {r.median_duplicate_track_proxy:.3f} & "
                   f"{r.median_interpolation_fraction:.3f} & {r.median_large_jump_fraction:.4f} & "
                   f"{r.common_cohort_videos} & {r.common_cohort_lovo_ba:.3f} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    (out / "tracker_robustness_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # ---- provenance ------------------------------------------------------- #
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip() or None
    except Exception:
        commit = None
    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "benchmark_root": str(bench),
        "arms": ARMS,
        "excluded_arms": {"bytetrack_iou": "bit-identical to bytetrack (report §4.1)",
                          "deepocsort": "never run (~153 h; deferred by decision)"},
        "tracker_configs": {a: {"path": f"configs/tracker_{a}.yaml",
                                "sha256_16": _sha(REPO_ROOT / f"configs/tracker_{a}.yaml")}
                            for a in ARMS},
        "benchmark_config": {"path": args.config, "sha256_16": _sha(Path(args.config))},
        "analysis_parameters": {
            "min_track_duration_s": cfg.species_analysis.min_track_duration_s,
            "logistic_l2": cfg.species_analysis.logistic_l2,
            "n_permutations": cfg.species_analysis.n_permutations,
            "apply_qc_gate": cfg.species_analysis.apply_qc_gate,
            "max_duplicate_track_proxy": cfg.species_analysis.max_duplicate_track_proxy,
            "min_coverage_pct": cfg.species_analysis.min_coverage_pct,
            "n_features": int(len(video_feature_columns(vt["botsort"]))),
        },
        "frozen_replay_videos_with_identical_box_hash": fp_ok,
        "common_cohort_videos": common,
        "prediction_agreement_common_cohort": {**agree_pred, "all_three_identical": all_same,
                                               "n_videos": len(common)},
        "published_reference": {
            "source": "<mosmon_outputs>/species_analysis",
            "batch": "full_eval (configs/eval.yaml: conf 0.15, max_det unset -> default 300)",
            "lovo_ba": 0.8612554112554114, "n_videos": 45,
            "note": "different detection pool from the benchmark replay; not a like-for-like target",
        },
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n",
                                         encoding="utf-8")

    print("\n=== main table ===")
    print(main_tbl[["tracker", "association_fps", "clean_tracks_total",
                    "own_cohort_videos", "own_cohort_lovo_ba",
                    "common_cohort_videos", "common_cohort_lovo_ba"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\ncommon cohort {len(common)} videos; identical predictions on "
          f"{all_same}/{len(common)} videos across all three arms")
    r = st.spearman_mean
    print(f"descriptor stability: median rho {r.median():.3f}, "
          f"{(r >= 0.8).sum()}/{len(r)} at >=0.8")
    print(f"\nwrote {len(list(out.glob('*')))} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
