"""Write TRACKER_BENCHMARK_REPORT.md from the evaluation tables.

Every number is read back from a written table. The report also states, up
front and without softening, what this benchmark cannot answer -- the scope was
set by the absence of identity ground truth, and a reader who skims the tables
must not come away thinking they are looking at HOTA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_CAVEAT = """\
## What this benchmark does and does not measure

**Measured.** Every arm consumed a byte-identical pool of detections, produced by
one detector pass per video and replayed unchanged. Detection quality is
therefore not merely controlled, it is *invariant*: every difference reported
below is attributable to the association algorithm and to nothing else.

**Not measured.** There is no manually annotated identity ground truth, so this
report contains **no HOTA, DetA, AssA, IDF1, ID-switch count or fragmentation
count against truth**. Nothing here says which arm tracks *correctly*. The
metrics below are of two kinds, and neither is a substitute for MOT scoring:

* *Output statistics* -- track duration, fragment counts, duplicate proxies.
  A tracker that merges two larvae into one identity scores well on all of them.
* *Cross-arm identity agreement* -- how differently two arms group the same
  observations. Two arms can agree perfectly and both be wrong.

In consequence:

* "Association degrades with crowding" can be examined only through proxies.
* "AssA deteriorates faster than DetA" is **untestable here**: detection is
  constant by construction, which is a stronger control than the original
  design asked for, but it is not a measurement of DetA.
* **No arm can be called best.** The findings are about *how much* arms differ,
  *where* they differ, and whether that difference reaches the biology.
"""


def write_report(result: dict, out_dir: str | Path,
                 downstream: dict | None = None,
                 provenance: dict | None = None) -> Path:
    out = Path(out_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    s = result["summary"]
    metrics = result["metrics"]

    lines: list[str] = []
    a = lines.append
    a("# MOSMON-Track — multi-tracker benchmark\n")
    nv = s["n_videos"]
    a(f"{s['n_arms']} association algorithms over {nv} "
      f"video{'' if nv == 1 else 's'}, all replaying one fixed detector output.\n")
    a(f"Arms: {', '.join(s['arms'])}. Reference arm: `{s['reference_arm']}`.\n")
    lo, hi = s["density_range_detections_per_frame"]
    if lo and np.isfinite(lo) and np.isfinite(hi):
        span = f" ({hi / lo:.0f}x)" if hi > lo * 1.05 else ""
        a(f"Shared detection density spans **{lo:.1f}-{hi:.1f} "
          f"detections/frame**{span}.\n")
    a(_CAVEAT)

    a("\n## 1. Per-arm summary (mean over videos, 95% bootstrap CI)\n")
    a(_summary_table(result["overall"]))

    a("\n## 2. Cross-arm identity agreement\n")
    a("Each arm's mean adjusted Rand index against every other arm, over the "
      "shared detections. Lower means this arm's grouping is more distinctive; "
      "it does **not** mean more or less correct.\n")
    a(_agreement_table(metrics))

    a("\n## 3. Crowding dependence\n")
    a("Slope of each metric against log10 shared detection density. Density "
      "comes from the shared detection pool, so it is identical for every arm — "
      "an arm that fragments more cannot thereby appear to face a denser video.\n")
    a(_trend_table(result["trends"]))

    if not result["by_bin"].empty:
        a("\n### By density bin\n")
        a("Bins are quantiles of the shared detection density, fixed before any "
          "tracker output was examined.\n")
        a(_bin_table(result["by_bin"]))

    a("\n## 4. Paired comparison against the reference arm\n")
    a("Every arm sees every video, so comparisons are paired with the video as "
      "the replicate. Effect size (matched-pairs rank-biserial) and confidence "
      "interval carry the result; p-values are BH-corrected and secondary.\n")
    a(_paired_table(result["paired"], s["reference_arm"]))

    a("\n## 5. Duplicate-track proxy coherence\n")
    a("The published `duplicate_track_proxy` tests box geometry only — despite "
      "its docstring it never required the overlapping boxes to carry different "
      "track ids. `duplicate_id_proxy` is the identity-aware version, reported "
      "alongside; the published metric is unchanged so the species cohort is "
      "unaffected.\n\n"
      "**This is coherence, not validation.** With no ground truth there is no "
      "measured association error to correlate against, so §17's question — "
      "does the proxy predict real identity failure? — remains open.\n")
    a(_proxy_table(result["proxy"]))

    if downstream:
        a("\n## 6. Does tracker choice reach the biology?\n")
        a(_downstream_section(downstream))

    a("\n## 7. Runtime\n")
    a(_runtime_table(metrics))

    if provenance:
        a("\n## 8. Provenance\n")
        a("```json\n" + json.dumps(provenance, indent=2, default=str) + "\n```\n")

    path = out / "reports" / "TRACKER_BENCHMARK_REPORT.md"
    path.write_text("\n".join(x for x in lines if x), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown``, which needs ``tabulate``.
    The rest of this project is deliberately dependency-free and a report
    formatter is not worth breaking that for.
    """
    if df is None or df.empty:
        return "_no data_\n"
    cols = [str(c) for c in df.columns]
    rows = [[_cell(v) for v in rec] for rec in df.itertuples(index=False, name=None)]
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v).replace("|", "\\|")


def _summary_table(overall: pd.DataFrame) -> str:
    if overall.empty:
        return "_no data_\n"
    rows = []
    for _, r in overall.iterrows():
        row = {"Arm": r["arm"], "n videos": r.get("n_videos")}
        for m, label in [("median_track_duration_s", "Median duration (s)"),
                         ("p95_track_duration_s", "p95 duration (s)"),
                         ("fragments_per_object", "Fragments/object"),
                         ("duplicate_id_proxy", "Duplicate-ID proxy"),
                         ("identity_agreement_mean", "Mean ARI vs others")]:
            if f"{m}_mean" in r and np.isfinite(r[f"{m}_mean"]):
                row[label] = f"{r[f'{m}_mean']:.3f} [{r[f'{m}_lo']:.3f}, {r[f'{m}_hi']:.3f}]"
        rows.append(row)
    return _md(pd.DataFrame(rows))


def _agreement_table(metrics: pd.DataFrame) -> str:
    if "identity_agreement_mean" not in metrics:
        return "_no agreement data_\n"
    g = (metrics.groupby("arm")[["identity_agreement_mean", "identity_f1_mean"]]
         .mean().reset_index()
         .rename(columns={"arm": "Arm", "identity_agreement_mean": "Mean ARI",
                          "identity_f1_mean": "Mean identity-F1"}))
    return _md(g.round(4))


def _trend_table(trends: pd.DataFrame) -> str:
    if trends is None or trends.empty:
        return "_needs several videos spanning a density range_\n"
    t = trends.rename(columns={
        "arm": "Arm", "metric": "Metric",
        "slope_per_log10_density": "Slope / log10 density",
        "spearman_rho": "Spearman rho", "spearman_p": "p", "n_videos": "n"})
    keep = ["Arm", "Metric", "Slope / log10 density", "Spearman rho", "p", "n"]
    return _md(t[[c for c in keep if c in t]].round(4))


def _bin_table(by_bin: pd.DataFrame) -> str:
    cols = ["arm", "density_bin", "n_videos"]
    cols += [c for c in by_bin.columns if c.endswith("_mean")][:4]
    return _md(by_bin[[c for c in cols if c in by_bin]].round(4))


def _paired_table(paired: pd.DataFrame, ref: str) -> str:
    if paired is None or paired.empty:
        return "_needs several videos_\n"
    p = paired.rename(columns={
        "metric": "Metric", "arm_b": "Arm", "mean_diff": f"Δ vs {ref}",
        "ci_lo": "CI low", "ci_hi": "CI high",
        "rank_biserial": "Effect size", "q_value": "q"})
    keep = ["Metric", "Arm", f"Δ vs {ref}", "CI low", "CI high", "Effect size", "q"]
    return _md(p[[c for c in keep if c in p]].round(4))


def _proxy_table(proxy: pd.DataFrame) -> str:
    if proxy is None or proxy.empty:
        return "_needs several videos_\n"
    g = (proxy.groupby(["proxy", "target"])["spearman_rho"].mean().reset_index()
         .rename(columns={"proxy": "Proxy", "target": "Against",
                          "spearman_rho": "Mean Spearman rho (across arms)"}))
    return _md(g.round(4))


def _runtime_table(metrics: pd.DataFrame) -> str:
    cols = [c for c in ("association_fps", "association_seconds", "postprocess_seconds")
            if c in metrics]
    if not cols:
        return "_no timing recorded_\n"
    g = metrics.groupby("arm")[cols].median().reset_index()
    g = g.rename(columns={"arm": "Arm", "association_fps": "Association FPS",
                          "association_seconds": "Association s/video",
                          "postprocess_seconds": "Post-processing s/video"})
    return (_md(g.round(2))
            + "\nDetector time is excluded: detections are computed once and shared. "
              "Video decode is shared across all frame-consuming arms in a single pass.\n")


def _downstream_section(d: dict) -> str:
    parts = []
    if isinstance(d.get("cohorts"), pd.DataFrame) and not d["cohorts"].empty:
        parts.append("### Cohorts (§22)\n")
        parts.append("Tracker choice can move a video across the QC line, so the "
                     "eligible set is not constant. Both cohorts are reported; "
                     "never compare accuracies across differing cohorts.\n")
        parts.append(_md(d["cohorts"]))
    if isinstance(d.get("classification"), pd.DataFrame) and not d["classification"].empty:
        parts.append("### Species classification per arm (§23)\n")
        parts.append(_md(d["classification"].round(4)))
    if isinstance(d.get("stability"), pd.DataFrame) and not d["stability"].empty:
        parts.append("### Descriptor stability (§24)\n")
        parts.append("How much each video-level descriptor moves when the "
                     "association algorithm changes, over videos all arms share.\n")
        parts.append(_md(d["stability"].round(4)))
        counts = d["stability"]["stability"].value_counts().to_dict()
        parts.append("\nDescriptors by band: "
                     + ", ".join(f"{k} = {v}" for k, v in counts.items()) + "\n")
    return "\n".join(parts) if parts else "_downstream analysis not run_\n"
