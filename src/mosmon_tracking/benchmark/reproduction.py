"""Validate an arm against a previously produced track table (§6.5).

Before any arm is trusted, the BoT-SORT arm must be shown to reproduce the
2026-08 full evaluation it stands in for. If it does not, the benchmark is
measuring the harness rather than the association algorithms.

Boxes are matched **per frame by nearest neighbour within a tolerance**, not by
exact coordinate equality. That is not a loosening to make the test pass: the
trackers emit Kalman-filtered states rather than the raw detections they
consumed (measured ~0.8 px from the input box), so two runs that made every
identical decision still differ in the last bits once global motion compensation
reads independently decoded pixels. The distribution of the residual is reported
alongside the match rate, so a real divergence cannot hide behind the tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .identity_agreement import adjusted_rand_index, identity_f1, v_measure

#: Two boxes are the same box if their top-left corners are within this many
#: pixels. Measured divergence between an identical replay and the reference is
#: ~0.08 px median / 0.20 px max, against larvae ~100 px across.
BOX_TOL_PX = 1.0


def _match_frame(a: np.ndarray, b: np.ndarray, tol: float):
    """Greedy nearest-neighbour matching of two box sets in one frame."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    from scipy.optimize import linear_sum_assignment

    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(d)
    keep = d[ri, ci] <= tol
    return ri[keep], ci[keep], d[ri, ci][keep]


def compare_track_tables(new: pd.DataFrame, ref: pd.DataFrame,
                         tol: float = BOX_TOL_PX) -> dict:
    """Compare a replayed track table against a reference one.

    Reports the two questions separately, because they fail for different
    reasons: did the arm see the same boxes, and did it group them the same way?
    """
    out = {"n_new": int(len(new)), "n_ref": int(len(ref)), "tol_px": float(tol)}
    if new.empty or ref.empty:
        return {**out, "box_recall": float("nan"), "box_precision": float("nan"),
                "exact_id_match": float("nan"), "ari": float("nan"),
                "v_measure": float("nan"), "identity_f1": float("nan")}

    ids_new, ids_ref, deltas = [], [], []
    frames = sorted(set(new["frame_idx"]) | set(ref["frame_idx"]))
    gn = {f: g for f, g in new.groupby("frame_idx")}
    gr = {f: g for f, g in ref.groupby("frame_idx")}
    n_new_only = n_ref_only = 0
    for f in frames:
        A = gn.get(f)
        B = gr.get(f)
        if A is None or B is None:
            n_new_only += 0 if A is None else len(A)
            n_ref_only += 0 if B is None else len(B)
            continue
        pa = A[["x1", "y1"]].to_numpy(dtype=np.float64)
        pb = B[["x1", "y1"]].to_numpy(dtype=np.float64)
        ri, ci, dd = _match_frame(pa, pb, tol)
        ids_new.append(A["track_id"].to_numpy()[ri])
        ids_ref.append(B["track_id"].to_numpy()[ci])
        deltas.append(dd)
        n_new_only += len(A) - len(ri)
        n_ref_only += len(B) - len(ci)

    a = np.concatenate(ids_new) if ids_new else np.zeros(0, int)
    b = np.concatenate(ids_ref) if ids_ref else np.zeros(0, int)
    d = np.concatenate(deltas) if deltas else np.zeros(0, float)

    out["n_matched"] = int(a.size)
    out["n_boxes_only_in_new"] = int(n_new_only)
    out["n_boxes_only_in_ref"] = int(n_ref_only)
    out["box_recall"] = float(a.size / len(ref))
    out["box_precision"] = float(a.size / len(new))
    out["box_delta_px_median"] = float(np.median(d)) if d.size else float("nan")
    out["box_delta_px_p95"] = float(np.percentile(d, 95)) if d.size else float("nan")
    out["box_delta_px_max"] = float(d.max()) if d.size else float("nan")
    if a.size == 0:
        return {**out, "exact_id_match": float("nan"), "ari": float("nan"),
                "v_measure": float("nan"), "identity_f1": float("nan")}
    out["exact_id_match"] = float((a == b).mean())
    out["ari"] = adjusted_rand_index(a, b)
    out["v_measure"] = v_measure(a, b)
    out["identity_f1"] = identity_f1(a, b)
    return out


def verdict(cmp: dict, box_min: float = 0.995, id_min: float = 0.995) -> tuple[bool, str]:
    """Turn a comparison into a pass/fail with a one-line reason."""
    if not np.isfinite(cmp.get("box_recall", np.nan)):
        return False, "no comparable boxes"
    if cmp["box_recall"] < box_min or cmp["box_precision"] < box_min:
        return False, (f"detection pools differ: recall {cmp['box_recall']:.4f}, "
                       f"precision {cmp['box_precision']:.4f}")
    if cmp["ari"] < id_min:
        return False, (f"same boxes but different grouping: ARI {cmp['ari']:.4f}, "
                       f"exact-id {cmp['exact_id_match']:.4f}")
    return True, (f"reproduced: boxes {cmp['box_recall']:.4f}/{cmp['box_precision']:.4f}, "
                  f"ARI {cmp['ari']:.4f}, exact-id {cmp['exact_id_match']:.4f}, "
                  f"residual {cmp['box_delta_px_median']:.3f} px median / "
                  f"{cmp['box_delta_px_max']:.3f} px max")
