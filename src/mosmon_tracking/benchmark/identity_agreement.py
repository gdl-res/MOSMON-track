"""Compare two trackers as two partitions of one shared detection set.

Without ground truth there is no HOTA and no AssA. But every arm of this
benchmark consumes an *identical* pool of detections, so two arms' outputs are
two labellings of the same observations. Comparing those labellings needs no
ground truth and no box matching at all -- it is a clustering-comparison
problem, and it measures exactly the thing association algorithms disagree
about: which observations belong to the same individual.

This is a measure of *disagreement*, not of correctness. Two arms can agree
perfectly and both be wrong. It is reported as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _contingency(a: np.ndarray, b: np.ndarray):
    """Dense contingency table. Only safe for small label sets -- see below."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    n = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(n, (ia, ib), 1)
    return n


def _sparse_contingency(a: np.ndarray, b: np.ndarray):
    """Nonzero contingency entries plus both marginals, without the dense table.

    A dense contingency is quadratic in the number of tracks, and on this corpus
    that is fatal: the densest video pairs 116,604 BoT-SORT tracks against
    124,961 ByteTrack tracks, whose dense table would be **117 GB**. The table is
    also almost entirely zeros -- 149,668 nonzeros, 0.001 % occupancy -- because
    a track can only co-occur with the handful of tracks that share its
    detections.

    Every quantity the three metrics need (pair counts, entropies, mutual
    information, the matching) depends only on the nonzero cells and the row and
    column sums, so working from this representation is **exact**, not an
    approximation. Verified against :func:`_contingency` in the tests.

    Returns ``(rows, cols, counts, row_sums, col_sums, total)`` with ``rows`` and
    ``cols`` indexing the sorted unique labels of ``a`` and ``b``.
    """
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    # Pair each observation's (row, col) and count identical pairs.
    order = np.lexsort((ib, ia))
    ia_s, ib_s = ia[order], ib[order]
    new = np.empty(ia_s.size, dtype=bool)
    new[0] = True
    np.not_equal(ia_s[1:], ia_s[:-1], out=new[1:])
    np.logical_or(new[1:], ib_s[1:] != ib_s[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, ia_s.size)).astype(np.int64)
    rows, cols = ia_s[starts], ib_s[starts]
    row_sums = np.bincount(rows, weights=counts, minlength=ua.size)
    col_sums = np.bincount(cols, weights=counts, minlength=ub.size)
    return rows, cols, counts, row_sums, col_sums, int(counts.sum())


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI between two labellings of the same observations. 1.0 = identical."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0:
        return float("nan")
    _, _, counts, row_sums, col_sums, total = _sparse_contingency(a, b)
    if total < 2:
        return float("nan")

    def comb2(x):
        x = np.asarray(x, dtype=float)
        return float((x * (x - 1) / 2.0).sum())

    sum_ij = comb2(counts)
    sum_i = comb2(row_sums)
    sum_j = comb2(col_sums)
    expected = sum_i * sum_j / comb2([float(total)])
    maximum = 0.5 * (sum_i + sum_j)
    denom = maximum - expected
    return float((sum_ij - expected) / denom) if denom else 1.0


def v_measure(a: np.ndarray, b: np.ndarray) -> float:
    """Harmonic mean of homogeneity and completeness. 1.0 = identical."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0:
        return float("nan")
    rows, cols, counts, row_sums, col_sums, total = _sparse_contingency(a, b)
    pij = counts / total
    pi = row_sums / total
    pj = col_sums / total

    def ent(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    hi, hj = ent(pi), ent(pj)
    mi = float((pij * np.log(pij / (pi[rows] * pj[cols]))).sum())
    hom = 1.0 if hi == 0 else mi / hi
    com = 1.0 if hj == 0 else mi / hj
    return float(0.0 if (hom + com) == 0 else 2 * hom * com / (hom + com))


def identity_f1(a: np.ndarray, b: np.ndarray) -> float:
    """IDF1-style score: optimal one-to-one match between the two ID sets.

    Mirrors how IDF1 scores a tracker against ground truth, but here neither side
    is ground truth -- it is a symmetric agreement between two arms.
    """
    from scipy.optimize import linear_sum_assignment

    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0:
        return float("nan")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows, cols, counts, row_sums, col_sums, _ = _sparse_contingency(a, b)
    n_r, n_c = row_sums.size, col_sums.size

    # The optimal one-to-one matching never needs a zero cell -- a zero edge adds
    # nothing to the total -- so it can be found on the nonzero subgraph alone.
    # That subgraph splits into connected components, and the optimum is the sum
    # of each component's optimum, which keeps every Hungarian call tiny. Solving
    # this densely would mean a 116,604 x 124,961 assignment problem.
    adj = coo_matrix((np.ones(rows.size), (rows, cols + n_r)),
                     shape=(n_r + n_c, n_r + n_c))
    adj = adj + adj.T
    _, labels = connected_components(adj.tocsr(), directed=False)

    matched = 0
    comp_of_edge = labels[rows]
    for comp in np.unique(comp_of_edge):
        sel = comp_of_edge == comp
        r, c, w = rows[sel], cols[sel], counts[sel]
        ur, ri = np.unique(r, return_inverse=True)
        uc, ci = np.unique(c, return_inverse=True)
        block = np.zeros((ur.size, uc.size), dtype=np.int64)
        block[ri, ci] = w
        rr, cc = linear_sum_assignment(-block)
        matched += int(block[rr, cc].sum())
    return float(2 * matched / (a.size + b.size))


def compare_assignments(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Agreement between two arms' ``assignments`` tables.

    Both tables map ``det_row`` (a row of the shared detection cache) to a
    ``track_id``. Only detections that *both* arms emitted can be compared;
    coverage records how large that intersection is, because an arm that simply
    drops more detections is not thereby "in agreement".
    """
    cols = ["det_row", "track_id"]
    if left is None or right is None or left.empty or right.empty:
        return {"n_common": 0, "ari": float("nan"), "v_measure": float("nan"),
                "identity_f1": float("nan"), "coverage_left": float("nan"),
                "coverage_right": float("nan")}
    m = (left[cols].drop_duplicates("det_row")
         .merge(right[cols].drop_duplicates("det_row"),
                on="det_row", suffixes=("_l", "_r")))
    if m.empty:
        return {"n_common": 0, "ari": float("nan"), "v_measure": float("nan"),
                "identity_f1": float("nan"), "coverage_left": 0.0,
                "coverage_right": 0.0}
    a = m["track_id_l"].to_numpy()
    b = m["track_id_r"].to_numpy()
    return {
        "n_common": int(len(m)),
        "ari": adjusted_rand_index(a, b),
        "v_measure": v_measure(a, b),
        "identity_f1": identity_f1(a, b),
        "coverage_left": float(len(m) / left["det_row"].nunique()),
        "coverage_right": float(len(m) / right["det_row"].nunique()),
    }


def pairwise_agreement(assignments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every unordered arm pair's agreement, one row per pair."""
    names = sorted(assignments)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"arm_a": a, "arm_b": b,
                         **compare_assignments(assignments[a], assignments[b])})
    return pd.DataFrame(rows)
