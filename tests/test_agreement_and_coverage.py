"""Guards for the two changes the tracker-robustness analysis depended on.

Both are exactness claims, so both are tested against the thing they replaced
rather than against hand-written expectations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from mosmon_tracking.benchmark.identity_agreement import (
    _contingency,
    _sparse_contingency,
    adjusted_rand_index,
    identity_f1,
    v_measure,
)
from mosmon_tracking.species_analysis import _coverage_pct


# --------------------------------------------------------------------------- #
# Sparse contingency: exact, not approximate
# --------------------------------------------------------------------------- #
def _ref_ari(a, b):
    n = _contingency(a, b).astype(float)
    c2 = lambda x: (x * (x - 1) / 2.0).sum()  # noqa: E731
    si, sj, sij = c2(n.sum(1)), c2(n.sum(0)), c2(n)
    e = si * sj / c2(np.array([n.sum()]))
    m = 0.5 * (si + sj)
    return float((sij - e) / (m - e)) if (m - e) else 1.0


def _ref_v(a, b):
    n = _contingency(a, b).astype(float)
    pij = n / n.sum()
    pi, pj = pij.sum(1), pij.sum(0)
    ent = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())  # noqa: E731
    hi, hj = ent(pi), ent(pj)
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / np.outer(pi, pj)[nz])).sum())
    hom = 1.0 if hi == 0 else mi / hi
    com = 1.0 if hj == 0 else mi / hj
    return float(0.0 if (hom + com) == 0 else 2 * hom * com / (hom + com))


def _ref_f1(a, b):
    n = _contingency(a, b)
    r, c = linear_sum_assignment(-n)
    return float(2 * int(n[r, c].sum()) / (a.size + b.size))


@pytest.mark.parametrize("seed", range(25))
def test_sparse_metrics_match_the_dense_reference(seed):
    """The sparse path must reproduce the dense one bit for bit.

    A dense contingency is quadratic in track count and reached 117 GB on the
    real corpus, so the metrics were re-expressed over nonzero cells. That is
    only legitimate if it changes nothing.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 400))
    a = rng.integers(0, int(rng.integers(1, 40)), n)
    b = rng.integers(0, int(rng.integers(1, 40)), n)
    for new, ref in ((adjusted_rand_index, _ref_ari), (v_measure, _ref_v),
                     (identity_f1, _ref_f1)):
        got, want = new(a, b), ref(a, b)
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9)


def test_identical_labellings_agree_perfectly():
    a = np.random.default_rng(0).integers(0, 50, 2000)
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert v_measure(a, a) == pytest.approx(1.0)
    assert identity_f1(a, a) == pytest.approx(1.0)


def test_sparse_contingency_marginals():
    a = np.array([0, 0, 1, 1, 1, 2])
    b = np.array([5, 5, 5, 7, 7, 7])
    rows, cols, counts, rs, cs, total = _sparse_contingency(a, b)
    dense = _contingency(a, b)
    assert total == a.size
    assert np.array_equal(rs, dense.sum(axis=1))
    assert np.array_equal(cs, dense.sum(axis=0))
    rebuilt = np.zeros_like(dense)
    rebuilt[rows, cols] = counts
    assert np.array_equal(rebuilt, dense)


# --------------------------------------------------------------------------- #
# Coverage fallback
# --------------------------------------------------------------------------- #
def test_coverage_prefers_the_stored_value():
    assert _coverage_pct({"pct_frames_with_detections": 99.5}, {}, {}, {}) == 99.5


def test_coverage_is_recomputed_when_the_replay_never_wrote_it():
    """The benchmark replay omits the key, which emptied every arm's cohort."""
    vi = {"frame_count": 23389}
    ms = {"frame_stride": 2}
    qc_full = {"n_frames_with_detections": 11678}
    got = _coverage_pct(qc_full, {}, vi, ms)
    assert got == pytest.approx(100.0 * 11678 / math.ceil(23389 / 2))


@pytest.mark.parametrize("qc_full,vi,ms", [
    ({}, {"frame_count": 100}, {"frame_stride": 2}),          # no detection count
    ({"n_frames_with_detections": 10}, {}, {"frame_stride": 2}),  # no frame count
])
def test_coverage_returns_none_when_it_cannot_be_derived(qc_full, vi, ms):
    assert _coverage_pct(qc_full, {}, vi, ms) is None
