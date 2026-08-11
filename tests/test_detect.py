"""Ground truth for the scorers and the evaluation metric.

The AUROC implementation is tested hardest, because every conclusion in the
detection arm is a comparison of AUROC values. An implementation that mishandles
ties silently inflates any score with a discrete or saturating distribution, and
`max_softmax_score` saturates near zero on confident positions, so ties are the
common case rather than the corner case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.detect import (
    MahalanobisScorer,
    PCAScorer,
    auroc,
    auroc_ci,
    logit_margin,
    max_softmax_score,
    predictive_entropy,
)


# --- AUROC -----------------------------------------------------------------


def test_perfect_separation_is_one():
    assert auroc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1])) == 1.0


def test_perfectly_inverted_is_zero():
    assert auroc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([0, 0, 1, 1])) == 0.0


def test_all_ties_is_one_half():
    """Every score identical carries no information. A tie-blind implementation
    returns 1.0 or 0.0 here depending on sort order."""
    assert auroc(np.ones(10), np.array([0, 1] * 5)) == pytest.approx(0.5)


def test_partial_ties_are_averaged():
    """Scores (0, 1, 1, 2) with labels (0, 0, 1, 1).

    The tied pair splits: one concordant, one tied at half. Enumerating the four
    (negative, positive) pairs gives 0<1, 0<2, 1==1 -> 0.5, 1<2, so
    (1 + 1 + 0.5 + 1) / 4 = 0.875.
    """
    assert auroc(np.array([0.0, 1.0, 1.0, 2.0]), np.array([0, 0, 1, 1])) == pytest.approx(0.875)


def test_single_class_returns_one_half_not_nan():
    """Undefined, but a NaN would propagate silently into a reported table."""
    assert auroc(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])) == 0.5
    assert auroc(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0])) == 0.5


def test_auroc_matches_the_brute_force_pair_count():
    """Cross-check against the definition on random data."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(6, 40))
        s = rng.integers(0, 5, n).astype(float)  # small range forces ties
        y = rng.integers(0, 2, n).astype(bool)
        if y.all() or (~y).all():
            continue
        pos, neg = s[y], s[~y]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        assert auroc(s, y) == pytest.approx(brute, abs=1e-12)


def test_random_scores_sit_near_one_half():
    rng = np.random.default_rng(1)
    vals = [
        auroc(rng.normal(size=400), rng.integers(0, 2, 400).astype(bool)) for _ in range(30)
    ]
    assert abs(np.mean(vals) - 0.5) < 0.03


def test_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300).astype(bool)
    s = rng.normal(size=300) + y * 0.8
    point, lo, hi = auroc_ci(s, y, n_boot=400, seed=0)
    assert lo <= point <= hi
    assert hi - lo > 0.0


# --- confidence baselines --------------------------------------------------


def test_max_softmax_is_zero_for_a_one_hot_and_maximal_for_uniform():
    logits = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got = max_softmax_score(logits)
    assert got[0] == pytest.approx(0.0, abs=1e-12)
    assert got[1] == pytest.approx(1.0 - 1.0 / 3.0)


def test_entropy_is_zero_for_a_one_hot_and_log_k_for_uniform():
    logits = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got = predictive_entropy(logits)
    assert got[0] == pytest.approx(0.0, abs=1e-9)
    assert got[1] == pytest.approx(np.log(3.0))


def test_confidence_scores_are_shift_invariant():
    """Adding a constant to every logit cannot change the distribution."""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(7, 50))
    for fn in (max_softmax_score, predictive_entropy, logit_margin):
        assert np.allclose(fn(logits), fn(logits + 37.0))


def test_logit_margin_is_negated_so_higher_means_less_certain():
    logits = np.array([[5.0, 1.0, 0.0], [5.0, 4.9, 0.0]])
    got = logit_margin(logits)
    assert got[0] < got[1], "the near-tie must score higher"


# --- distance baselines ----------------------------------------------------


def test_mahalanobis_is_zero_at_the_mean_and_grows_outward():
    rng = np.random.default_rng(4)
    H = rng.normal(size=(400, 8))
    sc = MahalanobisScorer(shrinkage=0.05).fit(H)
    assert sc.score(H.mean(axis=0)[None, :])[0] == pytest.approx(0.0, abs=1e-8)
    near = sc.score(H.mean(axis=0)[None, :] + 0.1)[0]
    far = sc.score(H.mean(axis=0)[None, :] + 10.0)[0]
    assert far > near


def test_mahalanobis_survives_more_dimensions_than_samples():
    """D > n makes the sample covariance singular; shrinkage is why this works."""
    rng = np.random.default_rng(5)
    H = rng.normal(size=(20, 64))
    s = MahalanobisScorer(shrinkage=0.2).fit(H).score(rng.normal(size=(5, 64)))
    assert np.all(np.isfinite(s)) and np.all(s >= 0)


def test_pca_reconstruction_is_zero_on_an_exactly_low_rank_set():
    rng = np.random.default_rng(6)
    basis = np.linalg.qr(rng.normal(size=(16, 16)))[0][:, :3]
    H = rng.normal(size=(100, 3)) @ basis.T
    s = PCAScorer(k=3).fit(H).score(H)
    assert np.max(s) < 1e-16


def test_pca_flags_a_point_off_the_subspace():
    rng = np.random.default_rng(7)
    basis = np.linalg.qr(rng.normal(size=(16, 16)))[0]
    H = rng.normal(size=(200, 3)) @ basis[:, :3].T
    sc = PCAScorer(k=3).fit(H)
    assert sc.score((H[0] + 5.0 * basis[:, 9])[None, :])[0] > 1.0


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        MahalanobisScorer(shrinkage=1.5)
    with pytest.raises(ValueError):
        PCAScorer(k=0)
    with pytest.raises(RuntimeError):
        MahalanobisScorer().score(np.zeros((2, 3)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
