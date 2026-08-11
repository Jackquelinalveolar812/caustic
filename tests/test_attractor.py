"""Ground truth for attractor dimension.

The load-bearing test is `test_lorenz_spectrum_gives_the_textbook_dimension`: the
Lorenz system at the classical parameters has a Kaplan-Yorke dimension of about
2.06 that is quoted in every dynamical-systems text, so an implementation that
gets it wrong is wrong against the literature and not merely against itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.attractor import (
    embedding_bound,
    kaplan_yorke_dimension,
    metric_entropy,
    spectrum_report,
)


def test_lorenz_spectrum_gives_the_textbook_dimension():
    """Lorenz at sigma=10, r=28, b=8/3 has lambda = (0.906, 0, -14.572).

    D_KY = 2 + (0.906 + 0) / 14.572 = 2.0622, the standard quoted value.
    """
    lam = np.array([0.906, 0.0, -14.572])
    assert kaplan_yorke_dimension(lam) == pytest.approx(2.0622, abs=1e-3)


def test_hand_computable_case():
    """lambda = (1, -2): j = 1, D_KY = 1 + 1/2."""
    assert kaplan_yorke_dimension(np.array([1.0, -2.0])) == pytest.approx(1.5)


def test_all_negative_collapses_to_zero():
    """Every direction contracts: the trajectory goes to a fixed point."""
    assert kaplan_yorke_dimension(np.array([-0.1, -1.0, -3.0])) == 0.0


def test_all_positive_fills_the_space():
    """No direction contracts: there is nothing to interpolate against."""
    assert kaplan_yorke_dimension(np.array([3.0, 2.0, 1.0])) == 3.0


def test_dimension_is_bounded_by_the_ambient_space():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lam = rng.normal(size=rng.integers(2, 40))
        d = kaplan_yorke_dimension(lam)
        assert 0.0 <= d <= len(lam) + 1e-12, f"{d} out of range for D={len(lam)}"


def test_input_order_does_not_matter():
    """A spectrum is a multiset."""
    rng = np.random.default_rng(1)
    lam = rng.normal(size=20)
    assert kaplan_yorke_dimension(lam) == pytest.approx(
        kaplan_yorke_dimension(rng.permutation(lam))
    )


def test_dimension_lies_between_the_positive_count_and_that_count_plus_one():
    """Structural: D_KY sits in [j, j+1) where j is where the partial sum turns."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        lam = np.sort(rng.normal(size=25))[::-1]
        d = kaplan_yorke_dimension(lam)
        if 0.0 < d < 25.0:
            assert np.floor(d) <= d < np.floor(d) + 1.0


def test_metric_entropy_is_the_positive_sum():
    lam = np.array([2.0, 0.5, -0.1, -9.0])
    assert metric_entropy(lam) == pytest.approx(2.5)
    assert metric_entropy(np.array([-1.0, -2.0])) == 0.0


def test_embedding_bound_is_takens():
    assert embedding_bound(2.0622) == 6  # ceil(4.1244) + 1 = 5 + 1
    assert embedding_bound(3.0) == 7  # ceil(6.0) + 1
    assert embedding_bound(0.0) == 1
    with pytest.raises(ValueError):
        embedding_bound(-1.0)


def test_scaling_the_spectrum_leaves_the_dimension_invariant():
    """D_KY is a ratio of exponents, so a change of time unit cannot move it.

    This is what makes it a candidate for a constant: it does not depend on how
    the trajectory was parameterised.
    """
    rng = np.random.default_rng(3)
    lam = rng.normal(size=30)
    base = kaplan_yorke_dimension(lam)
    for c in (0.1, 2.0, 1000.0):
        assert kaplan_yorke_dimension(c * lam) == pytest.approx(base, rel=1e-12)


def test_report_is_complete():
    r = spectrum_report(np.array([0.906, 0.0, -14.572]))
    assert set(r) == {
        "D",
        "lambda_max",
        "sum",
        "n_positive",
        "kaplan_yorke",
        "compression_ratio",
        "embedding_bound",
        "metric_entropy",
    }
    assert r["compression_ratio"] == pytest.approx(3.0 / 2.0622, abs=1e-3)


def test_empty_spectrum_is_rejected():
    with pytest.raises(ValueError):
        kaplan_yorke_dimension(np.array([]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
