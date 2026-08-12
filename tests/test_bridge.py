"""Contracts for the shared-frame projection.

The load-bearing property is not that the projection reduces dimension -- anything
does that -- but that a RATIO computed after projection agrees across source
widths when the seed is shared, and that the agreement is a population property
rather than a per-item one. Both halves are asserted, including the limitation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.bridge import (
    CANONICAL_SEED,
    comparable_ratio,
    jl_distortion_bound,
    jl_matrix,
    project,
)


def test_same_seed_gives_an_identical_matrix():
    """The whole point: two runs share a frame, not merely a dimension."""
    a = jl_matrix(512, 64, seed=CANONICAL_SEED)
    b = jl_matrix(512, 64, seed=CANONICAL_SEED)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_matrices():
    a = jl_matrix(512, 64, seed=1)
    b = jl_matrix(512, 64, seed=2)
    assert not np.allclose(a, b)


def test_projection_preserves_norms_in_expectation():
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(300):
        x = rng.normal(size=512)
        ratios.append(np.linalg.norm(project(x, 128)) / np.linalg.norm(x))
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.02)


def test_projection_preserves_pairwise_distances_within_the_bound():
    rng = np.random.default_rng(1)
    n, k = 40, 256
    eps = jl_distortion_bound(n, k)
    pts = rng.normal(size=(n, 512))
    proj = project(pts, k)
    for i in range(n):
        for j in range(i + 1, n):
            d0 = np.linalg.norm(pts[i] - pts[j]) ** 2
            d1 = np.linalg.norm(proj[i] - proj[j]) ** 2
            assert (1 - eps) * d0 <= d1 <= (1 + eps) * d0


def test_the_measured_cross_width_agreement():
    """A true ratio of 1.33 recovered within 1% across three source widths."""
    rng = np.random.default_rng(2)
    for k in (64, 256):
        means = []
        for D in (896, 2048, 4096):
            vals = []
            for _ in range(300):
                c = rng.normal(size=D)
                c /= np.linalg.norm(c)
                g = rng.normal(size=D)
                g = 1.33 * g / np.linalg.norm(g)
                vals.append(comparable_ratio(g, c, k))
            means.append(float(np.mean(vals)))
        spread = (max(means) - min(means)) / float(np.mean(means))
        assert spread < 0.01, f"k={k} spread {spread:.4f}"
        assert all(abs(m - 1.33) < 0.03 for m in means), means


def test_per_item_error_is_large_which_is_the_stated_limitation():
    """The projection carries population claims, not individual ones.

    Asserting the limitation keeps a future reader from quoting a single
    projected measurement as if it were exact.
    """
    rng = np.random.default_rng(3)
    errs = []
    for _ in range(400):
        c = rng.normal(size=896)
        c /= np.linalg.norm(c)
        g = rng.normal(size=896)
        g = 1.33 * g / np.linalg.norm(g)
        errs.append(abs(comparable_ratio(g, c, 64) - 1.33) / 1.33)
    assert np.mean(errs) > 0.03, "per-item error is not negligible and must not be sold as such"


def test_distortion_bound_shrinks_as_target_dimension_grows():
    bounds = [jl_distortion_bound(1000, k) for k in (256, 1024, 4096, 16384)]
    assert all(a > b for a, b in zip(bounds, bounds[1:]))


def test_useless_target_dimension_returns_infinity_not_a_false_promise():
    assert jl_distortion_bound(10**6, 4) == float("inf")


def test_expanding_the_dimension_is_rejected():
    with pytest.raises(ValueError):
        jl_matrix(64, 512)


def test_mismatched_shapes_and_zero_denominators_are_rejected():
    with pytest.raises(ValueError):
        comparable_ratio(np.ones(64), np.ones(32), 16)
    with pytest.raises(ValueError):
        comparable_ratio(np.ones(64), np.zeros(64), 16)
