"""Each theorem checked against a construction where the answer is known.

Proofs live in `caustic/theorems.py`. These are the executable witnesses: every
statement is verified on an instance built so the expected value can be written
down by hand, plus randomized stress wherever the theorem asserts an inequality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.theorems import (
    orbit_error_bound,
    path_integral_change,
    pooling_recovery_bound,
    volume_decay_rate,
    winding_witness,
)


# --- Theorem 1: Orbit Error Bound (topology) -------------------------------


def test_t1_measured_capital_case():
    assert orbit_error_bound(20, 15) == 5


def test_t1_total_pooling_certifies_all_but_one():
    assert orbit_error_bound(20, 1) == 19


def test_t1_full_separation_certifies_nothing():
    assert orbit_error_bound(20, 20) == 0


def test_t1_never_exceeds_the_true_error_count():
    """The inequality, against ground truth on 2000 random injective instances."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(2, 25))
        truth = np.arange(n)
        pred = rng.integers(0, max(2, n // 2), n)
        bound = orbit_error_bound(n, len(np.unique(pred)))
        actual = int((pred != truth).sum())
        assert bound <= actual, f"n={n} bound={bound} actual={actual}"


def test_t1_is_attained_and_therefore_tight():
    """Two orbits of two, one correct member each: bound 2, actual 2."""
    truth = np.array([0, 1, 2, 3])
    pred = np.array([0, 0, 2, 2])
    assert orbit_error_bound(4, len(np.unique(pred))) == 2
    assert int((pred != truth).sum()) == 2


def test_t1_rejects_impossible_partitions():
    with pytest.raises(ValueError):
        orbit_error_bound(5, 6)
    with pytest.raises(ValueError):
        orbit_error_bound(5, 0)


# --- Theorem 2: Pooling Recovery Bound (game theory) -----------------------


def test_t2_singleton_block_permits_full_recovery():
    assert pooling_recovery_bound(1) == 1.0


def test_t2_measured_total_pooling_case():
    assert pooling_recovery_bound(20) == pytest.approx(0.05)


def test_t2_no_constant_decoder_beats_the_bound():
    """Exhaustive over every constant decoder on a pooled block."""
    rng = np.random.default_rng(1)
    for k in (2, 3, 5, 8):
        entities = np.arange(k)
        for guess in range(k):
            assert float(np.mean(entities == guess)) <= pooling_recovery_bound(k) + 1e-12
        trials = [float(np.mean(entities == int(rng.integers(0, k)))) for _ in range(200)]
        assert max(trials) <= pooling_recovery_bound(k) + 1e-12


def test_t2_is_monotone_in_block_size():
    vals = [pooling_recovery_bound(k) for k in (1, 2, 4, 8, 16)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_t2_rejects_empty_blocks():
    with pytest.raises(ValueError):
        pooling_recovery_bound(0)


# --- Theorem 3: Zero Coupling Implies Pooling (differential geometry) ------


def test_t3_zero_gradient_gives_zero_change():
    h1, h2 = np.zeros(8), np.ones(8)
    assert path_integral_change(lambda h: np.zeros(8), h1, h2) == pytest.approx(0.0, abs=1e-12)


def test_t3_recovers_the_true_difference_for_a_linear_logit():
    """z(h) = w.h has gradient w, so the integral must equal w.(h2 - h1)."""
    rng = np.random.default_rng(2)
    w = rng.normal(size=6)
    h1, h2 = rng.normal(size=6), rng.normal(size=6)
    got = path_integral_change(lambda h: w, h1, h2)
    assert got == pytest.approx(float(w @ (h2 - h1)), rel=1e-10)


def test_t3_recovers_the_true_difference_for_a_quadratic_logit():
    """z(h) = 0.5 h.A h with symmetric A has gradient A h."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 5))
    A = A + A.T
    h1, h2 = rng.normal(size=5), rng.normal(size=5)
    expected = 0.5 * (h2 @ A @ h2) - 0.5 * (h1 @ A @ h1)
    assert path_integral_change(lambda h: A @ h, h1, h2) == pytest.approx(expected, rel=1e-6)


def test_t3_vanishing_coupling_forces_identical_logits():
    """The hand-off to Theorem 1: equal logits mean one orbit."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        h1, h2 = rng.normal(size=10), rng.normal(size=10)
        assert path_integral_change(lambda h: np.zeros(10), h1, h2) == pytest.approx(0.0, abs=1e-12)


def test_t3_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        path_integral_change(lambda h: h, np.zeros(3), np.zeros(4))


# --- Theorem 4: Dissipative Pooling (chaos theory) -------------------------


def test_t4_measured_token_product_contracts():
    """Per-step exponent sum of -226.74, measured at block 3."""
    assert volume_decay_rate(np.array([-226.74]), 1) == pytest.approx(-226.74)
    assert volume_decay_rate(np.array([-226.74]), 10) == pytest.approx(-2267.4)


def test_t4_matches_the_log_determinant_of_an_explicit_map():
    """For a linear map the exponent sum is log|det|, exactly."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 6))
    exponents = np.log(np.linalg.svd(A, compute_uv=False))
    logdet = float(np.linalg.slogdet(A)[1])
    assert volume_decay_rate(exponents, 1) == pytest.approx(logdet, rel=1e-9)
    assert volume_decay_rate(exponents, 7) == pytest.approx(7 * logdet, rel=1e-9)


def test_t4_volume_actually_shrinks_under_iteration():
    A = np.diag([0.5, 0.5, 2.0])
    exponents = np.log(np.abs(np.diag(A)))
    assert exponents.sum() < 0
    for n in (1, 5, 20):
        direct = float(np.linalg.slogdet(np.linalg.matrix_power(A, n))[1])
        assert volume_decay_rate(exponents, n) == pytest.approx(direct, rel=1e-9)


def test_t4_volume_preserving_map_has_zero_rate():
    A = np.array([[0.0, -1.0], [1.0, 0.0]])
    exponents = np.log(np.linalg.svd(A, compute_uv=False))
    assert volume_decay_rate(exponents, 100) == pytest.approx(0.0, abs=1e-12)


def test_t4_rejects_negative_step_counts():
    with pytest.raises(ValueError):
        volume_decay_rate(np.array([-1.0]), -1)


# --- Theorem 5: No Local Criterion Detects Pooling -------------------------


def test_t5_determinant_is_positive_everywhere():
    rng = np.random.default_rng(6)
    for _ in range(500):
        x = float(rng.uniform(-4, 4))
        y = float(rng.uniform(-20, 20))
        _, _, det = winding_witness(x, y)
        assert det > 0.0
        assert det == pytest.approx(np.exp(2 * x), rel=1e-9)


def test_t5_the_map_is_not_injective():
    rng = np.random.default_rng(7)
    for _ in range(200):
        x = float(rng.uniform(-3, 3))
        y = float(rng.uniform(-10, 10))
        a, _, _ = winding_witness(x, y)
        b, _, _ = winding_witness(x, y + 2 * np.pi)
        assert np.allclose(a, b, atol=1e-9), "two preimages must share an image"


def test_t5_the_two_preimages_share_every_spectral_invariant():
    """The no-go: no pointwise Jacobian statistic separates them."""
    rng = np.random.default_rng(8)
    for _ in range(200):
        x = float(rng.uniform(-3, 3))
        y = float(rng.uniform(-10, 10))
        _, j1, d1 = winding_witness(x, y)
        _, j2, d2 = winding_witness(x, y + 2 * np.pi)
        assert d1 == pytest.approx(d2, rel=1e-9)
        assert np.allclose(
            np.linalg.svd(j1, compute_uv=False),
            np.linalg.svd(j2, compute_uv=False),
            atol=1e-9,
        )


def test_t5_smallest_singular_value_never_signals_the_collision():
    for x in np.linspace(-2, 2, 40):
        _, j, _ = winding_witness(float(x), 1.0)
        assert float(np.linalg.svd(j, compute_uv=False).min()) > 0.9 * np.exp(x)


def test_t5_explains_the_measured_null_result():
    """Perfectly conditioned Jacobian, and a collision anyway.

    Structural reproduction of the measured case: zero of 768 singular values fell
    below 1e-6 times the largest, yet the partition still collapsed. A null result
    from spectral summaries is the predicted outcome, not a measurement failure.
    """
    _, j, det = winding_witness(0.0, 0.0)
    sv = np.linalg.svd(j, compute_uv=False)
    assert sv.min() / sv.max() == pytest.approx(1.0)
    assert det == pytest.approx(1.0)
    a, _, _ = winding_witness(0.0, 0.0)
    b, _, _ = winding_witness(0.0, 2 * np.pi)
    assert np.allclose(a, b, atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
