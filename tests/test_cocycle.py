"""Ground truth for the Lyapunov spectrum of a Jacobian cocycle.

Every case here has a closed-form answer. A cocycle implementation that merely
"looks plausible" fails these: the diagonal case pins the values exactly, the
orthogonal case is the negative control that a naive implementation gets wrong by
accumulating norm drift, and the volume identity ties the spectrum to the
determinant it must agree with.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.cocycle import lyapunov_spectrum


def test_diagonal_cocycle_recovers_log_of_diagonal():
    """For J_k = diag(a), the n-step product is diag(a^n), so lambda_i = log a_i."""
    a = np.array([4.0, 2.0, 0.5, 0.1])
    Js = [torch.diag(torch.tensor(a, dtype=torch.float64)) for _ in range(200)]
    got = lyapunov_spectrum(Js)
    assert np.allclose(got, np.log(a), atol=1e-8), f"got {got}, want {np.log(a)}"


def test_orthogonal_cocycle_has_zero_exponents():
    """Rotations preserve length, so every exponent is exactly 0.

    The negative control. An implementation that multiplies Jacobians directly and
    normalizes only at the end drifts off zero here.
    """
    g = torch.Generator().manual_seed(0)
    Js = [torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))[0] for _ in range(300)]
    assert np.allclose(lyapunov_spectrum(Js), 0.0, atol=1e-8)


def test_exponents_are_descending():
    g = torch.Generator().manual_seed(1)
    Js = [torch.randn(5, 5, generator=g, dtype=torch.float64) for _ in range(200)]
    assert np.all(np.diff(lyapunov_spectrum(Js)) <= 1e-12)


def test_sum_of_exponents_equals_mean_log_abs_det():
    """sum_i lambda_i = mean_k log|det J_k|. The volume identity.

    This is the tie between the spectrum and the geometry: the total exponent is
    the average log volume change per step, whatever the individual directions do.
    """
    g = torch.Generator().manual_seed(2)
    Js = [torch.randn(4, 4, generator=g, dtype=torch.float64) for _ in range(400)]
    expected = float(np.mean([torch.linalg.slogdet(J)[1].item() for J in Js]))
    assert lyapunov_spectrum(Js).sum() == pytest.approx(expected, abs=1e-8)


def test_scaling_every_jacobian_shifts_every_exponent():
    """J -> cJ sends lambda_i -> lambda_i + log c, for every i."""
    g = torch.Generator().manual_seed(3)
    Js = [torch.randn(4, 4, generator=g, dtype=torch.float64) for _ in range(200)]
    c = 3.0
    base = lyapunov_spectrum(Js)
    scaled = lyapunov_spectrum([c * J for J in Js])
    assert np.allclose(scaled - base, np.log(c), atol=1e-9)


def test_dt_divides_the_exponents():
    a = np.array([np.e**2, np.e**-1])
    Js = [torch.diag(torch.tensor(a, dtype=torch.float64)) for _ in range(50)]
    assert np.allclose(lyapunov_spectrum(Js, dt=2.0), lyapunov_spectrum(Js) / 2.0, atol=1e-12)


def test_empty_cocycle_raises():
    with pytest.raises(ValueError):
        lyapunov_spectrum([])


def test_singular_jacobian_does_not_produce_nan():
    """A rank-deficient step sends one exponent to a large negative number, not NaN.

    Folding produces exactly this, so the estimator must survive it rather than
    poisoning the whole spectrum.
    """
    J = torch.eye(4, dtype=torch.float64)
    J[3, 3] = 0.0
    lam = lyapunov_spectrum([J] * 50)
    assert np.all(np.isfinite(lam[:3]))
    assert not np.isnan(lam).any()
    assert lam[-1] < -100.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
