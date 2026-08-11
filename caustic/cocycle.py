"""Lyapunov exponents of a Jacobian cocycle, by QR iteration.

This is the differential-geometric linearization composed along a trajectory. A
single Jacobian is the local linear model of one step; the product

    J_n J_{n-1} ... J_1

is the local linear model of n steps, and its singular values grow or decay
exponentially at rates that Oseledets' Multiplicative Ergodic Theorem guarantees
exist. Those rates are what `lyapunov_spectrum` returns.

Multiplying the Jacobians directly does not work. Every column of the running
product collapses onto the leading singular direction within a few dozen steps,
and the norm overflows or underflows shortly after, so all but the top exponent
are lost to floating point. Benettin's algorithm re-orthonormalizes the frame
after each step and accumulates log|R_ii| from the QR factorization, which keeps
all D directions separated for as long as the trajectory runs.

The estimator is exact for the cases the test suite pins: a diagonal cocycle
returns log of the diagonal, an orthogonal cocycle returns zero, the exponents
sum to the mean log|det|, and scaling every Jacobian by c shifts every exponent
by log c.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["lyapunov_spectrum", "finite_time_spectrum"]

_FLOOR = 1e-300
"""Clamp for log|R_ii|. A genuinely singular step would give -inf and poison the
running sum; the clamp turns it into about -690, which reads as "this direction
died" without producing NaN. Folding produces exactly this case, so it is a
supported input rather than an error."""


def _qr_step(J: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One re-orthonormalized step. Returns the new frame and |diag(R)|.

    QR is unique only up to the sign of each column, and LAPACK's choice of sign
    is not stable across inputs. Folding the sign into Q keeps R's diagonal
    positive so that the accumulated logs are comparable step to step.
    """
    Q_next, R = torch.linalg.qr(J @ Q)
    d = torch.diagonal(R)
    s = torch.sign(d)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return Q_next * s, (d * s).clamp_min(_FLOOR)


def lyapunov_spectrum(jacobians, dt: float = 1.0) -> np.ndarray:
    """Lyapunov exponents of the cocycle, descending.

    Args:
        jacobians: sequence of (D, D) tensors, applied in the order given.
        dt: time represented by one step. Exponents are per unit time.

    Returns:
        Array of D exponents, descending.

    Raises:
        ValueError: if the sequence is empty or a Jacobian is not square.
    """
    jacobians = list(jacobians)
    if not jacobians:
        raise ValueError("need at least one Jacobian")
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}")

    D = jacobians[0].shape[0]
    if any(J.shape != (D, D) for J in jacobians):
        raise ValueError("every Jacobian must be square and the same size")

    Q = torch.eye(D, dtype=torch.float64)
    acc = np.zeros(D)
    for J in jacobians:
        Q, d = _qr_step(J.double().cpu(), Q)
        acc += np.log(d.numpy())
    return acc / (len(jacobians) * dt)


def finite_time_spectrum(jacobians, dt: float = 1.0) -> np.ndarray:
    """The running exponents after each step, shape (n, D).

    Lyapunov exponents are an asymptotic statement. Over a 6-block transformer the
    cocycle has 6 steps, which is nowhere near asymptotic, and calling the result
    a Lyapunov exponent would be an overclaim. This returns the whole convergence
    trace so that a report can show whether the value had settled or was still
    moving when the trajectory ran out.
    """
    jacobians = list(jacobians)
    if not jacobians:
        raise ValueError("need at least one Jacobian")
    D = jacobians[0].shape[0]
    Q = torch.eye(D, dtype=torch.float64)
    acc = np.zeros(D)
    out = np.empty((len(jacobians), D))
    for k, J in enumerate(jacobians, start=1):
        Q, d = _qr_step(J.double().cpu(), Q)
        acc += np.log(d.numpy())
        out[k - 1] = acc / (k * dt)
    return out
