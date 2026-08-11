"""Attractor dimension from a Lyapunov spectrum, and what it bounds.

This is where the three linearizations produce a single number. The Jacobian is
the local linearization; composing it along a trajectory gives the Lyapunov
spectrum, which is the chaos-theoretic invariant; and the Kaplan-Yorke formula
turns that spectrum into a *fractal* dimension:

    D_KY = j + (sum_{i<=j} lambda_i) / |lambda_{j+1}|

where j is the largest index whose partial sum is still non-negative. The
interpolation term is what makes it fractional, and therefore a fractal dimension
rather than a count of directions.

The Kaplan-Yorke conjecture states that D_KY equals the information dimension of
the attractor. It is a conjecture, not a theorem, and it is proved only in
restricted settings. Everything here reports D_KY, which is well-defined for any
spectrum; nothing here proves the attractor's information dimension equals it.

**Why this is worth computing on a transformer.** Two classical results attach to
an attractor dimension and both are statements about compression:

1. Whitney's embedding theorem: a d-dimensional manifold embeds in R^{2d+1}.
2. Takens' delay-embedding theorem: an attractor of box dimension d is
   reconstructed from 2d+1 generic delayed scalar observations.

If the hidden-state dynamics of a transformer sit on an attractor of dimension
D_KY much smaller than the model width, then the width is not the information
content of the state, and 2*D_KY+1 is the scale at which lossless reduction stops
being possible. That is a bound with a number in it rather than a hyperparameter,
and it is falsifiable: reduce below it and reconstruction must degrade.

**What this module does not claim.** It does not claim a transformer forward pass
is a measure-preserving dynamical system, which is the hypothesis Oseledets' and
Takens' theorems actually require. A finite non-autonomous trajectory driven by
input tokens is not obviously such a system. The numbers here are computed as if
it were, and that gap is the single largest caveat on any conclusion drawn from
them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kaplan_yorke_dimension", "embedding_bound", "metric_entropy", "spectrum_report"]


def kaplan_yorke_dimension(lam: np.ndarray) -> float:
    """Kaplan-Yorke (Lyapunov) dimension of a spectrum.

    Returns 0.0 when every exponent is negative (the trajectory collapses to a
    fixed point) and len(lam) when every partial sum stays non-negative (the
    attractor fills the space and the formula has nothing to interpolate).
    """
    lam = np.sort(np.asarray(lam, dtype=np.float64).ravel())[::-1]
    if lam.size == 0:
        raise ValueError("spectrum is empty")
    if lam[0] < 0:
        return 0.0

    cumulative = np.cumsum(lam)
    j = int(np.searchsorted(-cumulative, 0.0, side="right"))
    # searchsorted on the negated cumulative sum gives the count of leading
    # entries that are still non-negative, which is exactly j.
    if j >= lam.size:
        return float(lam.size)
    denom = abs(lam[j])
    if denom < 1e-300:
        return float(j)
    return float(j + cumulative[j - 1] / denom) if j > 0 else 0.0


def metric_entropy(lam: np.ndarray) -> float:
    """Sum of the positive exponents: the Pesin bound on Kolmogorov-Sinai entropy.

    Pesin's identity states that for a smooth system with an SRB measure the
    KS entropy equals the sum of positive Lyapunov exponents. This returns that
    sum. It is an information production rate, in nats per step: the amount of
    information about the initial condition destroyed by one step of the
    dynamics.
    """
    lam = np.asarray(lam, dtype=np.float64)
    return float(lam[lam > 0].sum())


def embedding_bound(d_ky: float) -> int:
    """Takens/Whitney sufficient embedding dimension, ceil(2 * d) + 1.

    The number of generic observations that suffice to reconstruct a state on an
    attractor of dimension `d_ky`. Sufficient, not necessary: a specific system
    may be reconstructible from fewer. Reducing below it removes the guarantee,
    which is the falsifiable half.
    """
    if d_ky < 0:
        raise ValueError(f"dimension must be non-negative; got {d_ky}")
    return int(np.ceil(2.0 * d_ky)) + 1


def spectrum_report(lam: np.ndarray) -> dict[str, float]:
    """Every scalar this module derives from one spectrum."""
    lam = np.sort(np.asarray(lam, dtype=np.float64).ravel())[::-1]
    d_ky = kaplan_yorke_dimension(lam)
    return {
        "D": float(lam.size),
        "lambda_max": float(lam[0]),
        "sum": float(lam.sum()),
        "n_positive": float((lam > 0).sum()),
        "kaplan_yorke": d_ky,
        "compression_ratio": float(lam.size) / max(d_ky, 1e-12),
        "embedding_bound": float(embedding_bound(d_ky)),
        "metric_entropy": metric_entropy(lam),
    }
