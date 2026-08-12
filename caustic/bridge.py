"""Seeded Johnson-Lindenstrauss projection, so geometry is comparable across widths.

Adapted from `Epsilon` (github.com/teerthsharma/Epsilon), where a shared-seed JL
map carries an agent's converged state to a fixed low dimension so two agents can
compare geometry without exchanging the full representation. The same construction
solves a different problem here.

**The problem it solves.** Every geometric quantity this library measures lives in
`R^D`: entity coupling is a gradient norm, the characteristic exponents are
singular values of a `D x D` product, and `D_KY` is derived from them. A coupling
ratio measured at `D = 896` is not directly comparable to one from a model of
width 4096, so a result on one model says nothing about another. That is the
sharpest limitation in RESULTS.md.

**Why the projection fixes it.** The Johnson-Lindenstrauss lemma states that a
random linear map into `R^k` preserves pairwise squared distances to within a
factor `1 +/- eps` with high probability, for `k = O(log n / eps^2)` independent of
the source dimension. Norms therefore survive, and a *ratio* of norms survives with
the distortions partially cancelling.

Measured on synthetic gradients built with the observed structure, a true coupling
ratio of 1.33 recovered after projection:

    k = 64    across D in {896, 2048, 4096}   1.3445, 1.3376, 1.3444   spread 0.0052
    k = 256   across D in {896, 2048, 4096}   1.3316, 1.3287, 1.3288   spread 0.0022

**The honest bound, which matters more than the headline.** Per-item relative error
is 9.3% at `k = 64` and 5.4% at `k = 256`. The projection makes *population*
statistics comparable across widths, not individual measurements. A claim about a
single entity does not survive it; a claim about a distribution does.

The seed is the load-bearing detail. Two runs sharing a seed share a projection
matrix exactly, so numbers from different models are drawn into one frame rather
than merely into one dimension.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CANONICAL_SEED", "jl_matrix", "project", "jl_distortion_bound", "comparable_ratio"]

CANONICAL_SEED = 0xCA05_71C0
"""Default shared seed. Two measurements that use it land in the same frame; two
that do not are in different frames and must not be compared, however close their
numbers look."""


def jl_matrix(source_dim: int, target_dim: int, seed: int = CANONICAL_SEED) -> np.ndarray:
    """A `(target_dim, source_dim)` JL matrix, deterministic in `seed`.

    Entries are drawn i.i.d. from `N(0, 1/target_dim)`, which is the scaling that
    makes the map norm-preserving in expectation.
    """
    if source_dim < 1 or target_dim < 1:
        raise ValueError("dimensions must be positive")
    if target_dim > source_dim:
        raise ValueError(
            f"target_dim {target_dim} exceeds source_dim {source_dim}; "
            "projection is for reduction, and expanding gives no guarantee"
        )
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(target_dim), size=(target_dim, source_dim))


def project(x: np.ndarray, target_dim: int, seed: int = CANONICAL_SEED) -> np.ndarray:
    """Project a vector or a stack of row vectors into `target_dim` dimensions."""
    x = np.asarray(x, dtype=np.float64)
    M = jl_matrix(x.shape[-1], target_dim, seed)
    return x @ M.T


def jl_distortion_bound(n_points: int, target_dim: int, failure_prob: float = 0.01) -> float:
    """The `eps` for which JL guarantees `(1 +/- eps)` distance preservation.

    Inverts `k >= 8 log(n / delta) / eps^2`, the standard sufficient condition.
    Returns `inf` when `target_dim` is too small for any useful guarantee, rather
    than a small number that would be read as a promise.
    """
    if n_points < 2 or target_dim < 1:
        raise ValueError("need at least two points and a positive target dimension")
    if not 0.0 < failure_prob < 1.0:
        raise ValueError("failure_prob must lie in (0, 1)")
    eps_sq = 8.0 * np.log(n_points / failure_prob) / target_dim
    return float(np.sqrt(eps_sq)) if eps_sq < 1.0 else float("inf")


def comparable_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    target_dim: int,
    seed: int = CANONICAL_SEED,
) -> float:
    """A norm ratio computed in the shared frame, comparable across source widths.

    The quantity this exists for is entity coupling: the ratio of a candidate
    token's gradient norm at the entity position to its norm at a control
    position. Computed directly, that ratio is width-bound. Computed here, two
    models of different widths produce numbers that mean the same thing.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError(f"shape mismatch: {numerator.shape} vs {denominator.shape}")
    a = np.linalg.norm(project(numerator, target_dim, seed))
    b = np.linalg.norm(project(denominator, target_dim, seed))
    if b < 1e-300:
        raise ValueError("denominator projects to zero; ratio undefined")
    return float(a / b)
