"""Scalar summaries of a Jacobian singular spectrum.

Each summary is one of the three linearizations the programme composes:

    sigma_max, log_volume   differential-geometric  -- local expansion and volume change
    stable_rank             algebraic               -- effective number of active directions
    tail_alpha              fractal                 -- log-log slope of the spectrum

`tail_alpha` is the candidate for the width-invariant constant. It is fitted over
the bulk of the spectrum only: the leading singular values are the spike and the
trailing ones sit on the numerical floor, and neither belongs to a power-law tail.

Measured on distilgpt2, 10 token positions, one 61-token passage, both a grounded
passage and its shuffled-token control:

    tail_alpha, blocks 1-5   grounded 0.2374 +/- 0.0293  (cv 0.124)
                             shuffled 0.2342 +/- 0.0481  (cv 0.205)
                             difference 1.4% of grounded
    block 0                  0.6168 -- an outlier, excluded post hoc

The two conditions agree to 1.4%, so tail_alpha is a property of the architecture
and not of the content. That is a negative result for using it to detect
ungrounded generation, and the reason it is instead a candidate for pruning
budgets. Block 0 was excluded after inspecting the data; that decision has not yet
been re-tested on held-out text and must not be reported as confirmed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BULK", "sigma_max", "stable_rank", "tail_alpha", "log_volume", "summarize"]

BULK = (10, 400)
"""Index range for the power-law fit. Excludes the leading spike and the floor."""


def sigma_max(sv: np.ndarray) -> float:
    """Largest singular value. Products across blocks bound perturbation growth."""
    return float(np.max(sv))


def stable_rank(sv: np.ndarray) -> float:
    """(sum sigma^2) / sigma_max^2. Effective number of active directions.

    Bounded above by len(sv) and below by 1. Scale-invariant by construction.
    """
    sv = np.asarray(sv, dtype=np.float64)
    return float((sv**2).sum() / max(sv.max() ** 2, 1e-300))


def tail_alpha(sv: np.ndarray, bulk: tuple[int, int] = BULK) -> float:
    """Negated slope of log sigma_i against log i over `bulk`.

    Returns the exponent a in sigma_i ~ i^-a. Larger means faster decay, so fewer
    directions carry the map. Scale-invariant: multiplying every sigma by c shifts
    the intercept and leaves the slope unchanged.
    """
    sv = np.sort(np.asarray(sv, dtype=np.float64))[::-1]
    lo, hi = bulk
    hi = min(hi, len(sv))
    if hi - lo < 2:
        raise ValueError(f"bulk {bulk} leaves fewer than 2 points for {len(sv)} values")
    i = np.arange(lo, hi)
    slope = np.polyfit(np.log(i + 1.0), np.log(np.maximum(sv[i], 1e-300)), 1)[0]
    return float(-slope)


def log_volume(sv: np.ndarray) -> float:
    """sum log sigma_i, the local log volume change.

    Large negative means the block contracts volume at this point, which is the
    coarse signature of folding. It is necessary but not sufficient: a map can
    contract volume and remain injective.
    """
    sv = np.asarray(sv, dtype=np.float64)
    return float(np.log(np.maximum(sv, 1e-300)).sum())


def summarize(sv: np.ndarray, bulk: tuple[int, int] = BULK) -> dict[str, float]:
    """All four summaries of one spectrum."""
    return {
        "sigma_max": sigma_max(sv),
        "stable_rank": stable_rank(sv),
        "tail_alpha": tail_alpha(sv, bulk),
        "log_volume": log_volume(sv),
    }
