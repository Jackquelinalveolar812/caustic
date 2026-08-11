"""The Oseledets growth-rate filtration, encoded as intervals.

Oseledets' Multiplicative Ergodic Theorem states that for a cocycle of Jacobians
along a trajectory there is a filtration of the state space

    V_1 subset V_2 subset ... subset V_k = R^D

in which every vector of ``V_i \\ V_{i-1}`` grows at exponential rate ``lambda_i``.
Encoding that as ``(birth, death, multiplicity)`` triples puts it in the same
shape as a persistence barcode, which is the bridge the programme is built to
test: barcode tooling then applies to a chaos-theoretic object without
modification.

**The honest caveat, stated before any result.** When all D exponents are
distinct the filtration has D steps of multiplicity one, and the encoding carries
exactly the information the sorted spectrum already carried. The bridge is
informative only when exponents cluster, so `filtration_entropy` is provided to
report how far from that degenerate case a measured spectrum actually is. A
measured spectrum with entropy at log(D) means the encoding bought nothing, and
the programme reports that rather than presenting the triples as a result.

Grouping requires a tolerance, and the choice of tolerance is a real modelling
decision rather than a detail: `tol` too small returns D singletons for any
floating-point spectrum, `tol` too large collapses everything into one bar. The
grouping is monotone in `tol`, which `tests/test_oseledets.py` pins, so a
tolerance sweep is the honest way to report the structure.
"""

from __future__ import annotations

import numpy as np

__all__ = ["growth_filtration", "filtration_entropy", "tolerance_sweep"]

Bar = tuple[float, float, int]


def growth_filtration(lam: np.ndarray, tol: float = 1e-3) -> list[Bar]:
    """Group a Lyapunov spectrum into Oseledets subspaces.

    Args:
        lam: exponents, in any order. A spectrum is a multiset, so the result
            does not depend on the order given.
        tol: absolute gap at or below which two adjacent exponents are treated
            as one subspace.

    Returns:
        ``(birth, death, multiplicity)`` per distinct exponent, descending by
        birth. ``death`` is the next distinct exponent below, or ``-inf`` for the
        last bar, so the bars tile the line and the filtration is nested.

    Raises:
        ValueError: if the spectrum is empty or `tol` is negative.
    """
    lam = np.asarray(lam, dtype=np.float64).ravel()
    if lam.size == 0:
        raise ValueError("spectrum is empty")
    if tol < 0:
        raise ValueError(f"tol must be non-negative; got {tol}")

    lam = np.sort(lam)[::-1]
    groups: list[list[float]] = [[float(lam[0])]]
    for x in lam[1:]:
        # Compare against the last member rather than the group mean: chaining on
        # the mean would let a long slow drift merge into one bar even where no
        # two adjacent exponents are within tol of each other.
        if abs(groups[-1][-1] - float(x)) <= tol:
            groups[-1].append(float(x))
        else:
            groups.append([float(x)])

    bars: list[Bar] = []
    for i, g in enumerate(groups):
        birth = float(np.mean(g))
        death = float(np.mean(groups[i + 1])) if i + 1 < len(groups) else -np.inf
        bars.append((birth, death, len(g)))
    return bars


def filtration_entropy(bars: list[Bar]) -> float:
    """Shannon entropy of the multiplicity distribution, in nats.

    The diagnostic that decides whether the encoding is worth anything.

    - ``0.0`` means one subspace carries every direction.
    - ``log(D)`` means D singletons, so the filtration is the sorted spectrum
      with extra steps and the barcode encoding is free and worthless.

    Anything strictly between is genuine subspace structure, and its distance
    below ``log(D)`` is how much the encoding actually bought.
    """
    mult = np.array([b[2] for b in bars], dtype=np.float64)
    total = mult.sum()
    if total <= 0:
        raise ValueError("filtration has no directions")
    p = mult / total
    return float(-(p * np.log(np.maximum(p, 1e-300))).sum())


def tolerance_sweep(lam: np.ndarray, tols: np.ndarray | None = None) -> list[dict]:
    """Bar count and entropy across a range of tolerances.

    Reporting one tolerance invites the reader to assume it was chosen after
    seeing the answer. The sweep is monotone in `tol` by construction, so the
    curve is the honest object: a spectrum with real subspace structure shows a
    plateau, and one without shows a smooth slide from D bars to one.
    """
    lam = np.asarray(lam, dtype=np.float64).ravel()
    if tols is None:
        spread = float(np.max(lam) - np.min(lam)) if lam.size > 1 else 1.0
        tols = np.geomspace(max(spread, 1e-12) * 1e-6, max(spread, 1e-12), 13)
    out = []
    for t in tols:
        bars = growth_filtration(lam, tol=float(t))
        out.append(
            {
                "tol": float(t),
                "n_bars": len(bars),
                "max_multiplicity": max(b[2] for b in bars),
                "entropy": filtration_entropy(bars),
            }
        )
    return out
