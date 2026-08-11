"""Ground truth for the Oseledets growth-rate filtration.

The spine claim of the programme is that the filtration Oseledets' theorem
produces is the same shape of object a persistence module decomposes into, so
barcode tooling applies to it unchanged. That claim has a failure mode which
these tests are written to expose rather than hide: when every exponent is
distinct, the filtration has D steps of multiplicity one and carries exactly the
information the sorted spectrum already carried. The encoding is then free and
worthless.

`test_distinct_spectrum_gives_unit_multiplicities` is the test that pins the
failure mode. If real spectra look like that, the bridge buys nothing and the
programme says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.oseledets import growth_filtration, filtration_entropy


def test_degenerate_spectrum_gives_one_interval_of_full_multiplicity():
    """All exponents equal: one Oseledets subspace, the whole space."""
    bars = growth_filtration(np.full(8, 0.5))
    assert len(bars) == 1
    assert bars[0][2] == 8
    assert bars[0][1] == -np.inf


def test_distinct_spectrum_gives_unit_multiplicities():
    """The trivial case the bridge must be honest about.

    D distinct exponents give D bars of multiplicity 1, which is the sorted
    spectrum with extra steps. If measured spectra look like this, the encoding
    carries no information and the programme reports that.
    """
    bars = growth_filtration(np.array([3.0, 2.0, 1.0]))
    assert [b[2] for b in bars] == [1, 1, 1]
    assert filtration_entropy(bars) == pytest.approx(np.log(3), abs=1e-12)


def test_clustered_spectrum_groups_within_tolerance():
    lam = np.array([2.0, 2.0 + 1e-6, 2.0 - 1e-6, -1.0, -1.0 + 1e-7])
    bars = growth_filtration(lam, tol=1e-3)
    assert sorted(b[2] for b in bars) == [2, 3]


def test_multiplicities_sum_to_dimension():
    rng = np.random.default_rng(0)
    lam = rng.normal(size=40)
    assert sum(b[2] for b in growth_filtration(lam)) == 40


def test_births_descend_and_death_is_the_next_birth():
    """The filtration is nested: each bar dies where the next one is born."""
    lam = np.array([5.0, 5.0, 2.0, -1.0, -1.0, -1.0])
    bars = growth_filtration(lam, tol=1e-9)
    births = [b[0] for b in bars]
    assert births == sorted(births, reverse=True)
    for i in range(len(bars) - 1):
        assert bars[i][1] == pytest.approx(bars[i + 1][0])
    assert bars[-1][1] == -np.inf


def test_input_order_does_not_matter():
    """A spectrum is a multiset. Shuffling it must not change the filtration."""
    rng = np.random.default_rng(1)
    lam = np.concatenate([np.full(3, 1.0), np.full(2, -2.0), [0.5]])
    a = growth_filtration(lam, tol=1e-9)
    b = growth_filtration(rng.permutation(lam), tol=1e-9)
    assert a == b


def test_wider_tolerance_never_increases_the_bar_count():
    """Monotone in tol. A grouping rule that is not monotone is not a filtration."""
    rng = np.random.default_rng(2)
    lam = rng.normal(size=60)
    counts = [len(growth_filtration(lam, tol=t)) for t in (1e-9, 1e-2, 1e-1, 1.0)]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts


def test_entropy_is_zero_for_a_single_subspace_and_maximal_when_all_distinct():
    """The negative control at both ends."""
    assert filtration_entropy(growth_filtration(np.full(16, 2.0))) == pytest.approx(0.0)
    distinct = growth_filtration(np.arange(16.0))
    assert filtration_entropy(distinct) == pytest.approx(np.log(16), abs=1e-12)


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError):
        growth_filtration(np.arange(4.0), tol=-1.0)


def test_empty_spectrum_is_rejected():
    with pytest.raises(ValueError):
        growth_filtration(np.array([]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
