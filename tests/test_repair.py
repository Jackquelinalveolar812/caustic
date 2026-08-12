"""Contracts for the repair measurement, against hand-computable partitions.

The model is a dictionary, so every expected value is exact. The load-bearing
cases are the two asymmetric verdicts: a prefix that separates a collapsed
partition is a repair, and a prefix that merges a separated one is a regression.
Both occur in the measurements this module is built on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec
from caustic.repair import NEUTRAL_PREFIX, repair_by_context, sweep_prefixes

TPLS = ("The capital of {e} is", "{e}'s capital is")
ENTS = ("France", "Italy", "Japan", "Spain")
GOLD = {"France": 1, "Italy": 2, "Japan": 3, "Spain": 4}
SPEC = RelationSpec(TPLS, ENTS)


def entity_of(prompt: str) -> str:
    for e in ENTS:
        if e in prompt:
            return e
    raise AssertionError(prompt)


def collapsed_until_prefixed(prefix: str):
    """Collapsed with no prefix, fully separated once the prefix is present."""

    def fn(prompt: str) -> int:
        return GOLD[entity_of(prompt)] if prompt.startswith(prefix) else 0

    return fn


def test_repair_is_detected_and_accuracy_reported():
    r = repair_by_context(SPEC, collapsed_until_prefixed(NEUTRAL_PREFIX), gold=GOLD)
    assert r.before_largest_orbit == 4 and r.after_largest_orbit == 1
    assert r.repaired and not r.worsened
    assert r.before_accuracy == pytest.approx(0.0)
    assert r.after_accuracy == pytest.approx(1.0)
    assert r.accuracy_delta == pytest.approx(1.0)
    assert "REPAIRED" in str(r)


def test_a_prefix_that_merges_entities_is_reported_as_worsened():
    """The " the" x 128 case: largest orbit went 4 -> 20 in the real measurement."""

    def fn(prompt: str) -> int:
        return 0 if prompt.startswith(NEUTRAL_PREFIX) else GOLD[entity_of(prompt)]

    r = repair_by_context(SPEC, fn, gold=GOLD)
    assert r.before_largest_orbit == 1 and r.after_largest_orbit == 4
    assert r.worsened and not r.repaired
    assert "WORSENED" in str(r)


def test_no_change_is_neither_repaired_nor_worsened():
    r = repair_by_context(SPEC, lambda p: GOLD[entity_of(p)], gold=GOLD)
    assert not r.repaired and not r.worsened
    assert "no repair" in str(r)


def test_partial_separation_is_not_called_a_repair():
    """Going from 4 merged to 2 merged is progress, not a repair. Only 1 counts."""

    def fn(prompt: str) -> int:
        if not prompt.startswith(NEUTRAL_PREFIX):
            return 0
        return {"France": 1, "Italy": 1, "Japan": 3, "Spain": 4}[entity_of(prompt)]

    r = repair_by_context(SPEC, fn)
    assert r.before_largest_orbit == 4 and r.after_largest_orbit == 2
    assert not r.repaired and not r.worsened


def test_verdict_never_consults_gold():
    """Repair is a statement about the partition, decidable with no ground truth."""
    fn = collapsed_until_prefixed(NEUTRAL_PREFIX)
    assert repair_by_context(SPEC, fn).repaired
    assert repair_by_context(SPEC, fn, gold=GOLD).repaired


def test_missing_gold_entity_is_rejected():
    with pytest.raises(ValueError):
        repair_by_context(SPEC, lambda p: 0, gold={"France": 1})


def test_sweep_returns_every_prefix_not_only_the_best():
    fn = collapsed_until_prefixed("GOOD ")
    out = sweep_prefixes(SPEC, fn, {"good": "GOOD ", "bad": "BAD "}, gold=GOLD)
    assert set(out) == {"good", "bad"}
    assert out["good"].repaired and not out["bad"].repaired
