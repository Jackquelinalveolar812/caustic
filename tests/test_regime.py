"""Contracts for the regime detector, against hand-computable answers.

The detector is driven by an injected `answer_fn`, so every case here is exact:
the model is replaced by a dictionary and the expected partition is written out by
hand. No tolerance, no sampling, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, orbit_partition, symmetry_scores

TPLS = ("The capital of {e} is", "{e}'s capital is", "The capital city of {e} is")
ENTS = ("France", "Italy", "Japan", "Spain")


def make_fn(mapping: dict[str, int], default: int = 0):
    """answer_fn keyed on whichever entity name appears in the prompt.

    Keys are tried longest-first. Naive insertion-order substring matching makes
    "C1" match the prompt for "C15", which silently mislabels five entities and
    was caught only because the resulting partition disagreed with a hand count.
    Same failure class as a tie-break bug in a kernel parity test: the fixture is
    wrong, the code is fine, and nothing raises.
    """

    def fn(prompt: str) -> int:
        for k in sorted(mapping, key=len, reverse=True):
            if k in prompt:
                return mapping[k]
        return default

    return fn


# --- partition -------------------------------------------------------------


def test_all_distinct_gives_singleton_orbits():
    spec = RelationSpec(TPLS, ENTS)
    r = orbit_partition(spec, make_fn({"France": 1, "Italy": 2, "Japan": 3, "Spain": 4}))
    assert r.n_distinct == 4
    assert r.largest_orbit == 1
    assert not r.collapsed
    assert r.collapse_ratio == pytest.approx(0.25)


def test_total_collapse_is_detected():
    """The " the" x 128 case: every entity onto one answer."""
    spec = RelationSpec(TPLS, ENTS)
    r = orbit_partition(spec, make_fn({}, default=7))
    assert r.n_distinct == 1
    assert r.largest_orbit == 4
    assert r.collapsed
    assert r.collapse_ratio == pytest.approx(1.0)


def test_partial_collapse_names_the_merged_entities():
    spec = RelationSpec(TPLS, ENTS)
    r = orbit_partition(spec, make_fn({"France": 1, "Italy": 1, "Japan": 3, "Spain": 4}))
    assert r.largest_orbit == 2
    assert r.collapsed
    merged = [v for v in r.orbits.values() if len(v) > 1]
    assert merged == [["France", "Italy"]]
    assert "France, Italy" in str(r)


def test_non_injective_relation_never_reports_collapse():
    """Japan and India both mapping to Asia is correct, not a defect.

    Measured consequence of ignoring this: 0.995 AUROC on an injective relation
    against 0.273 on a many-to-one one, where the signal inverts.
    """
    spec = RelationSpec(TPLS, ENTS, injective=False)
    r = orbit_partition(spec, make_fn({}, default=7))
    assert r.largest_orbit == 4
    assert not r.collapsed
    assert "not injective" in str(r)


def test_only_the_first_template_drives_the_partition():
    """The partition is defined on one template; paraphrases are for invariance."""
    calls = []

    def fn(prompt: str) -> int:
        calls.append(prompt)
        return 1

    orbit_partition(RelationSpec(TPLS, ENTS), fn)
    assert len(calls) == len(ENTS)
    assert all(c.startswith("The capital of") for c in calls)


# --- symmetry scores -------------------------------------------------------


def test_perfect_model_scores_one_and_zero():
    """Consistent across paraphrases, distinct across entities."""
    spec = RelationSpec(TPLS, ENTS)
    s = symmetry_scores(spec, make_fn({"France": 1, "Italy": 2, "Japan": 3, "Spain": 4}))
    for e in ENTS:
        assert s[e]["invariance"] == pytest.approx(1.0)
        assert s[e]["collision"] == pytest.approx(0.0)
        assert s[e]["score"] == pytest.approx(1.0)


def test_total_collapse_scores_one_and_one():
    """Invariant where it should be equivariant: the prior-driven failure."""
    spec = RelationSpec(TPLS, ENTS)
    s = symmetry_scores(spec, make_fn({}, default=7))
    for e in ENTS:
        assert s[e]["invariance"] == pytest.approx(1.0)
        assert s[e]["collision"] == pytest.approx(1.0)
        assert s[e]["score"] == pytest.approx(0.0)


def test_collision_counts_only_other_entities():
    """France and Italy share; Japan and Spain do not. 4 entities, so 3 others."""
    spec = RelationSpec(TPLS, ENTS)
    s = symmetry_scores(spec, make_fn({"France": 1, "Italy": 1, "Japan": 3, "Spain": 4}))
    assert s["France"]["collision"] == pytest.approx(1 / 3)
    assert s["Japan"]["collision"] == pytest.approx(0.0)


def test_invariance_falls_when_paraphrases_disagree():
    """One template out of three disagreeing gives 1 agreeing pair of 3."""

    def fn(prompt: str) -> int:
        return 9 if prompt.startswith("The capital city") else 1

    s = symmetry_scores(RelationSpec(TPLS, ENTS), fn)
    assert s["France"]["invariance"] == pytest.approx(1 / 3)


# --- validation ------------------------------------------------------------


def test_template_without_placeholder_is_rejected():
    with pytest.raises(ValueError):
        RelationSpec(("The capital of France is",), ENTS)


def test_single_entity_is_rejected():
    with pytest.raises(ValueError):
        RelationSpec(TPLS, ("France",))


def test_invariance_requires_two_templates():
    spec = RelationSpec(("The capital of {e} is",), ENTS)
    with pytest.raises(ValueError):
        symmetry_scores(spec, make_fn({}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --- the orbit error bound -------------------------------------------------


def test_bound_is_zero_when_every_orbit_is_a_singleton():
    spec = RelationSpec(TPLS, ENTS)
    r = orbit_partition(spec, make_fn({"France": 1, "Italy": 2, "Japan": 3, "Spain": 4}))
    assert r.certified_errors == 0
    assert r.certified_error_rate == pytest.approx(0.0)


def test_total_collapse_certifies_n_minus_one_errors():
    """Four entities, one answer: at most one can be right, so three are wrong."""
    spec = RelationSpec(TPLS, ENTS)
    r = orbit_partition(spec, make_fn({}, default=7))
    assert r.certified_errors == 3
    assert r.certified_error_rate == pytest.approx(0.75)


def test_bound_matches_the_measured_capital_case():
    """n = 20, m = 15 with no prefix: at least 5 answers are provably wrong."""
    ents = tuple(f"C{i}" for i in range(20))
    mapping = {e: i for i, e in enumerate(ents)}
    for e in ("C15", "C16", "C17", "C18", "C19"):
        mapping[e] = 0  # merge five entities onto C0's answer
    r = orbit_partition(RelationSpec(TPLS, ents), make_fn(mapping))
    assert r.n_distinct == 15
    assert r.certified_errors == 5
    assert r.certified_error_rate == pytest.approx(0.25)


def test_bound_is_never_negative_and_never_exceeds_n_minus_one():
    import random

    rng = random.Random(0)
    ents = tuple(f"E{i}" for i in range(12))
    for _ in range(200):
        mapping = {e: rng.randrange(1, 6) for e in ents}
        r = orbit_partition(RelationSpec(TPLS, ents), make_fn(mapping))
        assert 0 <= r.certified_errors <= len(ents) - 1


def test_bound_is_tight_when_each_orbit_holds_one_correct_answer():
    """Construct the equality case: 2 orbits of 2, exactly 2 errors, bound = 2."""
    ents = ("Zeta", "Yota", "Xano", "Wuvi")
    gold = {"Zeta": 1, "Yota": 2, "Xano": 3, "Wuvi": 4}
    answers = {"Zeta": 1, "Yota": 1, "Xano": 3, "Wuvi": 3}  # Zeta, Xano right
    r = orbit_partition(RelationSpec(TPLS, ents), make_fn(answers))
    actual = sum(answers[e] != gold[e] for e in ents)
    assert r.certified_errors == 2
    assert actual == 2, "the bound is attained, not merely respected"


def test_bound_never_exceeds_the_true_error_count():
    """The theorem, checked against ground truth on random instances."""
    import random

    rng = random.Random(1)
    ents = tuple(f"E{i}" for i in range(10))
    gold = {e: i for i, e in enumerate(ents)}
    for _ in range(500):
        answers = {e: rng.randrange(0, 7) for e in ents}
        r = orbit_partition(RelationSpec(TPLS, ents), make_fn(answers))
        actual = sum(answers[e] != gold[e] for e in ents)
        assert r.certified_errors <= actual, f"bound {r.certified_errors} > actual {actual}"


def test_non_injective_relation_certifies_nothing():
    spec = RelationSpec(TPLS, ENTS, injective=False)
    r = orbit_partition(spec, make_fn({}, default=7))
    assert r.certified_errors == 0
