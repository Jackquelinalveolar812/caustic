"""Contracts for prefix competition and system-prompt cost.

The model is a dictionary keyed on the prefix, so every verdict is exact. The two
load-bearing behaviours are that the governor declines when no candidate helps,
and that a prompt which improves accuracy while restructuring the partition is
still reported as an intervention rather than as neutral.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.governor import prompt_cost, select_prefix
from caustic.regime import RelationSpec

TPLS = ("Q {e} ?",)
ENTS = ("Alpha", "Bravo", "Charlie", "Delta")
SPEC = RelationSpec(TPLS, ENTS)
IDS = {e: i + 1 for i, e in enumerate(ENTS)}


def entity_of(prompt: str) -> str:
    for e in ENTS:
        if e in prompt:
            return e
    raise AssertionError(prompt)


def model(behaviour: dict[str, str]):
    """behaviour maps a prefix to one of 'separate', 'collapse', 'half'."""

    def fn(prompt: str) -> int:
        mode = "collapse"
        for pre, m in behaviour.items():
            if pre and prompt.startswith(pre):
                mode = m
                break
        else:
            mode = behaviour.get("", "collapse")
        e = entity_of(prompt)
        if mode == "separate":
            return IDS[e]
        if mode == "half":
            return IDS[e] if e in ("Alpha", "Bravo") else 99
        return 0

    return fn


# --- competition -----------------------------------------------------------


def test_governor_picks_the_candidate_that_certifies_fewest_errors():
    fn = model({"": "collapse", "GOOD ": "separate", "MEH ": "half"})
    v = select_prefix(SPEC, fn, {"good": "GOOD ", "meh": "MEH "})
    assert v.winner_name == "good"
    assert v.floor == pytest.approx(0.0)
    assert v.baseline_floor == pytest.approx(0.75)
    assert v.improvement == pytest.approx(0.75)
    assert v.intervened


def test_governor_declines_when_nothing_beats_doing_nothing():
    """A prefix that makes things worse must never be selected."""
    fn = model({"": "separate", "BAD ": "collapse"})
    v = select_prefix(SPEC, fn, {"bad": "BAD "})
    assert v.winner_name == "none"
    assert not v.intervened
    assert v.improvement == pytest.approx(0.0)
    assert "declined" in str(v)


def test_the_empty_prefix_is_always_a_candidate_even_if_not_supplied():
    fn = model({"": "separate", "X ": "collapse"})
    v = select_prefix(SPEC, fn, {"x": "X "})
    assert "none" in v.scores


def test_a_tie_resolves_toward_declining():
    """Equal floors mean the intervention bought nothing, so do not intervene."""
    fn = model({"": "separate", "SAME ": "separate"})
    v = select_prefix(SPEC, fn, {"same": "SAME "})
    assert v.winner_name == "none"


def test_every_candidate_is_scored_and_reported():
    fn = model({"": "collapse", "A ": "separate", "B ": "half"})
    v = select_prefix(SPEC, fn, {"a": "A ", "b": "B "})
    assert set(v.scores) == {"none", "a", "b"}
    assert v.scores["a"] < v.scores["b"] < v.scores["none"]


def test_improvement_is_never_negative():
    fn = model({"": "separate", "W1 ": "collapse", "W2 ": "half"})
    v = select_prefix(SPEC, fn, {"w1": "W1 ", "w2": "W2 "})
    assert v.improvement >= 0.0


def test_non_injective_relation_is_rejected():
    """The scorer is Theorem 1; on a many-to-one relation it inverts."""
    spec = RelationSpec(TPLS, ENTS, injective=False)
    with pytest.raises(ValueError):
        select_prefix(spec, model({"": "separate"}), {"a": "A "})


# --- system-prompt cost ----------------------------------------------------


def test_a_harmful_prompt_reports_a_positive_cost():
    fn = model({"": "separate", "SYS ": "collapse"})
    c = prompt_cost(SPEC, fn, "SYS ", name="sys")
    assert c.cost == pytest.approx(0.75)
    assert c.largest_without == 1 and c.largest_with == 4
    assert not c.neutral
    assert "costs" in str(c)


def test_a_helpful_prompt_reports_a_negative_cost():
    fn = model({"": "collapse", "SYS ": "separate"})
    c = prompt_cost(SPEC, fn, "SYS ")
    assert c.cost == pytest.approx(-0.75)
    assert "helps" in str(c)


def test_a_prompt_that_changes_nothing_is_neutral():
    fn = model({"": "separate", "SYS ": "separate"})
    c = prompt_cost(SPEC, fn, "SYS ")
    assert c.cost == pytest.approx(0.0)
    assert c.ari == pytest.approx(1.0)
    assert c.neutral


def test_a_prompt_that_helps_but_restructures_is_not_called_neutral():
    """The distinction that matters: improving accuracy is not neutrality.

    Measured analogue: a JSON-formatting instruction reached 0.950 accuracy while
    its partition scored ARI 0.0000 against the no-prompt condition. It helped and
    it intervened, and a report that called that neutral would be wrong.
    """
    fn = model({"": "half", "SYS ": "separate"})
    c = prompt_cost(SPEC, fn, "SYS ")
    assert c.cost < 0
    assert c.ari < 0.99
    assert not c.neutral
