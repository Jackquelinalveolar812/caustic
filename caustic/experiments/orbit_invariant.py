"""Does the orbit structure survive growing context?

The rotation claim, stated so it can be falsified: adding bulk to a context cannot
break the symmetry a fact carries, so the group action rotates rather than breaks.

A rotation preserves ORBIT STRUCTURE. Individual answers may move, but which
entities land together must not change. So the quantity to track is not any
particular answer, it is the PARTITION of entities induced by the model's answers
— the connected components of the relation "these two entities receive the same
answer". That is H0 of the answer-equivalence relation: a topological invariant
measuring a group-theoretic property.

**Why this design and not the previous one.** The previous run tracked detector
AUROC across context length and could not conclude anything, because accuracy rose
with bulk and the wrong-item count fell from 15 to 4, so the intervals widened for
reasons having nothing to do with symmetry. The partition is computed over ALL
entities regardless of correctness, so no class can collapse and no gold answer is
needed. The confound is removed by construction rather than controlled for.

**Discriminating prediction.**

    rotation        partition preserved across bulk levels (ARI near 1), while the
                    specific answers may change
    breaking        partition fragments or merges (ARI falls)
    inertia         both partition AND answers unchanged, which would mean context
                    does nothing and the claim is vacuous rather than supported

The third outcome is why answer churn is reported alongside the partition metric.
A high ARI with zero answer churn is not evidence of rotation; it is evidence that
nothing happened.

Adjusted Rand index is used rather than raw agreement because it corrects for
chance agreement, which matters when most entities receive distinct answers and
any two partitions look similar by default.

    python -m caustic.experiments.orbit_invariant
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.experiments.triangulate import FACTS

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BULK = (0, 32, 128, 512)

FILLER = (
    "Mechanical calculators, tide predictors, and looms that read punched cards all encoded "
    "procedures into physical arrangements of matter. Ocean currents move heat around the planet "
    "on timescales that dwarf weather. A language is a system of conventions that lets one mind "
    "reconstruct part of the state of another from a sequence of symbols. Photosynthesis converts "
    "light energy into chemical energy stored in sugars. "
) * 16


def adjusted_rand(a: list[int], b: list[int]) -> float:
    """Adjusted Rand index between two labellings of the same items.

    Corrects for chance agreement. Without the correction two partitions that are
    both mostly-singletons score near 1 by default, which is exactly the regime
    here since most entities receive distinct answers.
    """
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    c = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            c[i, j] = np.sum((a == x) & (b == y))

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_ij = comb2(c).sum()
    sum_i = comb2(c.sum(axis=1)).sum()
    sum_j = comb2(c.sum(axis=0)).sum()
    n = comb2(len(a))
    expected = sum_i * sum_j / n if n else 0.0
    maximum = 0.5 * (sum_i + sum_j)
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    filler_ids = tok(FILLER, add_special_tokens=False).input_ids

    def top1(text: str) -> int:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            return int(model(ids).logits[0, -1].argmax())

    for rel, spec in FACTS.items():
        pairs = list(spec["pairs"])
        tpl = spec["forward"][0]
        answers: dict[int, list[int]] = {}
        acc: dict[int, float] = {}
        for nb in BULK:
            prefix = tok.decode(filler_ids[:nb]) + " " if nb else ""
            answers[nb] = [top1(prefix + tpl.format(e=e)) for e in pairs]
            gold = [tok(" " + spec["pairs"][e], add_special_tokens=False).input_ids[0] for e in pairs]
            acc[nb] = float(np.mean([a == g for a, g in zip(answers[nb], gold)]))

        print(f"\n=== {rel}  ({len(pairs)} entities) ===")
        print(f"{'bulk':>6} {'acc':>6} {'n_distinct':>11} {'largest orbit':>14}")
        print("-" * 42)
        for nb in BULK:
            vals, counts = np.unique(answers[nb], return_counts=True)
            print(f"{nb:>6} {acc[nb]:>6.3f} {len(vals):>11} {counts.max():>14}")

        print(f"\n{'pair':>12} {'ARI':>8} {'answers changed':>17}")
        print("-" * 40)
        base = BULK[0]
        for nb in BULK[1:]:
            ari = adjusted_rand(answers[base], answers[nb])
            churn = float(np.mean([x != y for x, y in zip(answers[base], answers[nb])]))
            print(f"{f'{base}->{nb}':>12} {ari:>8.4f} {churn:>17.3f}")
        for i in range(len(BULK) - 1):
            a, b = BULK[i], BULK[i + 1]
            ari = adjusted_rand(answers[a], answers[b])
            churn = float(np.mean([x != y for x, y in zip(answers[a], answers[b])]))
            print(f"{f'{a}->{b}':>12} {ari:>8.4f} {churn:>17.3f}  (adjacent)")

    print("\nreading")
    print("-------")
    print("ARI near 1 with high answer churn is ROTATION: the entities keep landing")
    print("together while the labels move. That is the claim.")
    print("ARI near 1 with churn near 0 is INERTIA: nothing happened, and the claim is")
    print("vacuous rather than supported.")
    print("ARI falling is BREAKING: the orbit structure did not survive the context.")


if __name__ == "__main__":
    main()
