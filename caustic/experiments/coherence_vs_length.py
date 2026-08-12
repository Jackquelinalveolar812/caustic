"""Is orbit collapse cured by context length, or by context coherence?

The previous run established that at five tokens the model merges distinct
entities onto shared answers — twenty countries producing fifteen capitals with a
largest orbit of four — and that adding 128 tokens of unrelated prose resolves
every orbit to a singleton while accuracy rises from 0.550 to 0.850.

That leaves two incompatible explanations and they imply different interventions.

    length      any prefix of sufficient length works; the effect is about
                position, attention denominator, or simply not being at the
                start of a sequence
    coherence   only well-formed language works; the effect is about the model
                entering a distributional regime it was trained in

Token count is therefore held FIXED and only the character of the prefix varies:

    prose       coherent English on unrelated subjects
    shuffled    the same tokens as prose, order destroyed
    repeat      one innocuous token repeated
    random      uniform random token ids from the vocabulary

If all four resolve the orbits, the cause is length. If only prose does, the
cause is coherence. If prose and shuffled both work but repeat and random do not,
the cause is lexical diversity rather than syntax, which is a third answer neither
hypothesis predicted.

The distinction decides which pretraining augmentation could matter: augmenting
for format diversity is a different intervention from augmenting for context
length.

    python -m caustic.experiments.coherence_vs_length
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.experiments.orbit_invariant import adjusted_rand
from caustic.experiments.triangulate import FACTS

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NTOK = 128

PROSE = (
    "Mechanical calculators, tide predictors, and looms that read punched cards all encoded "
    "procedures into physical arrangements of matter. Ocean currents move heat around the planet "
    "on timescales that dwarf weather, and the resulting redistribution sets the climate of "
    "entire continents. A language is a system of conventions that lets one mind reconstruct "
    "part of the state of another from a sequence of symbols. Photosynthesis converts light "
    "energy into chemical energy stored in sugars, taking in carbon dioxide and releasing oxygen "
    "as a by-product of the reaction that sustains almost every food chain on the surface. "
) * 4


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    V = model.config.vocab_size

    prose_ids = tok(PROSE, add_special_tokens=False).input_ids[:NTOK]
    assert len(prose_ids) == NTOK, f"prose too short: {len(prose_ids)}"
    shuffled_ids = list(rng.permutation(prose_ids))
    repeat_ids = tok(" the", add_special_tokens=False).input_ids * NTOK
    repeat_ids = repeat_ids[:NTOK]
    # Sample from the low id range to avoid unused/special regions of the vocab.
    random_ids = [int(x) for x in rng.integers(1000, 100000, NTOK)]

    prefixes = {
        "none": [],
        "prose": prose_ids,
        "shuffled": shuffled_ids,
        "repeat": repeat_ids,
        "random": random_ids,
    }

    def top1(prefix_ids, text: str) -> int:
        ids = prefix_ids + tok(text, add_special_tokens=False).input_ids
        t = torch.tensor([ids], device=DEV)
        with torch.no_grad():
            return int(model(t).logits[0, -1].argmax())

    print(f"model={MODEL} fixed prefix length={NTOK} tokens (except 'none')")
    print("only the CHARACTER of the prefix varies; the count is held fixed\n")

    for rel, spec in FACTS.items():
        pairs = list(spec["pairs"])
        tpl = spec["forward"][0]
        gold = [tok(" " + spec["pairs"][e], add_special_tokens=False).input_ids[0] for e in pairs]
        print(f"=== {rel} ({len(pairs)} entities) ===")
        hdr = f"{'prefix':<10} {'acc':>6} {'n_distinct':>11} {'largest orbit':>14} {'ARI vs none':>12}"
        print(hdr)
        print("-" * len(hdr))
        base = None
        for name, pid in prefixes.items():
            ans = [top1(pid, tpl.format(e=e)) for e in pairs]
            if base is None:
                base = ans
            vals, counts = np.unique(ans, return_counts=True)
            acc = float(np.mean([a == g for a, g in zip(ans, gold)]))
            ari = adjusted_rand(base, ans)
            print(f"{name:<10} {acc:>6.3f} {len(vals):>11} {counts.max():>14} {ari:>12.4f}")
        print()

    print("reading")
    print("-------")
    print("All four prefixes resolving the orbits  -> the cause is LENGTH.")
    print("Only prose resolving them              -> the cause is COHERENCE.")
    print("prose and shuffled but not repeat      -> the cause is lexical DIVERSITY,")
    print("                                          which neither hypothesis predicted.")
    print("\nARI is against the no-prefix condition, so a value near 0 means the prefix")
    print("changed which entities share an answer, and near 1 means it did not.")


if __name__ == "__main__":
    main()
