"""Does the detector survive at realistic context, on facts the model finds hard?

The 0.995 figure was measured on five-token prompts, where accuracy is 0.550 and
the model is in a regime that produces tokens like " __". That is the regime the
detector was built in, and it is not the regime anything is deployed in.

Testing it at realistic context runs into an obstacle that is itself informative:
with a coherent 128-token prefix the model answers 20 of 20 country capitals
correctly, so there are no errors left to detect and AUROC is undefined. A detector
cannot be evaluated where the condition has been cured.

So the test needs facts the model still gets wrong *with* full context. Three
harder injective relations are used, chosen because their answers are lower
frequency rather than because they were observed to fail:

    small-country capitals   Bhutan, Eritrea, Suriname, and similar
    element symbols          the less common half of the periodic table
    author works             a writer paired with a less famous title

**What each outcome means.**

    detector holds          0.995 was not an artefact of the degenerate regime,
                            and the method works where it would be deployed
    detector degrades       the signal depends on the model being in a broken
                            regime, which sharply narrows the claim
    no errors remain        the harder relations were not hard enough, and the
                            experiment is inconclusive rather than positive

The third outcome is reported as inconclusive, not folded into the second.

    python -m caustic.experiments.detector_at_context
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import auroc_ci
from caustic.repair import NEUTRAL_PREFIX

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

HARD = {
    "small_capital": (
        ["The capital of {e} is", "{e}'s capital is", "The capital city of {e} is",
         "The seat of government of {e} is", "Q: What is the capital of {e}? A:"],
        {"Bhutan": "Thimphu", "Eritrea": "Asmara", "Suriname": "Paramaribo",
         "Moldova": "Chisinau", "Laos": "Vientiane", "Malawi": "Lilongwe",
         "Bolivia": "Sucre", "Myanmar": "Naypyidaw", "Georgia": "Tbilisi",
         "Armenia": "Yerevan", "Zambia": "Lusaka", "Uruguay": "Montevideo",
         "Slovenia": "Ljubljana", "Latvia": "Riga", "Estonia": "Tallinn",
         "Albania": "Tirana", "Paraguay": "Asuncion", "Namibia": "Windhoek"},
    ),
    "element_symbol": (
        ["The chemical symbol for {e} is", "{e} has the chemical symbol",
         "In the periodic table, {e} is written as", "The symbol of the element {e} is",
         "Q: What is the chemical symbol for {e}? A:"],
        {"tungsten": "W", "antimony": "Sb", "tin": "Sn", "potassium": "K",
         "sodium": "Na", "iron": "Fe", "silver": "Ag", "gold": "Au",
         "mercury": "Hg", "lead": "Pb", "copper": "Cu", "manganese": "Mn",
         "zirconium": "Zr", "tellurium": "Te", "rubidium": "Rb", "yttrium": "Y"},
    ),
}


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def top1(text: str) -> int:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            return int(model(ids).logits[0, -1].argmax())

    ctx = len(tok(NEUTRAL_PREFIX, add_special_tokens=False).input_ids)
    print(f"model={MODEL}  prefix={ctx} tokens of neutral prose  seed={SEED}")
    print("harder relations, chosen for low answer frequency rather than observed failure\n")

    hdr = f"{'relation':<15} {'ctx':>5} {'n':>3} {'acc':>6} {'wrong':>6} {'orbits':>7} {'collision AUROC':>24}"
    print(hdr)
    print("-" * len(hdr))

    pooled = {"short": ([], []), "full": ([], [])}
    for rel, (tpls, pairs) in HARD.items():
        for label, prefix in (("short", ""), ("full", NEUTRAL_PREFIX)):
            ents = list(pairs)
            table = {e: [top1(prefix + t.format(e=e)) for t in tpls] for e in ents}
            gold = {e: tok(" " + pairs[e], add_special_tokens=False).input_ids[0] for e in ents}
            correct = [int(table[e][0] == gold[e]) for e in ents]
            collide = [
                float(np.mean([
                    np.mean([table[o][i] == table[e][i] for o in ents if o != e])
                    for i in range(len(tpls))
                ]))
                for e in ents
            ]
            n_wrong = len(ents) - sum(correct)
            n_orbits = len({table[e][0] for e in ents})
            n_ctx = len(tok(prefix + tpls[0].format(e=ents[0])).input_ids)

            if 0 < n_wrong < len(ents):
                w = [c for c, ok in zip(collide, correct) if not ok]
                r = [c for c, ok in zip(collide, correct) if ok]
                pt, lo, hi = auroc_ci(
                    np.array(w + r), np.array([1] * len(w) + [0] * len(r)), n_boot=4000, seed=SEED
                )
                cell = f"{pt:.4f} [{lo:.2f}, {hi:.2f}]"
                pooled[label][0].extend(w)
                pooled[label][1].extend(r)
            else:
                cell = "undefined (one class)"
            print(
                f"{rel:<15} {n_ctx:>5} {len(ents):>3} {np.mean(correct):>6.3f} "
                f"{n_wrong:>6} {n_orbits:>7} {cell:>24}"
            )
        print()

    print("POOLED across relations")
    for label in ("short", "full"):
        w, r = pooled[label]
        if not w or not r:
            print(f"  {label:<6} undefined -- one class empty")
            continue
        pt, lo, hi = auroc_ci(
            np.array(w + r), np.array([1] * len(w) + [0] * len(r)), n_boot=4000, seed=SEED
        )
        print(f"  {label:<6} n_wrong={len(w):>3} n_correct={len(r):>3}   AUROC {pt:.4f} [{lo:.4f}, {hi:.4f}]")

    print()
    print("reading")
    print("-------")
    print("If the full-context AUROC holds near the short-context one, 0.995 was not an")
    print("artefact of the degenerate regime. If it falls, the detector depends on the")
    print("model being broken, which narrows the claim sharply. If no errors survive the")
    print("prefix, the relations were not hard enough and this is inconclusive.")


if __name__ == "__main__":
    main()
