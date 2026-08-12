"""Does the symmetry survive when the context grows?

The claim under test: adding bulk to a context cannot break the symmetry a fact
carries, because self-attention depends on that structure, so the symmetry rotates
rather than breaks and the rotation is bounded by entropy.

That predicts something specific and falsifiable. If the symmetry merely rotates,
a detector built on it keeps working as context grows even if individual answers
shift. If instead the symmetry degrades, detector performance decays with bulk.
Those are different curves and this measures which one occurs.

It is also the robustness question the equivariance result most needs answered.
Every prompt behind the 0.9451 figure was under fifteen tokens, and a detector
that only works on short prompts is not a detector.

**Design.** The same injective facts, with N tokens of irrelevant prose prepended
for N across two orders of magnitude. At each N: accuracy, the collision
(equivariance) signal, the invariance signal, and the late-layer attention entropy
that the entropy-bound claim refers to.

**The confound this controls.** Distractor text is fixed across all entities and
all paraphrases at a given N, so any change is attributable to context length
rather than to distractor content. A per-item random distractor would confound
length with topic.

    python -m caustic.experiments.symmetry_under_bulk
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
from caustic.experiments.triangulate import FACTS

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BULK_TOKENS = (0, 32, 128, 512)

FILLER = (
    "Mechanical calculators, tide predictors, and looms that read punched cards all encoded "
    "procedures into physical arrangements of matter. Ocean currents move heat around the planet "
    "on timescales that dwarf weather, and the resulting redistribution sets the climate of "
    "entire continents. A language is a system of conventions that lets one mind reconstruct "
    "part of the state of another from a sequence of symbols. The printing press reduced the "
    "cost of copying a book by orders of magnitude. Photosynthesis converts light energy into "
    "chemical energy stored in sugars, taking in carbon dioxide and releasing oxygen. "
) * 12


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        .to(DEV)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers

    filler_ids = tok(FILLER, add_special_tokens=False).input_ids

    def run(text: str, want_entropy: bool = False):
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            out = model(ids, output_attentions=want_entropy)
        t1 = int(out.logits[0, -1].argmax())
        ent = None
        if want_entropy:
            es = []
            for l in range(L - 8, L):
                m = out.attentions[l][0, :, -1, :].mean(0).clamp_min(1e-30)
                es.append(float(-(m * m.log()).sum()))
            ent = float(np.mean(es))
        return t1, ent

    print(f"model={MODEL} bulk levels={BULK_TOKENS}")
    print("distractor prose is identical across entities and paraphrases at each level\n")

    hdr = f"{'bulk':>6} {'ctx':>6} {'acc':>6} {'invariance':>22} {'collision':>22} {'attn H':>8}"
    print(hdr)
    print("-" * len(hdr))

    for nbulk in BULK_TOKENS:
        prefix = tok.decode(filler_ids[:nbulk]) + " " if nbulk else ""
        rows, ctx_len = [], 0
        for rel, spec in FACTS.items():
            pairs = spec["pairs"]
            fwd, ents = {}, {}
            for e in pairs:
                answers, en = [], []
                for ti, t in enumerate(spec["forward"]):
                    a, h = run(prefix + t.format(e=e), want_entropy=(ti == 0))
                    answers.append(a)
                    if h is not None:
                        en.append(h)
                fwd[e] = answers
                ents[e] = float(np.mean(en)) if en else float("nan")
                if not ctx_len:
                    ctx_len = len(tok(prefix + spec["forward"][0].format(e=e)).input_ids)
            for e, ans in pairs.items():
                gold = tok(" " + ans, add_special_tokens=False).input_ids[0]
                mine = fwd[e]
                inv = float(np.mean([a == b for a, b in itertools.combinations(mine, 2)]))
                col = float(
                    np.mean([
                        np.mean([fwd[o][i] == mine[i] for o in pairs if o != e])
                        for i in range(len(mine))
                    ])
                )
                rows.append({"correct": int(mine[0] == gold), "inv": inv, "col": col, "H": ents[e]})

        c = [r for r in rows if r["correct"]]
        w = [r for r in rows if not r["correct"]]
        acc = len(c) / len(rows)
        H = np.nanmean([r["H"] for r in rows])
        if not c or not w:
            print(f"{nbulk:>6} {ctx_len:>6} {acc:>6.3f}   (all one class -- AUROC undefined) {H:>8.4f}")
            continue
        cells = []
        for key, orient in (("inv", -1), ("col", +1)):
            a = np.array([r[key] for r in c])
            b = np.array([r[key] for r in w])
            pt, lo_, hi_ = auroc_ci(
                np.concatenate([b, a]) * orient,
                np.concatenate([np.ones(len(b)), np.zeros(len(a))]),
                n_boot=4000, seed=SEED,
            )
            cells.append(f"{pt:.3f} [{lo_:.2f},{hi_:.2f}]")
        print(f"{nbulk:>6} {ctx_len:>6} {acc:>6.3f} {cells[0]:>22} {cells[1]:>22} {H:>8.4f}")

    print("\nreading")
    print("-------")
    print("If the symmetry rotates rather than breaks, the detector AUROCs hold as bulk")
    print("grows even where accuracy moves. If it degrades, both fall together.")
    print("Attention entropy rising with context is expected from length alone; the claim")
    print("is only supported if the detector holds WHILE entropy rises.")


if __name__ == "__main__":
    main()
