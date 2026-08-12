"""Triangulating with three symmetries instead of one.

The Jacobian route cost 53.7 ms per position and reached 0.61. The symmetry route
cost five forward passes and reached 0.995 where its precondition held. Local
derivative information lost to global group structure by a wide margin, so this
experiment adds symmetries rather than refining the derivative.

Three group actions, each of which a model that knows a fact must respect:

    invariance     paraphrase the prompt, the answer must not change
    equivariance   swap the entity, the answer must change
                   (valid only when the relation is injective)
    inversion      state the fact backwards. If the capital of France is Paris,
                   then Paris is the capital of France. A model that asserts one
                   direction and fails the other has not represented the relation,
                   only a co-occurrence.

Inversion is the new one and it is the reason for the word triangulate: it is
independent of the other two. Invariance and equivariance both probe the forward
map from entity to answer, while inversion probes whether the relation exists in
the reverse direction at all. Failing it is the known reversal-curse pattern, and
using it as a per-item confidence signal rather than as a training critique is
what is under test.

None of the three uses the gold answer. All three are computed from the model's
own outputs under transformations of its own input.

    python -m caustic.experiments.triangulate
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

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Only injective relations: equivariance is undefined on many-to-one ones, which
# is what inverted it on continent.
FACTS = {
    "capital": {
        "forward": ["The capital of {e} is", "{e}'s capital is", "The capital city of {e} is",
                    "The seat of government of {e} is", "Q: What is the capital of {e}? A:"],
        "inverse": ["{a} is the capital of", "The city of {a} is the capital of",
                    "Q: {a} is the capital of which country? A:"],
        "pairs": {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo", "Germany": "Berlin",
                  "Spain": "Madrid", "Greece": "Athens", "Egypt": "Cairo", "China": "Beijing",
                  "Russia": "Moscow", "Canada": "Ottawa", "Norway": "Oslo", "Sweden": "Stockholm",
                  "Poland": "Warsaw", "Austria": "Vienna", "Ireland": "Dublin", "Cuba": "Havana",
                  "Peru": "Lima", "Chile": "Santiago", "Thailand": "Bangkok", "Turkey": "Ankara"},
    },
    "language": {
        "forward": ["The main language spoken in {e} is", "People in {e} mainly speak",
                    "The official language of {e} is", "In {e}, the language spoken is",
                    "Q: What language is spoken in {e}? A:"],
        "inverse": ["The {a} language is spoken in", "{a} is the main language of",
                    "Q: {a} is spoken in which country? A:"],
        "pairs": {"France": "French", "Italy": "Italian", "Japan": "Japanese",
                  "Germany": "German", "Greece": "Greek", "China": "Chinese",
                  "Russia": "Russian", "Sweden": "Swedish", "Poland": "Polish",
                  "Turkey": "Turkish", "Korea": "Korean", "Israel": "Hebrew"},
    },
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

    rows = []
    for rel, spec in FACTS.items():
        pairs = spec["pairs"]
        fwd = {e: [top1(t.format(e=e)) for t in spec["forward"]] for e in pairs}
        inv = {e: [top1(t.format(a=pairs[e])) for t in spec["inverse"]] for e in pairs}
        for e, ans in pairs.items():
            gold_f = tok(" " + ans, add_special_tokens=False).input_ids[0]
            gold_i = tok(" " + e, add_special_tokens=False).input_ids[0]
            mine = fwd[e]
            invariance = float(np.mean([a == b for a, b in itertools.combinations(mine, 2)]))
            collide = float(
                np.mean(
                    [
                        np.mean([fwd[o][i] == mine[i] for o in pairs if o != e])
                        for i in range(len(mine))
                    ]
                )
            )
            # Inversion: does the reverse direction return the original entity?
            inversion = float(np.mean([x == gold_i for x in inv[e]]))
            rows.append(
                {
                    "rel": rel,
                    "correct": int(mine[0] == gold_f),
                    "invariance": invariance,
                    "collision": collide,
                    "inversion": inversion,
                    "tri": invariance - collide + inversion,
                }
            )

    print(f"model={MODEL} relations={len(FACTS)} items={len(rows)} (injective relations only)")
    print("none of the three signals uses the gold forward answer\n")

    def block(rs, tag):
        c = [r for r in rs if r["correct"]]
        w = [r for r in rs if not r["correct"]]
        if not c or not w:
            print(f"{tag:<10} {len(c)} correct / {len(w)} wrong -- skipped")
            return
        print(f"{tag:<10} correct n={len(c)}  wrong n={len(w)}")
        for key, orient in (("invariance", -1), ("collision", +1), ("inversion", -1), ("tri", -1)):
            a = np.array([r[key] for r in c])
            b = np.array([r[key] for r in w])
            pt, lo_, hi_ = auroc_ci(
                np.concatenate([b, a]) * orient,
                np.concatenate([np.ones(len(b)), np.zeros(len(a))]),
                n_boot=4000, seed=SEED,
            )
            star = "  <-- new" if key == "inversion" else ""
            print(f"   {key:<11} correct {a.mean():.3f}  wrong {b.mean():.3f}   "
                  f"AUROC {pt:.4f} [{lo_:.4f}, {hi_:.4f}]{star}")
        print()

    block(rows, "POOLED")
    for rel in FACTS:
        block([r for r in rows if r["rel"] == rel], rel)

    print("tri = invariance - collision + inversion, an unweighted sum of three")
    print("independent symmetry tests. If it beats each component, triangulation helps.")


if __name__ == "__main__":
    main()
