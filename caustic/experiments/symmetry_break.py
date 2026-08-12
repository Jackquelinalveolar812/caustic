"""Hallucination as symmetry breaking, tested without the gold answer.

A fact carries a group action, and a model that knows the fact must respect two
different halves of it:

    invariance     paraphrase the prompt and the answer must NOT change
    equivariance   swap the entity and the answer MUST change

A model retrieving from the prior fails the second while passing the first: it is
invariant where it should be equivariant, giving the same answer whichever entity
is named. That is the same failure the gradient test saw as an entity coupling of
0.93 against a control of 1.0, measured here with finite transformations instead
of derivatives, and without needing to know the correct answer.

**Why the no-gold property matters.** Both quantities are computed from the
model's own outputs under transformations of its input. If they predict error,
they are a detector that runs where no ground truth exists, which is the only
setting where a hallucination detector is useful.

**Prior art, stated plainly.** The invariance half is essentially self-consistency
and is close to semantic entropy, which is established work; it is included as the
baseline the equivariance half has to beat. The equivariance half — penalising a
model for giving the SAME answer to different entities — is the part this
framework implies and is what is under test.

    python -m caustic.experiments.symmetry_break
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
from caustic.experiments.coupling_gap import RELATIONS

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PARAPHRASES = {
    "capital": ["The capital of {e} is", "{e}'s capital is", "The capital city of {e} is",
                "The seat of government of {e} is", "Q: What is the capital of {e}? A:"],
    "language": ["The main language spoken in {e} is", "People in {e} mainly speak",
                 "The official language of {e} is", "In {e}, the language spoken is",
                 "Q: What language is spoken in {e}? A:"],
    "currency": ["The currency of {e} is the", "{e} uses a currency called the",
                 "The money used in {e} is the", "The official currency of {e} is the",
                 "Q: What is the currency of {e}? A:"],
    "continent": ["The country of {e} is located on the continent of",
                  "{e} is on the continent of", "Geographically, {e} lies in",
                  "The continent containing {e} is", "Q: Which continent is {e} in? A:"],
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
    for rel, (tpl, facts) in RELATIONS.items():
        tpls = PARAPHRASES[rel]
        answers = {e: [top1(t.format(e=e)) for t in tpls] for e in facts}
        for e, ans in facts.items():
            gold = tok(" " + facts[e], add_special_tokens=False).input_ids[0]
            mine = answers[e]
            # Invariance: agreement across paraphrases of the same fact.
            inv = float(np.mean([a == b for a, b in itertools.combinations(mine, 2)]))
            # Equivariance: for each paraphrase, how many OTHER entities receive the
            # same answer. High means the model is not distinguishing entities.
            collide = []
            for i in range(len(tpls)):
                others = [answers[o][i] for o in facts if o != e]
                collide.append(float(np.mean([a == mine[i] for a in others])))
            rows.append(
                {
                    "rel": rel,
                    "entity": e,
                    "correct": int(mine[0] == gold),
                    "invariance": inv,
                    "collision": float(np.mean(collide)),
                    "score": inv - float(np.mean(collide)),
                }
            )

    print(f"model={MODEL} relations={len(RELATIONS)} items={len(rows)}")
    print("invariance = agreement across paraphrases (should be HIGH if the fact is known)")
    print("collision  = fraction of OTHER entities given the same answer (should be LOW)")
    print("neither uses the gold answer\n")

    def block(rs, tag):
        c = [r for r in rs if r["correct"]]
        w = [r for r in rs if not r["correct"]]
        if not c or not w:
            print(f"{tag:<11} {len(c)} correct / {len(w)} wrong -- need both, skipped")
            return
        line = f"{tag:<11}"
        for key, orient in (("invariance", -1), ("collision", +1), ("score", -1)):
            a = np.array([r[key] for r in c])
            b = np.array([r[key] for r in w])
            s = np.concatenate([b, a]) * orient
            y = np.concatenate([np.ones(len(b)), np.zeros(len(a))])
            pt, lo_, hi_ = auroc_ci(s, y, n_boot=4000, seed=SEED)
            line += f"  {key} {a.mean():.3f}/{b.mean():.3f} AUROC {pt:.3f}[{lo_:.2f},{hi_:.2f}]"
        print(line)

    print("format: metric correct/wrong, AUROC oriented so predicting WRONG scores above 0.5\n")
    block(rows, "POOLED")
    for rel in RELATIONS:
        block([r for r in rows if r["rel"] == rel], rel)

    print("\nequivariance failure in the raw: how often distinct entities share an answer")
    print(f"{'relation':<11} {'n':>4} {'collision':>10} {'accuracy':>9}")
    print("-" * 36)
    for rel in RELATIONS:
        rs = [r for r in rows if r["rel"] == rel]
        print(
            f"{rel:<11} {len(rs):>4} {np.mean([r['collision'] for r in rs]):>10.4f} "
            f"{np.mean([r['correct'] for r in rs]):>9.3f}"
        )
    print("\nA model retrieving from the prior gives the same answer to many entities,")
    print("so collision is the direct measure of the equivariance the framework predicts.")


if __name__ == "__main__":
    main()
