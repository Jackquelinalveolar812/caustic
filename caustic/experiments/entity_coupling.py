"""Does the derivative separate the true equilibrium from the false one?

The framing this tests, stated precisely.

A model that scores candidates by probability alone cannot distinguish a true
answer from a plausible false one when both sit at comparable probability. Both
are equilibria of the internal competition and neither carries a marker saying
which corresponds to the world. The measured picture matches: on items answered
wrongly, the correct token sits at median rank 3 of 151936 and 88 of 100 have it
in the top ten. The model is not uncertain between them. It is content, which is
what an equilibrium means.

So the question is whether some quantity separates them that probability cannot
see. This experiment tests one candidate, and it is a derivative rather than a
distribution:

    coupling(c) = || d logit_c / d h_entity ||

the sensitivity of candidate token `c`'s logit to the hidden state at the entity
position. The reasoning is causal rather than statistical. A correct answer must
depend on which entity was named: "Paris" is the answer only because "France" is
in the context, so perturbing France's representation must move Paris's logit. A
hallucinated answer is drawn from the prior — it is whatever is frequent or
locally plausible, and it would be produced whichever country was named, so its
logit should be comparatively insensitive to the entity.

Probability is blind to this by construction. Two tokens can carry the same
logit while depending on entirely different parts of the context, and only a
derivative distinguishes them.

**The prediction, and what refutes it.** Among the top-k candidates on items the
model gets wrong, the correct token should show higher entity coupling than the
token the model actually chose. If coupling is equal, the derivative carries no
more information than the probability did, and this route closes.

**The control that decides it.** Coupling to the entity position is compared
against coupling to a non-entity position in the same prompt. A candidate whose
logit is simply sensitive to everything would score high on both; only a gap
between them is evidence of entity-specific dependence.

    python -m caustic.experiments.entity_coupling
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import auroc_ci
from caustic.experiments.recoverability_probe import CAPITALS

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 5
LAYER = 0
"""Couple to the hidden state entering the stack, so the derivative measures
dependence on the entity as represented before any mixing has occurred."""


def entity_span(tok, country: str) -> tuple[int, int]:
    """Token span of the country name inside 'The capital of {c} is'."""
    prefix = tok("The capital of", add_special_tokens=False).input_ids
    with_c = tok(f"The capital of {country}", add_special_tokens=False).input_ids
    return len(prefix), len(with_c)


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for c in sorted(CAPITALS):
        gold_id = tok(" " + CAPITALS[c], add_special_tokens=False).input_ids[0]
        prompt = f"The capital of {c} is"
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        lo, hi = entity_span(tok, c)
        if hi <= lo or hi > ids.shape[1]:
            continue

        emb = model.model.embed_tokens(ids).detach().clone().requires_grad_(True)
        out = model(inputs_embeds=emb)
        logits = out.logits[0, -1]
        top = torch.topk(logits, TOPK)
        chosen_id = int(top.indices[0])

        def coupling(token_id: int, span: tuple[int, int]) -> float:
            g = torch.autograd.grad(logits[token_id], emb, retain_graph=True)[0]
            return float(g[0, span[0] : span[1]].norm())

        # Control span: an equal number of non-entity tokens from the prefix.
        ctrl = (0, min(hi - lo, lo))
        cand = [int(i) for i in top.indices]
        if gold_id not in cand:
            cand.append(gold_id)

        for tid in cand:
            rows.append(
                {
                    "country": c,
                    "token": tok.decode(tid).strip(),
                    "is_gold": int(tid == gold_id),
                    "is_chosen": int(tid == chosen_id),
                    "model_correct": int(chosen_id == gold_id),
                    "logit": float(logits[tid]),
                    "ent": coupling(tid, (lo, hi)),
                    "ctrl": coupling(tid, ctrl),
                }
            )
        del emb, out, logits

    import collections

    n_items = len({r["country"] for r in rows})
    wrong = [r for r in rows if not r["model_correct"]]
    print(f"model={MODEL} entities={n_items} top-k={TOPK} candidates={len(rows)}")
    print(f"items answered wrongly: {len({r['country'] for r in wrong})}\n")

    def summarize(rs, tag):
        gold = np.array([r["ent"] for r in rs if r["is_gold"]])
        chosen = np.array([r["ent"] for r in rs if r["is_chosen"] and not r["is_gold"]])
        other = np.array([r["ent"] for r in rs if not r["is_gold"] and not r["is_chosen"]])
        gc = np.array([r["ctrl"] for r in rs if r["is_gold"]])
        cc = np.array([r["ctrl"] for r in rs if r["is_chosen"] and not r["is_gold"]])
        print(f"{tag}")
        print(f"  {'':16} {'entity coupling':>18} {'control coupling':>18} {'ratio':>8}")
        for name, e, k in (("gold", gold, gc), ("chosen (wrong)", chosen, cc)):
            if len(e) == 0:
                continue
            r = e.mean() / max(k.mean(), 1e-12)
            print(f"  {name:16} {e.mean():>10.4f} +/-{e.std():<6.4f} {k.mean():>10.4f} +/-{k.std():<6.4f} {r:>8.2f}")
        if len(other):
            print(f"  {'other top-k':16} {other.mean():>10.4f} +/-{other.std():<6.4f}")
        if len(gold) and len(chosen):
            s = np.concatenate([gold, chosen])
            y = np.concatenate([np.ones(len(gold)), np.zeros(len(chosen))])
            pt, lo_, hi_ = auroc_ci(s, y, n_boot=2000, seed=SEED)
            print(f"  AUROC gold vs chosen, by entity coupling: {pt:.4f} [{lo_:.4f}, {hi_:.4f}]")
            sl = np.concatenate(
                [
                    np.array([r["logit"] for r in rs if r["is_gold"]]),
                    np.array([r["logit"] for r in rs if r["is_chosen"] and not r["is_gold"]]),
                ]
            )
            pl, ll, hl = auroc_ci(sl, y, n_boot=2000, seed=SEED)
            print(f"  AUROC gold vs chosen, by LOGIT (the baseline):  {pl:.4f} [{ll:.4f}, {hl:.4f}]")
            print(f"  the derivative is informative only if it beats the logit")
        print()

    summarize(wrong, "ON ITEMS THE MODEL GETS WRONG")
    summarize([r for r in rows if r["model_correct"]], "ON ITEMS THE MODEL GETS RIGHT")

    print("reading")
    print("-------")
    print("Probability cannot separate two candidates at similar logits. If the gold")
    print("token shows higher entity coupling than the token actually chosen, a")
    print("derivative distinguishes the true equilibrium from the false one where the")
    print("distribution cannot, and that is a signal the model does not currently use.")
    print("AUROC near 0.5, or below the logit baseline, closes this route.")


if __name__ == "__main__":
    main()
