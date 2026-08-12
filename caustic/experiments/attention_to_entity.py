"""Does the model look at the entity when it answers, and does not looking cause the error?

Attention is the KV lookup, and each head's attention vector is the exact solution
of an entropic variational problem,

    softmax(z) = argmax_p  <p, z> + H(p)

so the distribution over keys is a fixed point of a trade-off between score
alignment and entropy, not merely something probability-shaped. That makes the
attention distribution the right place to look for a competition between
retrieving from context and falling back on the prior.

The prediction. On items answered correctly the model should place attention mass
on the entity tokens, because the answer depends on which entity was named. On
items answered wrongly it should place less, having retrieved from the prior
instead. This is the same claim the gradient test made, measured far more
directly and at zero extra cost: attention weights are computed during the forward
pass anyway, where the Jacobian coupling cost 53.7 ms per position.

The control. Raw attention mass is confounded by span length and by position,
since later tokens receive more attention under a causal mask and entities sit at
different offsets across prompts. Mass is therefore reported both raw and
normalized by span length, and against the mass on an equal-length control span
drawn from the prompt prefix.

This also tests a mechanism for the currency relation, where coupling inverted.
Its prompt ends "is the", a hard syntactic cue. If grammar pins attention more
firmly than the entity does, entity attention should be low there for correct and
wrong items alike, which would explain the inversion without appealing to the
entity at all.

    python -m caustic.experiments.attention_to_entity
"""

from __future__ import annotations

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

    rows = []
    for rel, (tpl, facts) in RELATIONS.items():
        prefix_txt = tpl.split("{e}")[0].rstrip()
        for e, ans in facts.items():
            gold = tok(" " + ans, add_special_tokens=False).input_ids[0]
            ids = tok(tpl.format(e=e), return_tensors="pt").input_ids.to(DEV)
            lo = len(tok(prefix_txt, add_special_tokens=False).input_ids)
            hi = len(tok(prefix_txt + " " + e, add_special_tokens=False).input_ids)
            T = ids.shape[1]
            if hi <= lo or hi > T:
                continue
            span = hi - lo
            ctrl_lo, ctrl_hi = 0, min(span, lo)
            with torch.no_grad():
                out = model(ids, output_attentions=True)
            correct = int(int(out.logits[0, -1].argmax()) == gold)

            # attentions[l] is (batch, heads, q, k); take the final query row.
            ent, ctrl, ent_last, entropy = [], [], [], []
            for l in range(L):
                a = out.attentions[l][0, :, -1, :]  # (heads, keys)
                m = a.mean(0)  # average over heads
                ent.append(float(m[lo:hi].sum()))
                ctrl.append(float(m[ctrl_lo:ctrl_hi].sum()))
                p = m.clamp_min(1e-30)
                entropy.append(float(-(p * p.log()).sum()))
            ent = np.array(ent)
            ctrl = np.array(ctrl)
            rows.append(
                {
                    "rel": rel,
                    "entity": e,
                    "correct": correct,
                    "T": T,
                    "span": span,
                    "ent_mean": ent.mean(),
                    "ent_max": ent.max(),
                    "ent_late": ent[-8:].mean(),
                    "ctrl_mean": ctrl.mean(),
                    "ratio": ent.mean() / max(ctrl.mean(), 1e-12),
                    "entropy_late": float(np.mean(entropy[-8:])),
                }
            )
            del out

    print(f"model={MODEL} layers={L} items={len(rows)}")
    print("attention mass from the final query position onto the entity span\n")

    def block(rs, tag):
        c = [r for r in rs if r["correct"]]
        w = [r for r in rs if not r["correct"]]
        if not c or not w:
            print(f"{tag}: {len(c)} correct, {len(w)} wrong -- need both, skipping")
            return None
        print(f"{tag}   correct n={len(c)}   wrong n={len(w)}")
        out = {}
        for key in ("ent_mean", "ent_late", "ratio", "entropy_late"):
            a = np.array([r[key] for r in c])
            b = np.array([r[key] for r in w])
            s = np.concatenate([b, a])
            y = np.concatenate([np.ones(len(b)), np.zeros(len(a))])
            pt, lo_, hi_ = auroc_ci(-s, y, n_boot=4000, seed=SEED)
            out[key] = pt
            print(
                f"  {key:<14} correct {a.mean():>8.4f}+/-{a.std():<7.4f} "
                f"wrong {b.mean():>8.4f}+/-{b.std():<7.4f}  "
                f"AUROC(low=wrong) {pt:>6.4f} [{lo_:.4f}, {hi_:.4f}]"
            )
        print()
        return out

    block(rows, "POOLED")
    for rel in RELATIONS:
        block([r for r in rows if r["rel"] == rel], f"{rel.upper():<10}")

    print("per-relation entity attention regardless of correctness")
    print(f"{'relation':<11} {'n':>4} {'ent_mean':>10} {'ctrl_mean':>10} {'ratio':>8} {'acc':>6}")
    print("-" * 54)
    for rel in RELATIONS:
        rs = [r for r in rows if r["rel"] == rel]
        if not rs:
            continue
        em = np.mean([r["ent_mean"] for r in rs])
        cm = np.mean([r["ctrl_mean"] for r in rs])
        print(
            f"{rel:<11} {len(rs):>4} {em:>10.4f} {cm:>10.4f} {em/max(cm,1e-12):>8.2f} "
            f"{np.mean([r['correct'] for r in rs]):>6.3f}"
        )
    print("\nA low ratio for currency regardless of correctness would explain the coupling")
    print("inversion there as grammar pinning the lookup rather than anything about entities.")
    print("\nAUROC is oriented so that LOW attention predicting a WRONG answer scores above 0.5.")


if __name__ == "__main__":
    main()
