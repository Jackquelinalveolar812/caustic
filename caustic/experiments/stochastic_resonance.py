"""Is the gold token a sub-threshold signal that noise can lift?

Section 10 measured that on wrong items the correct token trails the chosen one by
0.2526 standard deviations among roughly thirty live candidates. That is the
definition of a sub-threshold signal: present, ordered correctly in some sense, and
below the level at which the readout commits to it.

Stochastic resonance is the phenomenon in which a nonlinear system's response to
such a signal *improves* when moderate noise is added, and degrades again when too
much is added. It predicts a specific and falsifiable shape.

    monotone increase   noise is smoothing something; not resonance
    monotone decrease   no resonance; noise only destroys
    inverted U          resonance, with an optimum at intermediate noise

Only the third is evidence. The peak must also exceed the zero-noise baseline, or
the curve is a decline with a flat start.

**Where the noise is injected.** Into the input embeddings, scaled relative to
their own standard deviation, so `sigma` is dimensionless and comparable across
models. Injecting into logits directly would be a different and much weaker claim:
it would test whether adding noise to a ranking changes the ranking, which is
trivially true.

**Why averaging is required.** A single noisy pass is a worse estimator than a
clean one. Resonance appears in the *expectation*, so each level is evaluated by
majority vote over `n_samples` draws, which Section 10's ensemble result already
established is the rule that survives a positive exponent.

    python -m caustic.experiments.stochastic_resonance
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.experiments.triangulate import FACTS

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80)
N_SAMPLES = 16


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"model={MODEL}  samples per level={N_SAMPLES}  seed={SEED}")
    print("noise is added to input embeddings, scaled by their own std, so sigma is")
    print("dimensionless. each level is scored by majority vote over the samples.\n")

    for rel, spec in FACTS.items():
        pairs = spec["pairs"]
        tpl = spec["forward"][0]
        print(f"=== {rel} ({len(pairs)} entities) ===")
        print(f"{'sigma':>7} {'accuracy':>9} {'vs baseline':>12} {'unique answers':>15}")
        print("-" * 48)
        base_acc = None
        curve = []
        for sigma in SIGMAS:
            gen = torch.Generator(device=DEV).manual_seed(SEED)
            ok, answers = [], []
            for e, ans in pairs.items():
                gold = tok(" " + ans, add_special_tokens=False).input_ids[0]
                ids = tok(tpl.format(e=e), return_tensors="pt").input_ids.to(DEV)
                emb = model.model.embed_tokens(ids)
                scale = float(emb.std()) * sigma
                votes = Counter()
                reps = 1 if sigma == 0.0 else N_SAMPLES
                for _ in range(reps):
                    noisy = emb if sigma == 0.0 else emb + torch.randn(
                        emb.shape, generator=gen, device=DEV, dtype=emb.dtype
                    ) * scale
                    with torch.no_grad():
                        votes[int(model(inputs_embeds=noisy).logits[0, -1].argmax())] += 1
                pick = votes.most_common(1)[0][0]
                ok.append(pick == gold)
                answers.append(pick)
            acc = float(np.mean(ok))
            curve.append(acc)
            if base_acc is None:
                base_acc = acc
            print(f"{sigma:>7.2f} {acc:>9.3f} {acc - base_acc:>+12.3f} {len(set(answers)):>15}")

        arr = np.array(curve)
        peak = int(arr.argmax())
        rising = peak > 0
        falling = peak < len(arr) - 1 and arr[-1] < arr[peak]
        gain = arr[peak] - arr[0]
        print()
        if rising and falling and gain > 0:
            print(f"  INVERTED U: peak {arr[peak]:.3f} at sigma={SIGMAS[peak]}, "
                  f"baseline {arr[0]:.3f}, gain {gain:+.3f}, falls to {arr[-1]:.3f}")
            print("  this is the resonance signature")
        elif peak == 0:
            print(f"  MONOTONE DECLINE from {arr[0]:.3f}. No resonance; noise only destroys.")
        elif not falling:
            print(f"  STILL RISING at the largest sigma tested. Not resonance as measured;")
            print("  extend the sweep before drawing any conclusion.")
        else:
            print(f"  peak {arr[peak]:.3f} at sigma={SIGMAS[peak]} does not exceed baseline "
                  f"{arr[0]:.3f}. No resonance.")
        print()


if __name__ == "__main__":
    main()
