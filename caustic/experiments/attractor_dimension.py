"""Attractor dimension of a live transformer, against a control.

Computes the Kaplan-Yorke dimension of the token cocycle at each block, which is
the one number the three linearizations jointly produce: Jacobian (local),
composed along a trajectory into a Lyapunov spectrum (chaos), collapsed by the
Kaplan-Yorke formula into a fractal dimension.

Two things are being asked.

1. Is D_KY much smaller than the width D? If so the hidden-state dynamics occupy
   far fewer directions than the model carries, and 2*D_KY+1 is a Takens-style
   sufficient reconstruction size — a bound with a number in it rather than a
   tuned constant.

2. Does D_KY separate grounded text from its shuffled control? Every other
   summary so far has either failed to separate or separated only in sign.

The hypothesis under which these theorems apply is not satisfied here and the
report says so: Oseledets and Takens require a measure-preserving dynamical
system, and a finite non-autonomous trajectory driven by input tokens is not
obviously one.

    python -m caustic.experiments.attractor_dimension
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.attractor import spectrum_report
from caustic.cocycle import lyapunov_spectrum
from caustic.jacobian import exact_jacobian

MODEL = "distilgpt2"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

TEXT = (
    "The capital of France is Paris. The Seine flows through the city. "
    "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911. "
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure. "
    "The mitochondrion is the organelle that produces most of the cell's ATP."
)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
        .to(DEV)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    ids = tok(TEXT, return_tensors="pt").input_ids.to(DEV)
    T = ids.shape[1]
    t0 = T // 4
    L = len(model.transformer.h)
    conditions = {"grounded": ids, "shuffled": ids[:, rng.permutation(T)].contiguous()}

    print(f"model={MODEL} D=768 T={T} blocks={L} cocycle steps={T - t0} device={DEV} seed={SEED}")
    print("control = same token multiset, shuffled order\n")

    hdr = (
        f"{'block':>5} {'cond':>9} {'lam_max':>9} {'sum':>10} {'n_pos':>6} "
        f"{'D_KY':>8} {'D/D_KY':>8} {'2D_KY+1':>8} {'h_KS':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    results: dict[tuple[int, str], dict] = {}
    for l in range(L):
        for cond, seq in conditions.items():
            with torch.no_grad():
                hs = model(seq, output_hidden_states=True).hidden_states
            h_in = hs[l].detach()
            Js = [exact_jacobian(model.transformer.h[l], h_in, t).cpu() for t in range(t0, T)]
            r = spectrum_report(lyapunov_spectrum(Js))
            results[(l, cond)] = r
            print(
                f"{l:>5} {cond:>9} {r['lambda_max']:>9.4f} {r['sum']:>10.1f} "
                f"{int(r['n_positive']):>6} {r['kaplan_yorke']:>8.2f} "
                f"{r['compression_ratio']:>8.1f} {int(r['embedding_bound']):>8} "
                f"{r['metric_entropy']:>8.3f}"
            )
        print()

    print("separation (grounded - shuffled) as a fraction of the grounded value:")
    for key in ("kaplan_yorke", "metric_entropy", "sum", "n_positive"):
        d = []
        for l in range(L):
            g, s = results[(l, "grounded")][key], results[(l, "shuffled")][key]
            d.append((g - s) / abs(g) if g != 0 else 0.0)
        arr = np.array(d)
        same_sign = int(np.sum(arr > 0)) if arr.mean() > 0 else int(np.sum(arr < 0))
        print(
            f"  {key:<16} {np.array2string(arr, precision=3)}  "
            f"|mean| {abs(arr.mean()):.4f}  same-sign {same_sign}/{L}"
        )

    dky = np.array([results[(l, "grounded")]["kaplan_yorke"] for l in range(L)])
    print(f"\nD_KY across blocks (grounded): mean {dky.mean():.2f}  sd {dky.std(ddof=1):.2f}  cv {dky.std(ddof=1)/dky.mean():.4f}")
    print(f"D_KY range {dky.min():.2f} to {dky.max():.2f} against width 768")
    print("\na low cv would make D_KY the width-invariant constant the thesis asks for;")
    print("the width test itself needs gpt2-medium and gpt2-large and has not been run.")


if __name__ == "__main__":
    main()
