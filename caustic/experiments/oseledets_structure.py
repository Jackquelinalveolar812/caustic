"""The spine test: does the Oseledets filtration carry information?

The programme's organising claim is that the filtration Oseledets' theorem
produces on a Jacobian cocycle is the same shape of object a persistence module
decomposes into, so barcode tooling applies to a chaos-theoretic object.

That claim has one failure mode and this experiment exists to trigger it. If the
D exponents of a real transformer cocycle are all distinct, the filtration is D
singletons — the sorted spectrum with extra steps — and the encoding carries
nothing. `filtration_entropy` measures exactly the distance from that case:
log(D) means worthless, 0 means one subspace carries everything.

A shuffled-token control is run alongside. If the two conditions produce the same
structure, whatever structure exists belongs to the architecture rather than to
the content, which is the same verdict `tail_alpha` already received.

A single tolerance is not reported, because choosing one after seeing the answer
is a forking path. The whole monotone sweep is printed instead.

    python -m caustic.experiments.oseledets_structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.cocycle import lyapunov_spectrum
from caustic.jacobian import exact_jacobian
from caustic.oseledets import filtration_entropy, growth_filtration, tolerance_sweep

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
    lmid = len(model.transformer.h) // 2
    t0 = T // 4
    conditions = {"grounded": ids, "shuffled": ids[:, rng.permutation(T)].contiguous()}

    spectra: dict[str, np.ndarray] = {}
    for cond, seq in conditions.items():
        with torch.no_grad():
            hs = model(seq, output_hidden_states=True).hidden_states
        h_in = hs[lmid].detach()
        Js = [exact_jacobian(model.transformer.h[lmid], h_in, t).cpu() for t in range(t0, T)]
        spectra[cond] = lyapunov_spectrum(Js)

    D = len(spectra["grounded"])
    print(f"model={MODEL} D={D} block={lmid} cocycle steps={T - t0} device={DEV}")
    print(f"log(D) = {np.log(D):.4f}  <- entropy at this value means the encoding bought nothing\n")

    for cond, lam in spectra.items():
        gaps = np.abs(np.diff(np.sort(lam)[::-1]))
        print(f"{cond}: adjacent-gap median {np.median(gaps):.3e}  min {gaps.min():.3e}  max {gaps.max():.3e}")
    print()

    hdr = f"{'tol':>12} {'cond':>9} {'n_bars':>7} {'max_mult':>9} {'entropy':>9} {'entropy/logD':>13}"
    print(hdr)
    print("-" * len(hdr))
    sweeps = {c: tolerance_sweep(lam) for c, lam in spectra.items()}
    for i in range(len(sweeps["grounded"])):
        for cond in ("grounded", "shuffled"):
            r = sweeps[cond][i]
            print(
                f"{r['tol']:12.3e} {cond:>9} {r['n_bars']:7d} {r['max_multiplicity']:9d} "
                f"{r['entropy']:9.4f} {r['entropy'] / np.log(D):13.4f}"
            )
        print()

    print("verdict")
    print("-------")
    for cond, lam in spectra.items():
        bars = growth_filtration(lam, tol=float(np.median(np.abs(np.diff(np.sort(lam)[::-1])))))
        e = filtration_entropy(bars)
        ratio = e / np.log(D)
        nontrivial = sum(1 for b in bars if b[2] > 1)
        print(
            f"  {cond:9s} at the median adjacent gap: {len(bars)} bars, "
            f"{nontrivial} with multiplicity > 1, entropy/logD = {ratio:.4f}"
        )
    print()
    print("entropy/logD near 1.0 means D singletons: the filtration is the sorted")
    print("spectrum with extra steps and the barcode encoding carries no information.")
    print("A value meaningfully below 1.0 is genuine subspace structure.")


if __name__ == "__main__":
    main()
