"""Finite-time Lyapunov exponents of a live language model, against a control.

Two cocycles are measured, and they are different objects:

  block cocycle   J_5 ... J_0 at one fixed token   -- depth-wise transport
  token cocycle   J_l(t) for t = t0 .. T-1         -- position-wise transport

The block cocycle has exactly `n_layers` steps. On distilgpt2 that is 6, which is
nowhere near the asymptotic regime Oseledets' theorem describes, so its output is
reported as a *finite-time* exponent and the convergence trace is printed
alongside it. Calling a 6-step average a Lyapunov exponent would be the single
easiest overclaim available here.

The control is the same token multiset in shuffled order. It preserves the
embedding statistics and destroys only the grounded structure, so any quantity
that fails to separate the two conditions is measuring lexical content. This is
the control that already killed the point-cloud channel in
`sigmoid/examples/why_tokens_fail.py`, where psi read distinct-type count at
R^2 0.537 and predicted nothing forward.

    python experiments/lyapunov_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.cocycle import finite_time_spectrum, lyapunov_spectrum
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


def report(name: str, cond: str, lam: np.ndarray, trace: np.ndarray | None = None) -> dict:
    n_pos = int((lam > 0).sum())
    out = {
        "lam1": float(lam[0]),
        "lam_min": float(lam[-1]),
        "sum": float(lam.sum()),
        "n_positive": n_pos,
        "D": len(lam),
    }
    print(
        f"  {cond:9s} lam1 {out['lam1']:+8.4f}  lam_min {out['lam_min']:+10.4f}  "
        f"sum {out['sum']:+10.2f}  positive {n_pos:>4}/{len(lam)}"
    )
    if trace is not None and len(trace) > 1:
        # Has the leading exponent settled, or was it still moving at the end?
        drift = abs(trace[-1, 0] - trace[-2, 0])
        print(f"  {'':9s} lam1 trace {np.array2string(trace[:, 0], precision=3)}")
        print(f"  {'':9s} last-step drift in lam1: {drift:.4f}  <- large means not converged")
    return out


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
    L = len(model.transformer.h)
    conditions = {"grounded": ids, "shuffled": ids[:, rng.permutation(T)].contiguous()}

    print(f"model={MODEL} D=768 T={T} blocks={L} device={DEV} seed={SEED}")
    print("control = same token multiset, shuffled order\n")

    results: dict[tuple[str, str], dict] = {}

    print(f"BLOCK cocycle at the last token (n={L} steps -- FINITE-TIME, not asymptotic)")
    for cond, seq in conditions.items():
        with torch.no_grad():
            hs = model(seq, output_hidden_states=True).hidden_states
        Js = [exact_jacobian(b, hs[l].detach(), seq.shape[1] - 1).cpu() for l, b in enumerate(model.transformer.h)]
        results[("block", cond)] = report("block", cond, lyapunov_spectrum(Js), finite_time_spectrum(Js))

    lmid = L // 2
    t0 = T // 4
    print(f"\nTOKEN cocycle at block {lmid} (n={T - t0} steps)")
    for cond, seq in conditions.items():
        with torch.no_grad():
            hs = model(seq, output_hidden_states=True).hidden_states
        h_in = hs[lmid].detach()
        Jt = [exact_jacobian(model.transformer.h[lmid], h_in, t).cpu() for t in range(t0, T)]
        results[("token", cond)] = report("token", cond, lyapunov_spectrum(Jt), finite_time_spectrum(Jt))

    print("\nseparation (grounded - shuffled) as a fraction of the grounded value:")
    for kind in ("block", "token"):
        g, s = results[(kind, "grounded")], results[(kind, "shuffled")]
        parts = []
        for k in ("lam1", "sum", "n_positive"):
            denom = abs(g[k]) if g[k] != 0 else 1.0
            parts.append(f"{k} {(g[k] - s[k]) / denom:+.4f}")
        print(f"  {kind:6s} " + "   ".join(parts))

    print("\nsum of exponents is the mean log volume change per step.")
    print("negative means the map contracts volume, which is necessary for folding")
    print("but not sufficient: a map can contract volume and stay injective.")


if __name__ == "__main__":
    main()
