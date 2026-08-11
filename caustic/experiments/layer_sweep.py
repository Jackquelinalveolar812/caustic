"""First J-space curve across layers, with a control condition.

For every block l and a set of token positions t, computes the exact layer-to-layer
Jacobian at that position

    J_l(t) = d h_{l+1}[t] / d h_l[t]   in R^{D x D}

and summarises its singular spectrum. Two conditions are run on matched token
multisets:

  grounded  -- coherent factual English
  shuffled  -- the same tokens in random order

The shuffle is the control. It preserves the token multiset and therefore the
embedding statistics, and destroys only the grounded structure. Any J-space
quantity that does not separate the two conditions is measuring lexical content,
which is the failure mode already established for the point-cloud channel in
examples/why_tokens_fail.py (psi read distinct-type count, R^2 0.537).

Reported per (layer, condition):
  sigma_max      largest singular value; product over layers bounds perturbation growth
  stable_rank    (sum sigma^2) / sigma_max^2 -- effective number of active directions
  tail_alpha     slope of log sigma_i vs log i fitted over the bulk, i.e. the
                 power-law exponent that a fractal account predicts should be
                 stable across layers
  logdet_abs     sum log |sigma_i|, the local volume change; large negative means
                 the block contracts volume, which is the folding signature
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "distilgpt2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

TEXT = (
    "The capital of France is Paris. The Seine flows through the city. "
    "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911. "
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure. "
    "The mitochondrion is the organelle that produces most of the cell's ATP."
)

# Fit the power-law slope over the bulk only. The top few singular values are the
# spike and the last few are numerical floor; neither belongs to the tail.
BULK = (10, 400)


def spectrum_stats(sv: np.ndarray) -> dict:
    sv = np.sort(sv)[::-1]
    i = np.arange(BULK[0], BULK[1])
    slope = np.polyfit(np.log(i + 1.0), np.log(np.maximum(sv[i], 1e-30)), 1)[0]
    return {
        "sigma_max": float(sv[0]),
        "stable_rank": float((sv**2).sum() / sv[0] ** 2),
        "tail_alpha": float(-slope),
        "logdet_abs": float(np.log(np.maximum(sv, 1e-30)).sum()),
    }


def run(ids, model, positions) -> list[dict]:
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    rows = []
    for l, block in enumerate(model.transformer.h):
        h_in = hs[l].detach()

        def f(h_row, _t, _h_in=h_in, _block=block):
            h = _h_in.clone()
            h[0, _t, :] = h_row
            out = _block(h)
            out = out[0] if isinstance(out, tuple) else out
            return out[0, _t, :]

        for t in positions:
            J = torch.func.jacrev(lambda r, t=t: f(r, t))(h_in[0, t, :].clone())
            sv = torch.linalg.svdvals(J.float()).detach().cpu().numpy()
            rows.append({"layer": l, "pos": int(t), **spectrum_stats(sv)})
    return rows


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
    perm = rng.permutation(T)
    ids_shuf = ids[:, perm].contiguous()

    positions = list(range(T // 4, T, max(1, T // 12)))[:10]
    print(f"model={MODEL} D=768 T={T} positions={len(positions)} device={DEV}")
    print("control = same token multiset, shuffled order\n")

    out = {"grounded": run(ids, model, positions), "shuffled": run(ids_shuf, model, positions)}

    hdr = f"{'layer':>5} {'cond':>9} {'sigma_max':>10} {'stable_rk':>10} {'tail_alpha':>11} {'logdet':>10}"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for l in range(len(model.transformer.h)):
        for cond in ("grounded", "shuffled"):
            r = [x for x in out[cond] if x["layer"] == l]
            m = {k: float(np.mean([x[k] for x in r])) for k in ("sigma_max", "stable_rank", "tail_alpha", "logdet_abs")}
            summary[(l, cond)] = m
            print(
                f"{l:>5} {cond:>9} {m['sigma_max']:>10.4f} {m['stable_rank']:>10.2f} "
                f"{m['tail_alpha']:>11.4f} {m['logdet_abs']:>10.1f}"
            )
        print()

    print("separation (grounded - shuffled), as a fraction of the grounded value:")
    for k in ("sigma_max", "stable_rank", "tail_alpha", "logdet_abs"):
        d = [
            (summary[(l, "grounded")][k] - summary[(l, "shuffled")][k]) / abs(summary[(l, "grounded")][k])
            for l in range(len(model.transformer.h))
        ]
        print(f"  {k:<12} per-layer {np.array2string(np.array(d), precision=3)}  |mean| {abs(np.mean(d)):.4f}")

    a = np.array([summary[(l, "grounded")]["tail_alpha"] for l in range(len(model.transformer.h))])
    print(f"\ntail_alpha across layers (grounded): mean {a.mean():.4f}  sd {a.std():.4f}  cv {a.std()/abs(a.mean()):.4f}")
    print("a small cv would be the layer-invariant constant the thesis asks for")

    np.save("jspace_sweep.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
