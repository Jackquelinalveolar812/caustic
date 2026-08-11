"""Does attractor dimension predict how much a block can be pruned?

The falsifiable claim. Block 1 of distilgpt2 has Kaplan-Yorke dimension 29.57 of
768, a 26x compression, while block 5 has 674.67. If D_KY measures how many
directions a block's transport actually uses, then truncating block 1 to low rank
should cost less perplexity than truncating block 5 to the same rank.

If that correlation is absent, D_KY is a description of the dynamics with no
consequence for the weights, and this arm closes.

The intervention is a rank-r truncated SVD of the block's MLP projection
matrices, applied one block at a time with every other block left intact.
Perplexity is measured on held-out text against the unmodified model.

**The control that decides it.** A random rank-r projection of the same matrices,
at the same rank, is measured alongside. Truncated SVD keeping the top r singular
directions will always beat a random projection; that is linear algebra, not a
finding. The claim under test is narrower: that the ORDERING of damage across
blocks tracks D_KY. Spearman correlation between D_KY and perplexity increase is
the number that decides it, and with only six blocks the sample is tiny and the
correlation is reported with that stated rather than buried.

    python -m caustic.experiments.dky_predicts_pruning
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL = "distilgpt2"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RANKS = (8, 32, 128)

# Measured in caustic.experiments.attractor_dimension, grounded condition.
D_KY = {0: 768.00, 1: 29.57, 2: 225.76, 3: 298.08, 4: 482.20, 5: 674.67}

HELDOUT = (
    "The history of computation begins long before electronic machines existed. "
    "Mechanical calculators, tide predictors, and looms that read punched cards all "
    "encoded procedures into physical arrangements of matter. What changed in the "
    "twentieth century was not the idea of automatic calculation but the discovery "
    "that a single machine could imitate any other, provided it was given a "
    "description of that machine as data. Ocean currents move heat around the planet "
    "on timescales that dwarf weather, and the resulting redistribution sets the "
    "climate of entire continents. A language is a system of conventions that lets "
    "one mind reconstruct part of the state of another from a sequence of symbols."
)


def perplexity(model, ids) -> float:
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss).item())


def truncate_(weight: torch.Tensor, r: int, mode: str, gen: torch.Generator) -> None:
    """Replace `weight` in place by a rank-r approximation."""
    W = weight.data.float()
    if mode == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(r, S.numel())
        approx = (U[:, :r] * S[:r]) @ Vh[:r]
    elif mode == "random":
        # Project onto a random r-dimensional row subspace, matched rank.
        n = W.shape[1]
        r = min(r, n)
        G = torch.randn(n, r, generator=gen, dtype=W.dtype)
        Q, _ = torch.linalg.qr(G)
        approx = (W @ Q) @ Q.T
    else:
        raise ValueError(mode)
    weight.data.copy_(approx.to(weight.dtype))


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    ids = tok(HELDOUT, return_tensors="pt").input_ids.to(DEV)

    ppl0 = perplexity(base, ids)
    L = len(base.transformer.h)
    print(f"model={MODEL} blocks={L} heldout tokens={ids.shape[1]} device={DEV} seed={SEED}")
    print(f"baseline perplexity {ppl0:.4f}\n")

    results: dict[tuple[int, int, str], float] = {}
    hdr = f"{'block':>5} {'D_KY':>8} {'rank':>5} {'ppl svd':>10} {'ppl rand':>10} {'d% svd':>9} {'d% rand':>9}"
    print(hdr)
    print("-" * len(hdr))
    for l in range(L):
        for r in RANKS:
            row = []
            for mode in ("svd", "random"):
                m = copy.deepcopy(base)
                gen = torch.Generator().manual_seed(SEED + 1000 * l + r)
                # GPT-2 MLP is c_fc then c_proj; both are Conv1D with .weight
                truncate_(m.transformer.h[l].mlp.c_fc.weight, r, mode, gen)
                truncate_(m.transformer.h[l].mlp.c_proj.weight, r, mode, gen)
                p = perplexity(m, ids)
                results[(l, r, mode)] = p
                row.append(p)
                del m
            d_svd = 100.0 * (row[0] - ppl0) / ppl0
            d_rnd = 100.0 * (row[1] - ppl0) / ppl0
            print(
                f"{l:>5} {D_KY[l]:>8.2f} {r:>5} {row[0]:>10.3f} {row[1]:>10.3f} "
                f"{d_svd:>8.1f}% {d_rnd:>8.1f}%"
            )
        print()

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra**2).sum() * (rb**2).sum())
        return float((ra * rb).sum() / denom) if denom > 0 else 0.0

    dky = np.array([D_KY[l] for l in range(L)])
    print("Spearman correlation between D_KY and perplexity increase under SVD truncation")
    print("(positive = higher attractor dimension means more damage, which is the claim)")
    for r in RANKS:
        dmg = np.array([results[(l, r, "svd")] - ppl0 for l in range(L)])
        print(f"  rank {r:>4}: rho = {spearman(dky, dmg):+.4f}   n = {L} blocks")

    print()
    print(f"n = {L} blocks. With six points a Spearman correlation needs |rho| >= 0.886")
    print("to reach p < 0.05 two-sided, so anything below that is not evidence.")
    print("Block 0 carries D_KY = 768.00 only because the formula saturates there;")
    print("the correlation is therefore also reported excluding it.")
    for r in RANKS:
        dmg = np.array([results[(l, r, "svd")] - ppl0 for l in range(1, L)])
        print(f"  rank {r:>4}, blocks 1-5: rho = {spearman(dky[1:], dmg):+.4f}   n = {L - 1}")


if __name__ == "__main__":
    main()
