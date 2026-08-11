"""The real test: does J-space geometry predict that the model is about to be wrong?

Every earlier experiment used a shuffled-token control, which is an
out-of-distribution control and cannot speak to factual error. This one uses a
real label.

    label(t) = 1 if argmax logits[t] != actual token at t+1

That is the smallest honest unit of confabulation: a position where the model
commits to a continuation the grounded text does not support. It is not the same
as a hallucinated fact, and the report says so, but it is a real label rather
than a proxy for one.

Baselines are computed alongside and are not optional. A geometric score that
does not beat Mahalanobis on both AUROC and cost is a negative result. Costs are
reported per position so the comparison is not only about accuracy: the Jacobian
features cost roughly 53.7 ms each against roughly 0 for a score read off logits
that were computed anyway.

Calibration and test are split by passage, not by position, so that neighbouring
positions from the same passage cannot appear on both sides.

    python -m caustic.experiments.wrongness_auroc
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import (
    MahalanobisScorer,
    PCAScorer,
    auroc_ci,
    logit_margin,
    max_softmax_score,
    predictive_entropy,
)
from caustic.jacobian import exact_jacobian, singular_values
from caustic.spectrum import summarize

MODEL = "distilgpt2"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCKS = (1, 3)
"""Block 1 is where the attractor dimension was lowest (D_KY 29.57 of 768);
block 3 is mid-stack. Both are measured so a null result cannot be blamed on
having probed the wrong depth."""

PASSAGES = [
    "The capital of France is Paris, and the river that runs through it is the Seine. "
    "France shares borders with Spain, Italy, Germany, Belgium and Switzerland.",
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure, and "
    "freezes at zero degrees. Salt lowers the freezing point of water.",
    "The mitochondrion is the organelle that produces most of the chemical energy in a "
    "cell. It has its own DNA, separate from the DNA in the nucleus.",
    "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry "
    "in 1911. She remains the only person to win in two different sciences.",
    "A computer program is a sequence of instructions that a machine can execute. The "
    "same machine can imitate any other machine given a description of it as data.",
    "Ocean currents move heat around the planet on timescales far longer than weather. "
    "The Gulf Stream carries warm water from the Caribbean toward northern Europe.",
    "The printing press was developed in Europe in the fifteenth century. It reduced the "
    "cost of copying a book by orders of magnitude and spread literacy widely.",
    "Photosynthesis converts light energy into chemical energy stored in sugars. It takes "
    "in carbon dioxide and water and releases oxygen as a by-product.",
]


def collect(model, tok, passages, blocks):
    """Per-position features, labels and hidden states for a list of passages."""
    rows: list[dict] = []
    for text in passages:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        T = ids.shape[1]
        if T < 6:
            continue
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        logits = out.logits[0].float().cpu().numpy()
        hs = out.hidden_states

        pred = logits[:-1].argmax(axis=-1)
        actual = ids[0, 1:].cpu().numpy()
        wrong = (pred != actual).astype(int)

        lg = logits[:-1]
        base = {
            "max_softmax": max_softmax_score(lg),
            "entropy": predictive_entropy(lg),
            "logit_margin": logit_margin(lg),
        }

        for t in range(T - 1):
            row = {
                "wrong": int(wrong[t]),
                "h_final": hs[-1][0, t].float().cpu().numpy(),
                **{k: float(v[t]) for k, v in base.items()},
            }
            for l in blocks:
                J = exact_jacobian(model.transformer.h[l], hs[l].detach(), t)
                sv = singular_values(J).cpu().numpy()
                for k, v in summarize(sv, bulk=(10, 400)).items():
                    row[f"L{l}_{k}"] = float(v)
            rows.append(row)
    return rows


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
        .to(DEV)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    n_cal = len(PASSAGES) // 2
    t0 = time.perf_counter()
    cal = collect(model, tok, PASSAGES[:n_cal], BLOCKS)
    test = collect(model, tok, PASSAGES[n_cal:], BLOCKS)
    elapsed = time.perf_counter() - t0

    y = np.array([r["wrong"] for r in test])
    H_cal = np.stack([r["h_final"] for r in cal])
    H_test = np.stack([r["h_final"] for r in test])

    print(f"model={MODEL} blocks={BLOCKS} device={DEV} seed={SEED}")
    print(f"calibration {len(cal)} positions from {n_cal} passages")
    print(f"test        {len(test)} positions from {len(PASSAGES) - n_cal} passages")
    print(f"base rate   {y.mean():.3f} wrong  ({int(y.sum())} of {len(y)})")
    print(f"collection  {elapsed:.1f} s total\n")

    scores: dict[str, np.ndarray] = {
        "max_softmax": np.array([r["max_softmax"] for r in test]),
        "entropy": np.array([r["entropy"] for r in test]),
        "logit_margin": np.array([r["logit_margin"] for r in test]),
        "mahalanobis": MahalanobisScorer(shrinkage=0.1).fit(H_cal).score(H_test),
        "pca_recon_k32": PCAScorer(k=32).fit(H_cal).score(H_test),
    }
    cost = {k: 0.0 for k in scores}

    for l in BLOCKS:
        for key in ("sigma_max", "stable_rank", "tail_alpha", "log_volume"):
            name = f"L{l}_{key}"
            scores[name] = np.array([r[name] for r in test])
            cost[name] = 53.7

    hdr = f"{'score':<22} {'AUROC':>7} {'95% CI':>18} {'ms/pos':>8}  kind"
    print(hdr)
    print("-" * len(hdr))
    ranked = []
    for name, s in scores.items():
        # A geometric quantity has no a priori sign, so the honest reading is the
        # distance from chance; report the better orientation and say which.
        pt, lo, hi = auroc_ci(s, y, n_boot=2000, seed=SEED)
        flipped = pt < 0.5
        if flipped:
            pt, lo, hi = 1.0 - pt, 1.0 - hi, 1.0 - lo
        kind = "baseline" if cost[name] == 0.0 else "J-space"
        ranked.append((pt, name, lo, hi, cost[name], kind, flipped))
    for pt, name, lo, hi, c, kind, flipped in sorted(ranked, reverse=True):
        mark = " (sign flipped)" if flipped else ""
        print(f"{name:<22} {pt:>7.4f} [{lo:>6.4f}, {hi:>6.4f}] {c:>8.1f}  {kind}{mark}")

    best_base = max(r for r in ranked if r[5] == "baseline")
    best_geo = max((r for r in ranked if r[5] == "J-space"), default=None)
    print()
    print(f"best baseline : {best_base[1]} at {best_base[0]:.4f}, {best_base[4]:.1f} ms/pos")
    if best_geo:
        print(f"best J-space  : {best_geo[1]} at {best_geo[0]:.4f}, {best_geo[4]:.1f} ms/pos")
        beats = best_geo[0] > best_base[0] and best_geo[2] > best_base[3]
        print()
        print("VERDICT:", "J-space wins on AUROC with non-overlapping CI" if beats else
              "J-space does NOT beat the best baseline; intervals overlap or it is lower")
        print("A geometric score must beat the baseline on BOTH accuracy and cost to count.")


if __name__ == "__main__":
    main()
