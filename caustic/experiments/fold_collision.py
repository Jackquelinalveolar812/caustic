"""The direct test of the mechanism: do distinct contexts fold onto one state?

Every previous experiment measured a spectral summary of the Jacobian, and every
one of them sat at or near chance. That was predictable in hindsight and is worth
stating as the reason this experiment exists.

`sigma_max`, `log_volume` and `tail_alpha` are crease detectors: they respond to
the map becoming singular, `det J -> 0`. But the mechanism under test does not
require a singularity. The maps

    z -> z^2  on the punctured plane,   theta -> (cos theta, sin theta),   exp

are local diffeomorphisms at every single point and are all many-to-one. Local
invertibility everywhere does not imply global injectivity. So a fold that
destroys information can occur with a perfectly well-conditioned Jacobian, and no
spectral summary of that Jacobian can see it. Looking for hallucination at
`det J -> 0` is looking in the wrong place.

The mechanism's actual claim is about NON-INJECTIVITY, which is a statement about
two points, not one. This experiment measures it directly.

**Design.** Two families of context pairs.

    contrast     pairs whose correct continuations DIFFER
                 ("The capital of France is" / "The capital of Italy is")
                 These must NOT collide. If they do, information about which
                 context was seen has been destroyed, and the model cannot be
                 right about both.

    paraphrase   pairs whose correct continuations AGREE
                 ("The capital of France is" / "France's capital city is")
                 These SHOULD collide. Collapsing them is correct compression,
                 not a defect, and they are the control that stops "everything
                 collides" from being read as a finding.

The fold signature is a contrast pair whose hidden states are as close as a
paraphrase pair's. If contrast and paraphrase distances are indistinguishable,
the representation has folded exactly where it must not, and a forced error
follows. If contrast pairs are reliably further apart, the mechanism is not
operating at this scale and the arm closes.

    python -m caustic.experiments.fold_collision
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import auroc_ci

MODEL = "distilgpt2"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CONTRAST = [
    # Exactly one content word swapped, and the required continuation CHANGES.
    ("The capital of France is", "The capital of Italy is"),
    ("The capital of Japan is", "The capital of China is"),
    ("The capital of Spain is", "The capital of Portugal is"),
    ("The capital of Egypt is", "The capital of Kenya is"),
    ("The largest planet is", "The smallest planet is"),
    ("The currency of Japan is the", "The currency of India is the"),
    ("The chemical symbol for gold is", "The chemical symbol for silver is"),
    ("The longest river in Africa is the", "The longest river in Asia is the"),
    ("Mount Everest is in the country of", "Mount Fuji is in the country of"),
    ("The author of Hamlet was", "The author of Oliver Twist was"),
    ("The first president of the United States was", "The second president of the United States was"),
    ("The language spoken in Brazil is", "The language spoken in Mexico is"),
]

PARAPHRASE = [
    # Exactly one function word swapped, and the required continuation is UNCHANGED.
    # Matched to CONTRAST in edit distance so lexical overlap cannot explain a gap.
    ("The capital of France is", "A capital of France is"),
    ("The capital of Japan is", "That capital of Japan is"),
    ("The capital of Spain is", "This capital of Spain is"),
    ("The capital of Egypt is", "Its capital of Egypt is"),
    ("The largest planet is", "One largest planet is"),
    ("The currency of Japan is the", "Some currency of Japan is the"),
    ("The chemical symbol for gold is", "A chemical symbol for gold is"),
    ("The longest river in Africa is the", "That longest river in Africa is the"),
    ("Mount Everest is in the country of", "Mount Everest is in that country of"),
    ("The author of Hamlet was", "An author of Hamlet was"),
    ("The first president of the United States was", "That first president of the United States was"),
    ("The language spoken in Brazil is", "Some language spoken in Brazil is"),
]


def last_states(model, tok, text: str):
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    # Final token only: the state the next prediction is read from.
    return [h[0, -1].float().cpu().numpy() for h in out.hidden_states], out.logits[0, -1].float().cpu().numpy()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def js_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    def soft(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    p, q = soft(p_logits), soft(q_logits)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-30))).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.transformer.h) + 1
    dist = {"contrast": [[] for _ in range(n_layers)], "paraphrase": [[] for _ in range(n_layers)]}
    js = {"contrast": [], "paraphrase": []}

    for family, pairs in (("contrast", CONTRAST), ("paraphrase", PARAPHRASE)):
        for a, b in pairs:
            ha, la = last_states(model, tok, a)
            hb, lb = last_states(model, tok, b)
            for l in range(n_layers):
                dist[family][l].append(cosine_distance(ha[l], hb[l]))
            js[family].append(js_divergence(la, lb))

    print(f"model={MODEL} layers={n_layers} contrast pairs={len(CONTRAST)} paraphrase pairs={len(PARAPHRASE)}")

    # The confound check. An earlier version of this experiment compared one-token
    # entity swaps against full rewrites, so the contrast family was lexically far
    # closer and the whole result was surface form rather than semantics. Both
    # families must now differ by the same number of tokens, and that is verified
    # here rather than asserted.
    def token_edit(a: str, b: str) -> int:
        ta, tb = tok(a).input_ids, tok(b).input_ids
        n, m = len(ta), len(tb)
        d = np.zeros((n + 1, m + 1), dtype=int)
        d[:, 0] = np.arange(n + 1)
        d[0, :] = np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ta[i - 1] != tb[j - 1]))
        return int(d[n, m])

    ec = np.array([token_edit(a, b) for a, b in CONTRAST])
    ep = np.array([token_edit(a, b) for a, b in PARAPHRASE])
    print(f"token edit distance   contrast {ec.mean():.2f} +/- {ec.std():.2f}   "
          f"paraphrase {ep.mean():.2f} +/- {ep.std():.2f}")
    if abs(ec.mean() - ep.mean()) > 0.5:
        print("  WARNING: families are NOT lexically matched; any result below is confounded")
    else:
        print("  families are lexically matched, so surface overlap cannot explain a gap")
    print("\ncosine distance between final-token hidden states of the two members of a pair\n")

    hdr = f"{'layer':>5} {'contrast':>18} {'paraphrase':>18} {'separation':>11} {'AUROC':>7} {'95% CI':>18}"
    print(hdr)
    print("-" * len(hdr))
    for l in range(n_layers):
        c = np.array(dist["contrast"][l])
        p = np.array(dist["paraphrase"][l])
        scores = np.concatenate([c, p])
        labels = np.concatenate([np.ones(len(c)), np.zeros(len(p))])
        pt, lo, hi = auroc_ci(scores, labels, n_boot=2000, seed=SEED)
        print(
            f"{l:>5} {c.mean():>9.4f}+/-{c.std():<7.4f} {p.mean():>9.4f}+/-{p.std():<7.4f} "
            f"{c.mean() - p.mean():>11.4f} {pt:>7.4f} [{lo:>6.4f}, {hi:>6.4f}]"
        )

    jc, jp = np.array(js["contrast"]), np.array(js["paraphrase"])
    pt, lo, hi = auroc_ci(
        np.concatenate([jc, jp]), np.concatenate([np.ones(len(jc)), np.zeros(len(jp))]), seed=SEED
    )
    print(f"\nJensen-Shannon divergence of the next-token distributions")
    print(f"  contrast   {jc.mean():.4f} +/- {jc.std():.4f}")
    print(f"  paraphrase {jp.mean():.4f} +/- {jp.std():.4f}")
    print(f"  AUROC      {pt:.4f} [{lo:.4f}, {hi:.4f}]")

    print("\nreading")
    print("-------")
    print("AUROC near 1.0 at a layer means contrast pairs are reliably further apart than")
    print("paraphrase pairs there, so the representation keeps the distinction it must keep")
    print("and the folding mechanism is NOT operating at that layer.")
    print("AUROC near 0.5 means contrast pairs are no further apart than paraphrases: the")
    print("model has destroyed the information that distinguishes them, which is the fold.")
    print("\nn = 12 pairs per family. This is small and the intervals are wide; the")
    print("experiment can show a large effect and cannot show a small one.")


if __name__ == "__main__":
    main()
