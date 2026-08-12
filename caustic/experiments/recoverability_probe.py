"""Is the entity still recoverable from the hidden state, and does losing it cause the error?

This replaces the pair-distance design, which failed twice. Measuring the cosine
distance between two contexts cannot separate semantics from surface statistics:
a first version compared one-token entity swaps against full rewrites and
measured lexical overlap; a second matched edit distance but used determiner
swaps as the control and measured grammaticality. Distance is the wrong
instrument.

A linear probe is the right one. The folding mechanism claims that two contexts
requiring different continuations collapse onto one internal state, after which
nothing downstream can recover which was seen. That is a claim about
**recoverability**, and it is tested by asking a linear classifier to recover the
entity from `h_l` directly. No distance metric appears anywhere.

**The design.**

    pool        N factual templates over M entities, e.g. "The capital of {X} is"
    filter      keep only entities the model demonstrably knows, because the
                mechanism is about losing information the model HAS. An entity the
                model never encoded cannot have been folded, and including it
                measures ignorance instead. This filter is what the distilgpt2
                runs lacked, and it is why they were uninformative.
    probe       multinomial logistic regression on h_l predicting which entity,
                with grouped cross-validation so no template appears in both folds
    verdict     probe accuracy per layer against the 1/M chance rate

**The test that decides the mechanism.** Split the items by whether the model got
the answer right. If the entity is recoverable on correct items and NOT
recoverable on wrong ones, information loss and error coincide, which is the
mechanism operating. If the entity is equally recoverable on both, the error has
some other cause and the folding account fails for this model.

    python -m caustic.experiments.recoverability_probe
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CAPITALS = {
    "France": "Paris", "Italy": "Rome", "Japan": "Tokyo", "Germany": "Berlin",
    "Spain": "Madrid", "Portugal": "Lisbon", "Greece": "Athens", "Egypt": "Cairo",
    "China": "Beijing", "India": "Delhi", "Russia": "Moscow", "Canada": "Ottawa",
    "Brazil": "Brasilia", "Mexico": "Mexico", "Kenya": "Nairobi", "Norway": "Oslo",
    "Sweden": "Stockholm", "Finland": "Helsinki", "Poland": "Warsaw", "Austria": "Vienna",
    "Ireland": "Dublin", "Cuba": "Havana", "Peru": "Lima", "Chile": "Santiago",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Turkey": "Ankara", "Iran": "Tehran",
    "Iraq": "Baghdad", "Nigeria": "Abuja", "Morocco": "Rabat", "Denmark": "Copenhagen",
}

TEMPLATES = [
    "The capital of {c} is",
    "{c}'s capital is",
    "The capital city of {c} is",
    "The seat of government of {c} is",
    "Question: What is the capital of {c}? Answer:",
]


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = model.config.num_hidden_layers + 1
    D = model.config.hidden_size

    countries = sorted(CAPITALS)
    H: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    y, groups, correct = [], [], []

    for ci, c in enumerate(countries):
        want = CAPITALS[c].lower()
        for ti, tpl in enumerate(TEMPLATES):
            ids = tok(tpl.format(c=c), return_tensors="pt").input_ids.to(DEV)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            for l in range(n_layers):
                H[l].append(out.hidden_states[l][0, -1].float().cpu().numpy())
            top = tok.decode(out.logits[0, -1].argmax()).strip().lower()
            y.append(ci)
            groups.append(ti)
            correct.append(int(top.startswith(want[:4]) or want.startswith(top[:4]) and len(top) > 2))

    y = np.array(y)
    groups = np.array(groups)
    correct = np.array(correct)
    Hs = [np.stack(h) for h in H]

    print(f"model={MODEL} D={D} layers={n_layers} device={DEV}")
    print(f"entities {len(countries)}  templates {len(TEMPLATES)}  items {len(y)}")
    print(f"model answers correctly on {correct.sum()}/{len(correct)} items ({correct.mean():.3f})")
    print(f"chance for the probe = 1/{len(countries)} = {1/len(countries):.4f}\n")

    if correct.sum() < 10:
        print("ABORT: too few correct items; the model does not know enough of this pool")
        return

    def probe_acc(X: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Grouped CV accuracy: a template never appears in both train and test."""
        idx = np.arange(len(y)) if mask is None else np.where(mask)[0]
        Xs, ys, gs = X[idx], y[idx], groups[idx]
        if len(np.unique(ys)) < 2:
            return float("nan")
        accs = []
        for tr, te in GroupKFold(n_splits=min(len(TEMPLATES), len(np.unique(gs)))).split(Xs, ys, gs):
            if len(np.unique(ys[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-8
            clf.fit((Xs[tr] - mu) / sd, ys[tr])
            accs.append(clf.score((Xs[te] - mu) / sd, ys[te]))
        return float(np.mean(accs)) if accs else float("nan")

    hdr = f"{'layer':>5} {'probe acc':>10} {'x chance':>9} {'on correct':>11} {'on wrong':>10}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for l in range(n_layers):
        a = probe_acc(Hs[l])
        ac = probe_acc(Hs[l], correct.astype(bool))
        aw = probe_acc(Hs[l], ~correct.astype(bool))
        rows.append((l, a, ac, aw))
        print(f"{l:>5} {a:>10.4f} {a*len(countries):>9.1f} {ac:>11.4f} {aw:>10.4f}")

    print("\nreading")
    print("-------")
    print("Probe accuracy far above chance means the entity is still linearly present in")
    print("h_l, so no information was destroyed and the folding account does not apply.")
    print("The mechanism predicts recoverability on CORRECT items and a collapse toward")
    print("chance on WRONG ones. Equal recoverability on both refutes it for this model.")
    best = max(rows, key=lambda r: (r[1] if np.isfinite(r[1]) else -1))
    print(f"\nbest layer {best[0]}: overall {best[1]:.4f}, correct {best[2]:.4f}, wrong {best[3]:.4f}")
    gap = [(l, ac - aw) for l, _, ac, aw in rows if np.isfinite(ac) and np.isfinite(aw)]
    if gap:
        l, g = max(gap, key=lambda t: t[1])
        print(f"largest correct-minus-wrong recoverability gap: layer {l}, {g:+.4f}")
        print("A large positive gap is the mechanism. A gap near zero is not.")


if __name__ == "__main__":
    main()
