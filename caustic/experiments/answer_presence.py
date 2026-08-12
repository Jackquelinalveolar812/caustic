"""Is the correct answer present in the model when it answers wrongly?

This discriminates between two mechanisms in the framework that predict opposite
things about the same measurement.

    topological disconnection   the hallucination region is disconnected from the
                                reality anchor, so the correct answer is ABSENT
                                from the activation. A probe cannot find it and
                                the answer's logit rank is far from the top.

    Nash equilibrium            the answer is PRESENT but loses an internal
                                competition between components, none of which has
                                an incentive to change. A probe finds it and the
                                rank is near the top, beaten by a competitor.

Only one can be true of a given error. The previous experiment established that
the *entity* survives at 0.9624 recoverability on wrong items; this asks the
harder question about the *answer*, which is the one the framework's Phase 2
geodesic correction depends on. A correction that moves an activation toward a
truth region presupposes the truth is representable there. If the answer is
absent, there is nothing to steer toward and Phase 2 cannot work as described.

Three measurements, all on items the model gets WRONG:

    rank        where the correct answer sits in the output distribution
    logit lens  the rank of the correct answer when each layer's hidden state is
                pushed through the unembedding, which shows whether the answer was
                ever in play and then lost
    probe       whether a linear classifier recovers the correct answer token from
                h_l, which is presence independent of the output head

    python -m caustic.experiments.answer_presence
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

from caustic.experiments.recoverability_probe import CAPITALS, TEMPLATES

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = model.config.num_hidden_layers + 1
    norm = model.model.norm
    head = model.lm_head

    countries = sorted(CAPITALS)
    rows = []
    H: list[list[np.ndarray]] = [[] for _ in range(n_layers)]

    for c in countries:
        gold_id = tok(" " + CAPITALS[c], add_special_tokens=False).input_ids[0]
        for ti, tpl in enumerate(TEMPLATES):
            ids = tok(tpl.format(c=c), return_tensors="pt").input_ids.to(DEV)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            logits = out.logits[0, -1]
            top1 = int(logits.argmax())
            # Rank of the gold token: how many tokens outscore it.
            rank = int((logits > logits[gold_id]).sum()) + 1

            # Logit lens: push each layer's state through the final norm and head.
            #
            # transformers applies the final norm BEFORE storing the last entry of
            # hidden_states, so re-applying it there double-norms the state. That
            # bug produced a spurious "the last two layers degrade the answer"
            # result: the final layer's rank was computed from a state RMSNorm had
            # been applied to twice. Verified on this model: lm_head(h) matches the
            # returned logits to 1.05e-05 while lm_head(norm(h)) is off by 5.82.
            lens_rank = []
            for l in range(n_layers):
                h = out.hidden_states[l][0, -1]
                with torch.no_grad():
                    lg = head(h.unsqueeze(0))[0] if l == n_layers - 1 else head(norm(h.unsqueeze(0)))[0]
                lens_rank.append(int((lg > lg[gold_id]).sum()) + 1)
                H[l].append(h.float().cpu().numpy())

            rows.append(
                {
                    "country": c,
                    "template": ti,
                    "gold_id": gold_id,
                    "correct": int(top1 == gold_id),
                    "rank": rank,
                    "lens": lens_rank,
                }
            )

    correct = np.array([r["correct"] for r in rows], dtype=bool)
    rank = np.array([r["rank"] for r in rows])
    gold = np.array([r["gold_id"] for r in rows])
    groups = np.array([r["template"] for r in rows])
    Hs = [np.stack(h) for h in H]
    V = model.config.vocab_size

    print(f"model={MODEL} layers={n_layers} vocab={V} items={len(rows)}")
    print(f"top-1 correct on {correct.sum()}/{len(rows)} items\n")

    w = rank[~correct]
    print("WHERE THE CORRECT ANSWER SITS, on items answered wrongly")
    print(f"  n = {len(w)}")
    for k in (2, 3, 5, 10, 50, 100, 1000):
        print(f"  rank <= {k:>5}: {int((w <= k).sum()):>3}/{len(w)}  ({(w <= k).mean():.3f})")
    print(f"  median rank {np.median(w):.0f} of {V}   mean log10 rank {np.log10(w).mean():.2f}")
    print(f"  a disconnected region predicts ranks in the thousands;")
    print(f"  a lost internal competition predicts ranks in the single digits\n")

    lens = np.array([r["lens"] for r in rows])
    print("LOGIT-LENS rank of the correct answer by layer (median)")
    print(f"{'layer':>5} {'all':>10} {'correct':>10} {'wrong':>10}")
    print("-" * 38)
    for l in range(0, n_layers, max(1, n_layers // 12)):
        print(
            f"{l:>5} {np.median(lens[:, l]):>10.0f} {np.median(lens[correct, l]):>10.0f} "
            f"{np.median(lens[~correct, l]):>10.0f}"
        )
    print(f"{n_layers-1:>5} {np.median(lens[:, -1]):>10.0f} {np.median(lens[correct, -1]):>10.0f} "
          f"{np.median(lens[~correct, -1]):>10.0f}")

    print("\nLINEAR PROBE for the correct answer token, wrong items only")
    ys, gs = gold[~correct], groups[~correct]
    keep = np.isin(ys, [v for v in np.unique(ys) if (ys == v).sum() >= 2])
    if keep.sum() < 10 or len(np.unique(ys[keep])) < 2:
        print("  too few repeated answer tokens among wrong items to probe")
    else:
        print(f"  {keep.sum()} items, {len(np.unique(ys[keep]))} distinct answers, "
              f"chance {1/len(np.unique(ys[keep])):.4f}")
        for l in (n_layers // 2, n_layers - 3, n_layers - 1):
            X = Hs[l][~correct][keep]
            accs = []
            for tr, te in GroupKFold(n_splits=min(5, len(np.unique(gs[keep])))).split(X, ys[keep], gs[keep]):
                if len(np.unique(ys[keep][tr])) < 2:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
                clf = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, ys[keep][tr])
                accs.append(clf.score((X[te] - mu) / sd, ys[keep][te]))
            if accs:
                print(f"  layer {l:>3}: probe accuracy {np.mean(accs):.4f}")

    print("\nVERDICT")
    print("-------")
    frac_top10 = (w <= 10).mean()
    if frac_top10 >= 0.5:
        print(f"{frac_top10:.1%} of wrong answers have the correct token in the top 10.")
        print("The answer is PRESENT and loses a competition. This supports the Nash")
        print("framing and refutes topological disconnection for these errors.")
    else:
        print(f"only {frac_top10:.1%} of wrong answers have the correct token in the top 10;")
        print(f"median rank {np.median(w):.0f} of {V}. The answer is largely ABSENT, which")
        print("supports disconnection and means a geodesic correction has no target.")


if __name__ == "__main__":
    main()
