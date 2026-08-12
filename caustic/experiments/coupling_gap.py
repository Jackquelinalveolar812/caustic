"""Does the candidate set separate into components under entity coupling?

Composes the three moves.

    fractal        the context repeats structure across scales, so no single
                   coupling threshold is privileged and none is chosen here
    differential   break the context by taking a derivative: the sensitivity of
                   each candidate's logit to the entity's representation
    topology       ask how the candidate set is CONNECTED under that function,
                   not how large the function is

The topological step is 0-dimensional persistence, which for points on a line is
exactly single-linkage: sort the coupling values, and the largest gap is the
threshold at which the set falls into two components. That threshold is not a
hyperparameter. It is read off the data, which is what the fractal premise
requires — a scale-free statement rather than a tuned one.

The claim under test: the gold token lies in the HIGH-coupling component and the
token the model actually chose lies in the LOW-coupling one. If so, the candidate
set separates into entity-driven and prior-driven answers, and which side a token
falls on is decidable without knowing the answer.

Four relation types are used rather than one. The previous run had 17 wrong items
from capitals alone, and a result that only holds for capitals is a result about
capitals.

    python -m caustic.experiments.coupling_gap
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import auroc_ci

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 8

RELATIONS = {
    "capital": (
        "The capital of {e} is",
        {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo", "Germany": "Berlin",
         "Spain": "Madrid", "Greece": "Athens", "Egypt": "Cairo", "China": "Beijing",
         "Russia": "Moscow", "Canada": "Ottawa", "Norway": "Oslo", "Sweden": "Stockholm",
         "Poland": "Warsaw", "Austria": "Vienna", "Ireland": "Dublin", "Cuba": "Havana",
         "Peru": "Lima", "Chile": "Santiago", "Thailand": "Bangkok", "Turkey": "Ankara"},
    ),
    "language": (
        "The main language spoken in {e} is",
        {"France": "French", "Italy": "Italian", "Japan": "Japanese", "Germany": "German",
         "Spain": "Spanish", "Greece": "Greek", "China": "Chinese", "Russia": "Russian",
         "Brazil": "Portuguese", "Egypt": "Arabic", "Sweden": "Swedish", "Poland": "Polish",
         "Turkey": "Turkish", "Korea": "Korean", "Israel": "Hebrew", "Iran": "Persian"},
    ),
    "currency": (
        "The currency of {e} is the",
        {"Japan": "yen", "China": "yuan", "India": "rupee", "Russia": "ruble",
         "Britain": "pound", "Mexico": "peso", "Sweden": "krona", "Poland": "zloty",
         "Turkey": "lira", "Israel": "shekel", "Korea": "won", "Vietnam": "dong"},
    ),
    "continent": (
        "The country of {e} is located on the continent of",
        {"France": "Europe", "Japan": "Asia", "Egypt": "Africa", "Brazil": "South",
         "Canada": "North", "Australia": "Australia", "India": "Asia", "Kenya": "Africa",
         "Chile": "South", "Norway": "Europe", "Thailand": "Asia", "Morocco": "Africa"},
    ),
}


def largest_gap_split(values: np.ndarray) -> tuple[float, np.ndarray]:
    """0-dimensional persistence on a line: the largest gap splits the set.

    Returns the threshold and a boolean mask marking the high component. The
    threshold is read from the data, never chosen.
    """
    order = np.argsort(values)
    s = values[order]
    gaps = np.diff(s)
    if len(gaps) == 0:
        return float(s[0]), np.ones(len(values), dtype=bool)
    k = int(np.argmax(gaps))
    thresh = 0.5 * (s[k] + s[k + 1])
    return float(thresh), values > thresh


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    per_rel: dict[str, list[dict]] = {}
    for rel, (tpl, facts) in RELATIONS.items():
        rows = []
        # rstrip is required: BPE absorbs a trailing space into the following
        # token, so tokenizing "The capital of " and "The capital of France" gives
        # the same length and the entity span comes out empty.
        prefix_txt = tpl.split("{e}")[0].rstrip()
        for e, ans in facts.items():
            gold = tok(" " + ans, add_special_tokens=False).input_ids[0]
            ids = tok(tpl.format(e=e), return_tensors="pt").input_ids.to(DEV)
            lo = len(tok(prefix_txt, add_special_tokens=False).input_ids)
            hi = len(tok(prefix_txt + " " + e, add_special_tokens=False).input_ids)
            if hi <= lo or hi > ids.shape[1]:
                continue
            emb = model.model.embed_tokens(ids).detach().clone().requires_grad_(True)
            logits = model(inputs_embeds=emb).logits[0, -1]
            top = torch.topk(logits, TOPK)
            cand = [int(i) for i in top.indices]
            if gold not in cand:
                cand.append(gold)
            coup = []
            for tid in cand:
                g = torch.autograd.grad(logits[tid], emb, retain_graph=True)[0]
                coup.append(float(g[0, lo:hi].norm()))
            coup = np.array(coup)
            thresh, high = largest_gap_split(coup)
            gi = cand.index(gold)
            rows.append(
                {
                    "entity": e,
                    "correct": int(cand[0] == gold),
                    "gold_high": int(high[gi]),
                    "chosen_high": int(high[0]),
                    "gold_coup": coup[gi],
                    "chosen_coup": coup[0],
                    "n_high": int(high.sum()),
                    "n_cand": len(cand),
                }
            )
            del emb, logits
        per_rel[rel] = rows

    print(f"model={MODEL} top-k={TOPK} relations={len(RELATIONS)}")
    print("largest-gap split of the candidate set under entity coupling\n")

    hdr = f"{'relation':<11} {'items':>6} {'wrong':>6} {'gold in high':>13} {'chosen in high':>15} {'AUROC':>7}"
    print(hdr)
    print("-" * len(hdr))
    all_g, all_c = [], []
    for rel, rows in per_rel.items():
        wrong = [r for r in rows if not r["correct"]]
        if not wrong:
            print(f"{rel:<11} {len(rows):>6} {0:>6}   (model correct on all)")
            continue
        gh = np.mean([r["gold_high"] for r in wrong])
        ch = np.mean([r["chosen_high"] for r in wrong])
        g = np.array([r["gold_coup"] for r in wrong])
        c = np.array([r["chosen_coup"] for r in wrong])
        all_g.append(g)
        all_c.append(c)
        pt, _, _ = auroc_ci(np.concatenate([g, c]), np.concatenate([np.ones(len(g)), np.zeros(len(c))]), seed=SEED)
        print(f"{rel:<11} {len(rows):>6} {len(wrong):>6} {gh:>13.3f} {ch:>15.3f} {pt:>7.4f}")

    if all_g:
        g, c = np.concatenate(all_g), np.concatenate(all_c)
        pt, lo_, hi_ = auroc_ci(
            np.concatenate([g, c]), np.concatenate([np.ones(len(g)), np.zeros(len(c))]),
            n_boot=4000, seed=SEED,
        )
        print(f"\nPOOLED across relations, n = {len(g)} wrong items")
        print(f"  gold coupling    {g.mean():.4f} +/- {g.std():.4f}")
        print(f"  chosen coupling  {c.mean():.4f} +/- {c.std():.4f}")
        print(f"  AUROC            {pt:.4f} [{lo_:.4f}, {hi_:.4f}]")
        wrong_all = [r for rows in per_rel.values() for r in rows if not r["correct"]]
        gh = np.mean([r["gold_high"] for r in wrong_all])
        ch = np.mean([r["chosen_high"] for r in wrong_all])
        print(f"  gold lands in the high component   {gh:.3f}")
        print(f"  chosen lands in the high component {ch:.3f}")
        print(f"  separation {gh - ch:+.3f}")
        print("\n  the split threshold is the largest gap, never a tuned constant,")
        print("  so this is a scale-free statement about how the candidate set is connected")


if __name__ == "__main__":
    main()
