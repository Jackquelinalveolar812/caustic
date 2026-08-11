# Caustic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish whether the Jacobian of a live language model carries a causal, measurable invariant that a point cloud of hidden states cannot, and catalogue every hot path where the same linearization move pays.

**Architecture:** Three fields are treated as three linearization functors on different structures — differential geometry linearizes locally (tangent space, Jacobian), algebraic topology linearizes globally (homology is a functor to vector spaces; a persistence module decomposes into intervals), and fractal geometry linearizes across scale (a power law is a straight line in log-log). Chaos theory is what happens when the local functor is composed along a trajectory: Oseledets' Multiplicative Ergodic Theorem states that products of Jacobians induce a **filtration** of the state space by exponential growth rate. A filtration indexed by a real parameter and decomposing into interval pieces is the same object persistent homology consumes. That coincidence is the bridge between the existing topology engine and Jacobian space, and it is the spine of this plan.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, transformers 5.3.0, numpy, ripser/gudhi, pytest, hypothesis. Single RTX 4060 Laptop, 8 GiB VRAM.

---

## The one idea

> A hot path pays quadratic or redundant cost whenever it ignores an order, a connectivity, or an interval structure already present in its own data. Applying a linearization functor replaces the scan with the structure. This is not a hypothesis: it is the identical move behind all seven of this author's major-lab merged pull requests. The programme systematizes it, and the Jacobian of a live LLM is the primary test case because it is the one object where all four fields meet at once.

| PR | latent structure exploited |
|---|---|
| `google-deepmind/mujoco#3396` | connected components (H₀) replacing dense n×n scratch |
| `google-deepmind/mujoco#3450` | convex hull graph structure replacing a quadratic scan |
| `google/XNNPACK#10801` | interval/gap packing in the memory planner |
| `google/highway#3244` | slice structure pruning collision and scan tests |
| `tensorflow/tensorflow#124410` | transitive reduction of collective control edges (partial order) |
| `triton-lang/kernels#22` | 0D persistence over key-block centroids → causal CSR schedule |
| `NVIDIA/NeMo-Relay#481` | reuse of stable ACG scaffolds |

Game theory enters at the same seam. Existence of Nash equilibrium is proved by Kakutani's fixed-point theorem, which is topology; Sperner's lemma gives Brouwer combinatorially. The engine's existing Banach certificate (`sigmoid/operator.py`, `ρ = σ_max(A) < 1`) is the metric analogue: it buys uniqueness and a rate where Kakutani buys only existence. A scheduler choosing a sparsity pattern against a workload is playing a game whose strategy space is set by the topology, so controlling the topology is what makes the equilibrium controllable. This is the sense in which "topologically controlled environments provide causal control", and it is a design constraint on Phase 2, not a separate research arm.

## Global Constraints

- **A claim needs a control.** Every measured number states what it was compared against. Boring baselines are mandatory: carry-forward, predict-the-mean, majority class, persistence, Mahalanobis, PCA. Four previously shipped claims in `sigmoid` were wrong, all for a missing control, never for failing mathematics.
- **Same-budget ablation.** A topological arm and its baseline get equal dimensions, equal parameters, equal wall-clock. Dimension-matched or it does not count.
- **Negative results ship.** Any arm that fails is written up with what it measured, why it failed, and where the byproduct is useful. This is a completion condition, not a consolation.
- **Post-hoc decisions are flagged.** Block 0 was excluded from the `tail_alpha` fit after inspecting the data. Any such choice must be re-tested on held-out text before it is reported as confirmed.
- **`attn_implementation="eager"` is required** for any model differentiated with forward-mode AD. `torch.func.jvp` raises `NotImplementedError` on `_scaled_dot_product_efficient_attention`. Reverse mode is unaffected.
- **Determinism.** `torch.use_deterministic_algorithms(True)` for anything feeding an A/B comparison. Non-determinism in the measurement invalidates the measurement.
- **Off-limits upstream targets** for new contribution ideas: pytorch/pytorch, mujoco, mujoco_warp, triton, XNNPACK, vllm, rtp-llm, NVIDIA topograph/NeMo-Relay/TensorRT-LLM/cosmos, penzai, Graphormer, pytorch_geometric, cugraph, tensorflow, highway, zstd, fairchem, alphafold3, openxla/xla. Verify against the live PR list before proposing.

## What is already established

Committed at `f29eed0`. Do not redo.

- `caustic/jacobian.py` — exact and Krylov Jacobians of a transformer block. 22 tests pass against closed-form ground truth.
- `caustic/spectrum.py` — `sigma_max`, `stable_rank`, `tail_alpha`, `log_volume`.
- Cost, measured on RTX 4060 Laptop, distilgpt2, layer 3, float32: block forward `0.588 ms`; full 768×768 `jacrev` `53.694 ms` (91.3× forward); top-8 power iteration, 20 iters `1428.938 ms` (2429.5× forward). **The exact Jacobian is 27× cheaper than the Krylov estimator of its own top singular values at D=768.** Power iteration agrees with exact to `1.685e-04` max relative error, so the estimator is correct and simply loses at this D.
- First sweep, 6 blocks × 10 positions, one 61-token passage against its shuffled-token control: `sigma_max`, `tail_alpha` and `log_volume` **all fail** to separate grounded from shuffled — signs flip layer to layer, |mean| separation 0.0085, 0.0217, 0.0548 respectively. Only `stable_rank` holds sign, 5/6 layers positive, median +17.5%, block 4 reversing.
- `tail_alpha` over blocks 1–5: grounded `0.2374 ± 0.0293` (cv 0.124), shuffled `0.2342 ± 0.0481` (cv 0.205), differing by 1.4%. Block 0 is an outlier at `0.6168`. **The exponent is a property of the architecture, not the content** — which kills it as a detector and makes it a candidate for a pruning budget.

## What is already ruled out

- Persistence over a **point cloud of hidden states** does not predict forward. Measured in `sigmoid/examples/why_tokens_fail.py`: ψ reads lexical repetition (R² 0.537 for distinct-type count against 0.087 for the linear channel) and is "an observable of the past with no forward influence". Do not re-run this. It is the reason the programme works in Jacobian space, where the object is by construction the operator that transports perturbations forward.
- The Jacobian **Conjecture** does not transfer. It concerns polynomial maps ℂⁿ→ℂⁿ with constant nonzero determinant; a transformer is not polynomial and its det J is not constant. Only the moral of Alpöge's July 2026 counterexample transfers: **local invertibility everywhere does not imply global injectivity.** No task may cite the theorem as support for a claim about neural networks.

## File Structure

| file | responsibility |
|---|---|
| `caustic/jacobian.py` | *(exists)* exact and Krylov block Jacobians |
| `caustic/spectrum.py` | *(exists)* scalar summaries of one spectrum |
| `caustic/cocycle.py` | products of Jacobians along a trajectory; Lyapunov spectrum by QR iteration |
| `caustic/oseledets.py` | growth-rate filtration → interval decomposition (barcode) |
| `caustic/detect.py` | scoring an LLM generation from J-space quantities, plus the mandatory baselines |
| `caustic/experiments/probe_cost.py` | *(exists)* cost measurement |
| `caustic/experiments/layer_sweep.py` | *(exists)* layer sweep with shuffled control |
| `caustic/experiments/lyapunov_llm.py` | Lyapunov spectrum of a live model over a real trajectory |
| `caustic/experiments/width_invariance.py` | `tail_alpha` across gpt2 / gpt2-medium / gpt2-large |
| `caustic/experiments/hallucination_auroc.py` | the real test, against real factual-error labels |
| `caustic/experiments/prune_budget.py` | does the exponent buy wall-clock |
| `tests/test_jacobian.py` | *(exists)* 22 tests, ground truth |
| `tests/test_cocycle.py` | Lyapunov ground truth on systems with known exponents |
| `tests/test_oseledets.py` | filtration and barcode invariants |
| `CANDIDATES.md` | the move catalogue — one row per candidate, with its control |
| `EVIDENCE.md` | every measured number, its control, and every negative |

---

## Phase 1 — Idea 1: the Jacobian arm

### Task 1: Lyapunov spectrum of a Jacobian cocycle

**Files:**
- Create: `caustic/cocycle.py`
- Test: `tests/test_cocycle.py`

**Interfaces:**
- Consumes: `caustic.jacobian.exact_jacobian`
- Produces: `lyapunov_spectrum(jacobians: list[Tensor], dt: float = 1.0) -> np.ndarray`, returning exponents in descending order, length D.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cocycle.py
import numpy as np
import pytest
import torch
from caustic.cocycle import lyapunov_spectrum


def test_diagonal_cocycle_recovers_log_of_diagonal():
    """For J_k = diag(a), every product is diag(a^n), so lambda_i = log a_i exactly."""
    a = np.array([4.0, 2.0, 0.5, 0.1])
    Js = [torch.diag(torch.tensor(a)) for _ in range(200)]
    got = lyapunov_spectrum(Js)
    assert np.allclose(got, np.log(a), atol=1e-8), f"got {got}, want {np.log(a)}"


def test_orthogonal_cocycle_has_zero_exponents():
    """Rotations preserve length, so every exponent is 0. The negative control."""
    g = torch.Generator().manual_seed(0)
    Js = [torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))[0] for _ in range(300)]
    assert np.allclose(lyapunov_spectrum(Js), 0.0, atol=1e-8)


def test_exponents_are_descending():
    g = torch.Generator().manual_seed(1)
    Js = [torch.randn(5, 5, generator=g, dtype=torch.float64) for _ in range(200)]
    lam = lyapunov_spectrum(Js)
    assert np.all(np.diff(lam) <= 1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cocycle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caustic.cocycle'`

- [ ] **Step 3: Write minimal implementation**

Benettin's algorithm. Naively multiplying Jacobians overflows and collapses to the top exponent; re-orthonormalizing every step and accumulating log diagonals of R is the standard fix.

```python
# caustic/cocycle.py
"""Lyapunov exponents of a Jacobian cocycle by QR iteration.

Direct products of Jacobians are useless numerically: every column collapses onto
the leading singular direction and the norm overflows or underflows within a few
dozen steps. Benettin's algorithm re-orthonormalizes after each step and
accumulates log|R_ii|, which keeps all D exponents separated.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["lyapunov_spectrum"]


def lyapunov_spectrum(jacobians, dt: float = 1.0) -> np.ndarray:
    """Lyapunov exponents of the cocycle, descending.

    Args:
        jacobians: sequence of (D, D) tensors, applied in order.
        dt: time per step. Exponents are per unit time.
    """
    if len(jacobians) == 0:
        raise ValueError("need at least one Jacobian")
    D = jacobians[0].shape[0]
    Q = torch.eye(D, dtype=torch.float64)
    acc = np.zeros(D)
    for J in jacobians:
        Z = J.double() @ Q
        Q, R = torch.linalg.qr(Z)
        d = torch.diagonal(R)
        # QR is unique only up to sign; fold the sign into Q so R has a positive
        # diagonal, otherwise log|R_ii| is right but Q drifts between conventions.
        s = torch.sign(d)
        s[s == 0] = 1.0
        Q, d = Q * s, d * s
        acc += np.log(d.abs().clamp_min(1e-300).cpu().numpy())
    return acc / (len(jacobians) * dt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cocycle.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add caustic/cocycle.py tests/test_cocycle.py
git commit -m "Lyapunov spectrum of a Jacobian cocycle by QR iteration"
```

---

### Task 2: Lyapunov spectrum of a live model

**Files:**
- Create: `caustic/experiments/lyapunov_llm.py`

**Interfaces:**
- Consumes: `caustic.cocycle.lyapunov_spectrum`, `caustic.jacobian.exact_jacobian`
- Produces: a table of per-condition exponents written to `EVIDENCE.md`.

The trajectory is over **layers** at a fixed token, then over **tokens** at a fixed layer. These are different cocycles and may behave differently; run both. With 6 blocks the layer cocycle has only 6 steps, which is too short for convergence — report it as a finite-time exponent and say so, do not call it a Lyapunov exponent.

- [ ] **Step 1: Write the experiment**

```python
# caustic/experiments/lyapunov_llm.py
"""Finite-time Lyapunov exponents of distilgpt2, grounded text vs shuffled control.

Two cocycles: across blocks at fixed token, and across tokens at fixed block. The
block cocycle is only 6 steps long, so its exponents are finite-time and are
labelled as such. Only the token cocycle is long enough to approach a limit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caustic.cocycle import lyapunov_spectrum
from caustic.jacobian import exact_jacobian

MODEL, SEED = "distilgpt2", 0
TEXT = (
    "The capital of France is Paris. The Seine flows through the city. "
    "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911. "
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure."
)


def main() -> None:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
    T = ids.shape[1]
    conditions = {"grounded": ids, "shuffled": ids[:, rng.permutation(T)].contiguous()}

    for name, seq in conditions.items():
        with torch.no_grad():
            hs = model(seq, output_hidden_states=True).hidden_states
        # block cocycle at the last token
        Js = [exact_jacobian(b, hs[l].detach(), seq.shape[1] - 1).cpu()
              for l, b in enumerate(model.transformer.h)]
        lam_block = lyapunov_spectrum(Js)
        # token cocycle at a fixed middle block
        l = len(model.transformer.h) // 2
        Jt = [exact_jacobian(model.transformer.h[l], hs[l].detach(), t).cpu()
              for t in range(T // 4, T)]
        lam_tok = lyapunov_spectrum(Jt)
        print(f"{name:9s} block cocycle (finite-time, n={len(Js)}): "
              f"lam1 {lam_block[0]:+.4f}  lam_min {lam_block[-1]:+.4f}  "
              f"n_positive {int((lam_block > 0).sum())}/{len(lam_block)}")
        print(f"{name:9s} token cocycle (n={len(Jt)}): "
              f"lam1 {lam_tok[0]:+.4f}  lam_min {lam_tok[-1]:+.4f}  "
              f"n_positive {int((lam_tok > 0).sum())}/{len(lam_tok)}")
        print(f"{name:9s} sum lam (token) {lam_tok.sum():+.2f}   "
              f"<- negative means volume contracts, the folding precondition")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python -m caustic.experiments.lyapunov_llm`
Expected: two blocks of output per condition, no NaN, exponents descending.

- [ ] **Step 3: Record the numbers in `EVIDENCE.md` with the control named**

Write the grounded and shuffled numbers side by side. State the separation as a fraction of the grounded value, as `caustic/experiments/layer_sweep.py` already does. If the separation has inconsistent sign across conditions, write that it failed.

- [ ] **Step 4: Commit**

```bash
git add caustic/experiments/lyapunov_llm.py EVIDENCE.md
git commit -m "Finite-time Lyapunov exponents of distilgpt2 against a shuffled control"
```

---

### Task 3: The Oseledets filtration as a barcode

**Files:**
- Create: `caustic/oseledets.py`
- Test: `tests/test_oseledets.py`

**Interfaces:**
- Consumes: `caustic.cocycle.lyapunov_spectrum`
- Produces: `growth_filtration(lam: np.ndarray, tol: float = 1e-3) -> list[tuple[float, float, int]]` — a list of `(birth, death, multiplicity)` intervals, where birth and death are growth rates and multiplicity is the dimension of the corresponding Oseledets subspace.

This is the task that tests the spine claim. Oseledets gives a filtration `V_1 ⊂ V_2 ⊂ ... ⊂ V_k = R^D` where every vector in `V_i \ V_{i-1}` grows at rate `λ_i`. Encoding it as `(λ_i, λ_{i+1}, dim)` intervals makes it the same shape as a persistence barcode, so every downstream tool already written for barcodes applies unchanged. **If the exponents are all distinct the filtration is trivial (D intervals of multiplicity 1) and the bridge buys nothing — that is a real possible outcome and must be reported, not hidden.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oseledets.py
import numpy as np
import pytest
from caustic.oseledets import growth_filtration


def test_degenerate_spectrum_gives_one_interval_of_full_multiplicity():
    """All exponents equal: one Oseledets subspace, the whole space."""
    bars = growth_filtration(np.full(8, 0.5))
    assert len(bars) == 1
    assert bars[0][2] == 8


def test_distinct_spectrum_gives_unit_multiplicities():
    """The trivial case the bridge must be honest about."""
    bars = growth_filtration(np.array([3.0, 2.0, 1.0]))
    assert [b[2] for b in bars] == [1, 1, 1]


def test_clustered_spectrum_groups_within_tolerance():
    lam = np.array([2.0, 2.0 + 1e-6, 2.0 - 1e-6, -1.0, -1.0 + 1e-7])
    bars = growth_filtration(lam, tol=1e-3)
    assert sorted(b[2] for b in bars) == [2, 3]


def test_multiplicities_sum_to_dimension():
    rng = np.random.default_rng(0)
    lam = np.sort(rng.normal(size=40))[::-1]
    assert sum(b[2] for b in growth_filtration(lam)) == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oseledets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'caustic.oseledets'`

- [ ] **Step 3: Write minimal implementation**

```python
# caustic/oseledets.py
"""The Oseledets growth-rate filtration, encoded as intervals.

The Multiplicative Ergodic Theorem gives a filtration V_1 subset ... subset V_k
of the state space in which every vector of V_i \\ V_{i-1} grows at rate lambda_i.
Encoding it as (birth, death, multiplicity) triples puts it in the same shape as a
persistence barcode, so barcode tooling applies without modification.

The honest caveat: when all D exponents are distinct the filtration has D steps of
multiplicity one and carries exactly the information the sorted spectrum already
carried. The bridge is only informative when exponents cluster.
"""

from __future__ import annotations

import numpy as np

__all__ = ["growth_filtration"]


def growth_filtration(lam: np.ndarray, tol: float = 1e-3) -> list[tuple[float, float, int]]:
    """Group a Lyapunov spectrum into Oseledets subspaces.

    Args:
        lam: exponents, any order.
        tol: absolute gap below which two exponents are treated as equal.

    Returns:
        (birth, death, multiplicity) per distinct exponent, descending by birth.
        `death` is the next distinct exponent below, or -inf for the last.
    """
    lam = np.sort(np.asarray(lam, dtype=np.float64))[::-1]
    groups: list[list[float]] = [[lam[0]]]
    for x in lam[1:]:
        if abs(groups[-1][-1] - x) <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    bars = []
    for i, g in enumerate(groups):
        birth = float(np.mean(g))
        death = float(np.mean(groups[i + 1])) if i + 1 < len(groups) else -np.inf
        bars.append((birth, death, len(g)))
    return bars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oseledets.py -v`
Expected: 4 passed

- [ ] **Step 5: Run it on the real spectra from Task 2 and record whether multiplicities exceed 1**

If every multiplicity is 1 on real data, write in `EVIDENCE.md`: "the Oseledets filtration is trivial on distilgpt2; the barcode encoding carries no information beyond the sorted spectrum." That is the falsification of the spine's second half, and it must be stated plainly.

- [ ] **Step 6: Commit**

```bash
git add caustic/oseledets.py tests/test_oseledets.py EVIDENCE.md
git commit -m "Oseledets growth filtration encoded as intervals, with the degeneracy case reported"
```

---

### Task 4: The real test — factual-error labels, against real baselines

**Files:**
- Create: `caustic/detect.py`, `caustic/experiments/hallucination_auroc.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `score(model, tok, prompt) -> dict[str, float]` returning every J-space score and every baseline score for one generation.

This is the task the whole arm exists for. A shuffled-token control is an **OOD** control, not a hallucination control; it cannot settle the question. Real factual-error labels are required.

- [ ] **Step 1: Build the labelled set**

Use TriviaQA or NaturalQuestions short-answer pairs. Generate greedily with distilgpt2, mark a generation correct if the gold answer string appears in it. distilgpt2 will be wrong most of the time, which is fine — the label is what matters, and a base rate near 50% is ideal for AUROC. Record the base rate.

- [ ] **Step 2: Implement the mandatory baselines before any J-space score**

Order matters. If the baselines are written second they will be written to lose.

```python
# caustic/detect.py -- baselines
def max_softmax(logits): ...          # 1 - max p(next token), averaged over the generation
def mean_entropy(logits): ...         # predictive entropy, averaged
def mahalanobis(h, mu, prec): ...     # whitened distance of the final hidden state
def pca_recon_error(h, components): ...
```

`sigmoid`'s gate scored 0.846 mean AUROC against Mahalanobis 0.899 and PCA 0.888 — both cheaper. Assume the same until measured otherwise.

- [ ] **Step 3: Add the J-space scores**

`sigma_max`, `stable_rank`, `tail_alpha`, `log_volume`, `lam1`, `sum(lam)`, and the number of Oseledets bars with multiplicity > 1.

- [ ] **Step 4: Report AUROC for every score, with cost per token**

A table: score, AUROC, 95% CI by bootstrap, ms/token. Any J-space score that does not beat Mahalanobis on **both** AUROC and cost is a negative result and is written up as one.

- [ ] **Step 5: Commit**

```bash
git add caustic/detect.py caustic/experiments/hallucination_auroc.py EVIDENCE.md
git commit -m "Hallucination AUROC for J-space scores against Mahalanobis, PCA and softmax baselines"
```

---

### Task 5: Width invariance of the exponent

**Files:**
- Create: `caustic/experiments/width_invariance.py`

`gpt2` (D=768), `gpt2-medium` (D=1024) and `gpt2-large` (D=1280) share an architecture and differ in width. If `tail_alpha` is genuinely width-invariant it is the "finite definite constant" the thesis asks for; if it drifts with D it is not.

- [ ] **Step 1: Measure `tail_alpha` per block for all three widths on identical text**

The `BULK` index range `(10, 400)` is absolute and must be made relative to D before comparing across widths, or the comparison measures the window, not the model. Use `(D//77, D//1.92)` rounded, or state the chosen relative range explicitly.

- [ ] **Step 2: Report mean, sd and cv across widths, and re-test the block-0 exclusion on held-out text**

The block-0 exclusion was made post hoc. Re-deriving it on fresh text is what converts it from a forking path into a finding.

- [ ] **Step 3: Commit**

```bash
git add caustic/experiments/width_invariance.py EVIDENCE.md
git commit -m "tail_alpha across gpt2 widths 768/1024/1280 with a relative bulk window"
```

---

### Task 6: Does the exponent buy wall-clock

**Files:**
- Create: `caustic/experiments/prune_budget.py`

The payoff test. If `tail_alpha` says the spectrum decays at rate `a`, the rank needed for relative error `eps` follows, and that rank is a sparsity budget for the CSR schedule inherited from `triton-lang/kernels#22` via `sigmoid/schedule.py`.

- [ ] **Step 1: Derive the budget from the exponent and measure end-to-end latency against three same-budget baselines**

Random selection at the same density, locality-only (recent-k) at the same density, and oracle top-k using dense attention scores. The position between random and oracle is the result; the speedup number alone is not.

- [ ] **Step 2: Commit**

```bash
git add caustic/experiments/prune_budget.py EVIDENCE.md
git commit -m "Exponent-derived sparsity budget against random, locality and oracle at matched density"
```

---

## Phase 2 — the move catalogue

### Task 7: `CANDIDATES.md` and its screening gate

Each candidate is one row and must carry all six columns or it is not admitted:

| column | rule |
|---|---|
| hot path | exact repo, file and function, verified to exist today |
| latent structure | order / connectivity / interval / scaling — named precisely |
| current cost | the quadratic or redundant scan, with its complexity |
| predicted win | an exact formula, not an adjective |
| control | what it is measured against at equal budget |
| novelty check | the search that found no prior art, or the prior art it must beat |

Candidates are generated by the running research agents and by sweeping the author's own repos. **The target is 100 candidates, not 100 confirmed benefits.** The author's record is 601 PRs, 109 merged, 16 merged externally, 7 into major labs. A 100-candidate pool converting at that rate yields single-digit confirmed wins, and the plan is written to expect that.

Screening uses the adversarial discipline from the math-olympiad skill: a fresh-context verifier that never sees the generating argument, asymmetric voting — four confirmations to admit, two refutations to kill — and calibrated abstention over guessing.

### Task 8: The salvage rule

Every arm that fails writes a `SALVAGE` entry: what was measured, why the hypothesis died, and what the byproduct is good for. `tail_alpha` is the worked example — it failed as a hallucination detector at 1.4% separation and became a pruning-budget candidate in the same breath. This is a completion condition for every task in Phase 1.

---

## Phase 3 — Idea 2, only if Phase 1 Task 4 fails

### Task 9: Replace contour integration with combinatorial degree

The fallback thesis is "mimic complex analysis at half the compute using the three geometries together". Its concrete form: the **argument principle** counts zeros of an analytic function inside a contour by a contour integral, but the winding number it computes is a topological degree, and degree is computable combinatorially from sign changes on a subdivided boundary without any quadrature.

Target: counting eigenvalues in a region, the primitive underneath spectral slicing in eigensolvers. Baseline is Gauss–Legendre quadrature of the contour integral at the accuracy needed for a correct integer count. Measure both to the same guaranteed-correct-count criterion and report the ratio. The "half the compute" figure is a hypothesis to be measured, not an assumption.

---

## Self-review notes

- Phase 1 Task 4 is the only task that can confirm or kill Idea 1. Tasks 1–3 build the instrument; Tasks 5–6 pursue the salvage that Task 4's likely failure implies. This ordering is deliberate: the salvage path is planned before the failure, so a negative result does not strand the work.
- The seven research agents dispatched at the start of this session refine the Phase 2 candidate pool; their findings arrive after this plan is written and do not block Phase 1.
- The block-cocycle Lyapunov exponents in Task 2 are finite-time over 6 steps and are not asymptotic exponents. The plan labels them as such in three places because it is the single easiest overclaim to make here.
