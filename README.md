<h1 align="center">Caustic — The Jacobian Space of a Live Language Model</h1>

<p align="center">
  <strong>Three fields are three linearization functors. Chaos theory is what happens when you compose one of them along a trajectory.</strong><br>
  Local (tangent space) · global (homology) · across scale (power law) — met on one object,<br>
  <code>J_l(t) = ∂h_{l+1}[t] / ∂h_l[t]</code>, the layer-to-layer transport map at one token.
</p>

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong> ·
  <a href="https://github.com/teerthsharma/caustic">github.com/teerthsharma/caustic</a> ·
  <em>teerthsharma@outlook.com</em>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&color=00aaff" alt="MIT"></a>
  <a href="#10-validation"><img src="https://img.shields.io/badge/tests-30%20closed--form-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/core%20deps-numpy%20%2B%20torch-lightgrey?style=flat-square" alt="Deps"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-yellow?style=flat-square" alt="Python"></a>
  <a href="#8-negative-results"><img src="https://img.shields.io/badge/negatives-published-red?style=flat-square" alt="Negatives published"></a>
  <a href="EVIDENCE.md"><img src="https://img.shields.io/badge/every%20claim-has%20a%20control-orange?style=flat-square" alt="Controls"></a>
</p>

---

## Abstract

**Caustic** measures the Jacobian of a running transformer and asks whether that operator
carries an invariant a point cloud of hidden states cannot. The organising claim is that
differential geometry, algebraic topology and fractal geometry are three *linearization
functors* applied to different structures — the tangent space linearizes locally, homology
linearizes globally into vector spaces, a power law linearizes across scale — and that
chaos theory is what the local functor becomes when composed along a trajectory. Oseledets'
Multiplicative Ergodic Theorem states that products of Jacobians induce a filtration of the
state space by exponential growth rate, and a filtration indexed by a real parameter that
decomposes into interval pieces is the same shape of object persistent homology consumes.
That coincidence is the bridge this repository is built to test. The object studied is
`J_l(t) = ∂h_{l+1}[t] / ∂h_l[t]`, chosen over a point cloud because a point cloud is an
observable of the past — measured in the author's own `sigmoid`, a persistence signature
over a token window reads lexical repetition at **R² 0.537** and predicts nothing forward —
whereas the Jacobian is by construction the operator that transports a perturbation forward.

Three experiments ship, each against a control. The cost probe found that at `D = 768` the
**exact Jacobian is 26.6× cheaper than the Krylov estimator of its own top eight singular
values**, which agree to `1.685e-04`; the premise that Jacobian space needs a cheap bound to
be tractable is false at this scale. The layer sweep is a **negative result**: three of four
spectral summaries flip sign across layers against a shuffled-token control, with |mean|
separations of `0.0085`, `0.1138`, `0.0217` and `0.0548`. The Lyapunov experiment produced
the first quantity to separate with consistent sign — grounded text is *less* chaotic and
contracts *more* volume than its scrambled control — on a converged 46-step token cocycle,
while a 6-step block cocycle whose drift (`0.0255`) exceeded its own value (`0.0224`) was
discarded rather than reported. The geometric-detector arm has **not** beaten Mahalanobis,
and the honest baseline numbers are in §9.

**Keywords:** Jacobian, transformer interpretability, Lyapunov exponents, Oseledets
multiplicative ergodic theorem, cocycle, persistent homology, filtration, singular spectrum,
power-law tail exponent, stable rank, volume contraction, folding, caustics, singularity
theory, hallucination detection, out-of-distribution detection, reverse-mode automatic
differentiation, Krylov subspace methods

---

## Table of contents

| § | Section |
|---|---|
| [1](#1-introduction) | Introduction |
| [2](#2-what-the-jacobian-conjecture-does-not-give) | What the Jacobian *Conjecture* does not give |
| [3](#3-why-this-repository-is-called-caustic) | Why this repository is called Caustic |
| [4](#4-theoretical-foundation) | Theoretical foundation |
| [5](#5-implementation) | Implementation |
| [6](#6-the-cost-of-working-in-jacobian-space) | The cost of working in Jacobian space |
| [7](#7-finite-time-lyapunov-exponents) | Finite-time Lyapunov exponents |
| [8](#8-negative-results) | Negative results |
| [9](#9-prior-art) | Prior art |
| [10](#10-validation) | Validation |
| [11](#11-quick-start) | Quick start |
| [12](#12-requirements) | Requirements |
| [13](#13-limitations) | Limitations |
| [14](#14-what-is-not-built-yet) | What is not built yet |

---

## 1. Introduction

Three mathematical fields are usually taught as three subjects. They are better read as one
move applied to three different structures — replace a hard object by a linear one, and keep
a functor that says the replacement was faithful.

**Differential geometry linearizes locally.** The tangent space is the linear model of a
manifold at a point; the Jacobian is the induced map between two tangent spaces. Everything
it knows is about an infinitesimal neighbourhood.

**Algebraic topology linearizes globally.** Homology is literally a functor to vector
spaces, and a persistence module over a real parameter decomposes into interval pieces — a
barcode. What it knows is connectivity at every scale at once, with coordinates discarded.

**Fractal geometry linearizes across scale.** A power law is a straight line in log-log. The
exponent is what survives when the units are thrown away, so it is the natural home for any
quantity that should be a property of the architecture rather than of the run.

**Chaos theory is the composition.** Take the local functor and compose it along a
trajectory. Oseledets' Multiplicative Ergodic Theorem says the resulting products of
Jacobians induce a *filtration* of the state space by exponential growth rate. A filtration
indexed by a real parameter that decomposes into interval pieces is the same shape of object
persistent homology consumes. The two global structures meet, and that meeting is the spine
of the programme in [`PLAN.md`](PLAN.md).

### 1.1 Why the Jacobian, and not a point cloud of hidden states

This choice is the load-bearing one, and it was made by a measurement rather than by taste.

A point cloud of hidden states over a token window is an **observable of the past**. In the
author's own `sigmoid`, `examples/why_tokens_fail.py` measured what a persistence signature
over such a window actually encodes: **lexical repetition, at R² 0.537** for distinct-type
count against **0.087** for the equal-dimension linear channel. It reads which tokens
repeated. It has no forward influence on the dynamics, and it therefore cannot predict
forward. That negative is treated here as settled and is not re-run.

The Jacobian `J_l(t)` is the opposite kind of object. It is, by construction, the operator
that carries a perturbation of `h_l[t]` forward into `h_{l+1}[t]`. Causality is not a
hypothesis about it — it is its definition. That is the entire reason for working in Jacobian
space, and it is the sentence to keep if only one survives.

---

## 2. What the Jacobian *Conjecture* does not give

State this early, because it is the most likely misreading of the repository's name and
subject.

**The Jacobian Conjecture does not transfer.** It concerns polynomial maps
`ℂⁿ → ℂⁿ` whose Jacobian determinant is a nonzero constant, and asks whether such a map must
be invertible. A transformer is not a polynomial map, and its `det J` is not constant — the
layer sweep measured `logdet_abs` varying block to block and condition to condition. Any
argument of the form "the conjecture says full-rank implies invertible, therefore the network
is injective" is invalid twice over. **No claim in this repository rests on the theorem, and
no task in the plan may cite it as support.**

What does transfer is the **moral of Alpöge's July 2026 counterexample**, and only the moral:

> Local invertibility everywhere does not imply global injectivity.

That is a statement about maps, and it is the failure mode this programme is built to look
for in a language model. A model can have full-rank Jacobians at every point it ever visits
and still fold two distinct grounded contexts onto one internal state. Full rank is a local
certificate; injectivity is a global property; the gap between them is where a model can lose
the information that distinguished two situations. Volume contraction — measured in §7 — is
the **necessary precondition** for that folding. It is necessary and explicitly not
sufficient: a map can contract volume and remain injective.

---

## 3. Why this repository is called Caustic

In singularity theory a **caustic** is the image of the critical set: the locus where a
smooth map folds and distinct preimages merge. It is the bright curve at the bottom of a cup
of coffee, and it is exactly the geometric event described in §2 — the place where local
invertibility everywhere fails to prevent two inputs arriving at one output.

The name is a hypothesis about where to look, not a claim that the object has been found.

---

## 4. Theoretical foundation

Only equations the code implements appear here. Symbols are defined immediately below each
block.

### 4.1 The object

**(1)**

$$
J_l(t) \;=\; \frac{\partial h_{l+1}[t]}{\partial h_l[t]} \;\in\; \mathbb{R}^{D \times D}
$$

`h_l ∈ ℝ^{1×T×D}` is the observed hidden state entering block `l`; `t` is one token
position; `D` is the model width (768 for distilgpt2). Every other position is held at its
observed value, so (1) is the **diagonal block** of the full `(TD × TD)` Jacobian. The block
remains causal: position `t` still attends over the frozen prefix. This is implemented by
`caustic.jacobian.block_map`, and differentiated by batched reverse-mode AD in
`exact_jacobian`.

### 4.2 The cocycle

Composing (1) along a trajectory of `n` steps gives the linear model of the whole trajectory:

**(2)**

$$
J^{(n)} \;=\; J_n J_{n-1} \cdots J_1
$$

Two distinct cocycles exist in a transformer and are **not** interchangeable: across *blocks*
at a fixed token, and across *tokens* at a fixed block. Both are run in
`caustic/experiments/lyapunov_llm.py`; they behave differently, and §7 reports why only one
of them produced a usable number.

### 4.3 Oseledets' filtration

The Multiplicative Ergodic Theorem guarantees that the growth rates of (2) exist, and that
they organise the state space into a filtration

**(3)**

$$
\mathbb{R}^D = V_1 \supset V_2 \supset \cdots \supset V_k \supset \{0\},
\qquad
\lim_{n \to \infty} \frac{1}{n} \log \lVert J^{(n)} v \rVert = \lambda_i
\;\;\text{for}\;\; v \in V_i \setminus V_{i+1}
$$

`λ_1 ≥ λ_2 ≥ … ≥ λ_k` are the Lyapunov exponents and `dim V_i − dim V_{i+1}` is the
multiplicity of `λ_i`. **This is the bridge.** A filtration indexed by a real parameter,
decomposing into interval pieces of stated multiplicity, is structurally a persistence
barcode — so barcode tooling applies to it without modification. The honest caveat, stated
before the code was written: when all `D` exponents are distinct the filtration has `D` steps
of multiplicity one and carries exactly the information the sorted spectrum already carried.
The bridge is informative only when exponents **cluster**. Whether they do on a real model is
an open question here — see §14.

### 4.4 Benettin's algorithm

Evaluating (2) directly does not work in floating point. Every column of the running product
collapses onto the leading singular direction within a few dozen steps, and the norm
overflows or underflows shortly after, so all but the top exponent are lost. The fix is to
re-orthonormalize the frame after each step and accumulate the log diagonal of `R`:

**(4)**

$$
J_k Q_{k-1} = Q_k R_k,
\qquad
\lambda_i \;=\; \frac{1}{n\,\Delta t}\sum_{k=1}^{n} \log \lvert (R_k)_{ii} \rvert
$$

`Q_0 = I`. QR is unique only up to the sign of each column and LAPACK's choice of sign is not
stable across inputs, so `caustic.cocycle._qr_step` folds the sign into `Q` to keep `R`'s
diagonal positive — otherwise the accumulated logs are not comparable step to step. A
genuinely singular step is clamped at `1e-300`, giving about `−690` rather than `−inf`;
folding produces exactly that case, so it is a **supported input, not an error**.

### 4.5 Scalar summaries of one spectrum

With `σ_1 ≥ σ_2 ≥ … ≥ σ_D` the singular values of a single `J`:

**(5)**

$$
\sigma_{\max} = \sigma_1,
\qquad
\mathrm{srank}(J) = \frac{\lVert J \rVert_F^2}{\sigma_1^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2},
\qquad
\log \mathrm{vol}(J) = \sum_i \log \sigma_i
$$

`σ_max` is local expansion and bounds perturbation growth under products. Stable rank is the
effective number of active directions, bounded in `[1, D]` and scale-invariant by
construction; **measured at 9.69 out of 768** on distilgpt2 layer 3, so the operator is
effectively low rank and the "large matrix" premise behind iterative spectral methods is
false here — which is the mechanism behind the cost result in §6. `log vol` is the local log
volume change; large and negative is the coarse signature of folding.

The fractal summary is the log-log slope over the bulk of the spectrum:

**(6)**

$$
\sigma_i \sim i^{-a}
\quad\Longrightarrow\quad
a \;=\; -\,\mathrm{slope}\big(\log \sigma_i \;\text{vs}\; \log i\big),
\qquad i \in [\,10,\ 400\,)
$$

The bulk window excludes the leading spike and the trailing numerical floor, neither of which
belongs to a power-law tail. `a` is scale-invariant: multiplying every `σ` by `c` shifts the
intercept and leaves the slope unchanged, which is asserted in the test suite to `1e-9`. The
window `(10, 400)` is **absolute** and must be made relative to `D` before any cross-width
comparison, or the comparison measures the window rather than the model.

---

## 5. Implementation

The package ships flat at the repository root. There is no `src/` wrapper and no nested
`caustic/caustic/`.

| File | Responsibility |
|---|---|
| [`caustic/jacobian.py`](caustic/jacobian.py) | `block_map` isolates the diagonal block of the full Jacobian; `exact_jacobian` differentiates it by batched reverse-mode AD; `top_singular_values` is the Krylov path that never materializes `J`; `singular_values` is the exact reference |
| [`caustic/spectrum.py`](caustic/spectrum.py) | `sigma_max`, `stable_rank`, `tail_alpha`, `log_volume`, `summarize` — equations (5) and (6). `BULK = (10, 400)` is the fit window and is exported so it can be overridden |
| [`caustic/cocycle.py`](caustic/cocycle.py) | `lyapunov_spectrum` — equation (4) by QR iteration. `finite_time_spectrum` returns the running exponents after **every** step, shape `(n, D)`, so a report can show whether a value had settled or was still moving when the trajectory ran out |
| [`caustic/experiments/probe_cost.py`](caustic/experiments/probe_cost.py) | Cost of each estimator against the block forward pass as the unit; §6 |
| [`caustic/experiments/layer_sweep.py`](caustic/experiments/layer_sweep.py) | Spectral summaries across 6 blocks × 10 positions, grounded against a shuffled-token control; §8 |
| [`caustic/experiments/lyapunov_llm.py`](caustic/experiments/lyapunov_llm.py) | Both cocycles of §4.2 on a live model, against the same control; §7 |
| [`tests/test_jacobian.py`](tests/test_jacobian.py) | 22 assertions against closed-form ground truth |
| [`tests/test_cocycle.py`](tests/test_cocycle.py) | 8 assertions against systems whose exponents are known exactly |
| [`EVIDENCE.md`](EVIDENCE.md) | Every measured number, its control, and every negative |
| [`PLAN.md`](PLAN.md) | The thesis and the task-by-task programme |
| [`CANDIDATES.md`](CANDIDATES.md) | The move catalogue — one row per candidate, six mandatory columns, with the killed and deferred entries kept |

`finite_time_spectrum` exists because Lyapunov exponents are an **asymptotic** statement. Over
a 6-block transformer the block cocycle has 6 steps, which is nowhere near asymptotic, and
calling that result a Lyapunov exponent would be an overclaim. Returning the whole convergence
trace is what let §7 catch and discard a number that would otherwise have been reported.

---

## 6. The cost of working in Jacobian space

distilgpt2, layer 3, one token position. Median of 30 calls after 3 warmups, CUDA synced.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block forward                 0.588 ms
  one JVP                       6.144 ms      10.45x forward
  top-8 power, 20 iters      1428.938 ms    2429.5x forward
  full 768x768 jacrev          53.694 ms      91.3x forward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**The exact Jacobian is 26.6× cheaper than the Krylov estimator of its own top eight singular
values.**

**Control:** the exact computation, same input, same hardware, same accuracy target.

The estimator is not wrong. It agrees with exact `svdvals` to **`1.685e-04`** max relative
error on the top 8 (k=8, iters=20). It simply loses at `D = 768`, and the mechanism is
specific: batched reverse-mode AD vectorizes across all 768 output components in one pass,
while `k`-column power iteration runs `k × iters` sequential JVP/VJP passes with no batching.
The stable rank of 9.69 from §4.5 is the same fact from the other side — there is no large
effective matrix here for an iterative method to exploit.

**Not measured:** the width at which the ordering inverts. It is not assumed, and
`top_singular_values` is kept precisely for the regime beyond that crossover. Anyone quoting
this ratio for a model larger than distilgpt2 is quoting it outside its measured range.

**Consequence for the programme:** the premise that Jacobian space needs a cheap spectral
bound to be tractable is **false at this scale**. Bounds matter for large `D` only. This is
also the seed of candidate `C2` in [`CANDIDATES.md`](CANDIDATES.md) — a general class of
spectral estimator that defaults to an iterative method regardless of problem size — and it
was found by running the code rather than by reasoning about it.

---

## 7. Finite-time Lyapunov exponents

Both cocycles of §4.2, both against the same shuffled-token control. The shuffle preserves
the token multiset and therefore the embedding statistics, and destroys only grounded
structure.

### 7.1 Token cocycle, block 3, 46 steps — converged

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  lam1        sum       positive     last-step drift
  grounded     +0.1653    -226.74      139/768          0.0012
  shuffled     +0.1852    -170.37      151/768          0.0003
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Drift of `0.0012` on a value of `0.1653` is **0.7%**, and `0.0003` on `0.1852` is **0.2%**.
Both had settled, which is why these numbers are reported at all.

**What they say.** The model is locally chaotic and strongly dissipative. `λ₁ > 0` in every
condition means perturbations grow along at least one direction. A sum near `−227` over 768
dimensions means the map contracts volume enormously per step, with only 139 of 768 directions
expanding. A positive leading exponent together with a strongly negative sum is the signature
of a dissipative chaotic system on a low-dimensional attractor — and, per §2, volume
contraction is the necessary precondition for folding. Necessary, not sufficient.

### 7.2 Separation, and why it matters more than the values

```
  block  lam1 -5.3274   sum -0.0943   n_positive -0.1646
  token  lam1 -0.1202   sum -0.2486   n_positive -0.0863
```

Separation is `(grounded − shuffled)` as a fraction of the grounded value. **All six are
negative.** Grounded text is *less* chaotic and contracts *more* volume than the scrambled
control. The largest converged separation is `sum` on the token cocycle, at **25%**.

**This is the first quantity in the programme that separated with consistent sign.** In the
layer sweep of §8, three of four summaries flipped.

The block `λ₁` figure of `−5.3274` is an artefact of dividing by a near-zero grounded value
(`0.0224`); the absolute difference is `−0.119`, and the underlying number is not converged
anyway — see §7.3.

### 7.3 Block cocycle, 6 steps — discarded, and why

```
  grounded     +0.0224    -276.65       79/768          0.0255
  shuffled     +0.1416    -250.56       92/768          0.0083
```

The grounded leading exponent traces `0.364, 0.027, −0.016, −0.002, −0.003, 0.022` across its
six steps and is still oscillating at the last one. **The drift, `0.0255`, is larger than the
value, `0.0224`.** That number is noise, and no claim in this repository rests on it.

Six steps is nowhere near the asymptotic regime equation (3) describes, and a six-step average
is not a Lyapunov exponent. It is reported here rather than deleted because deleting it is how
a plausible-looking figure survives into a table.

### 7.4 What §7 is not

`n = 1` passage, 1 model, 1 seed, no error bars across texts. **The shuffle is an
out-of-distribution control, not a hallucination control.** It cannot settle whether Jacobian
space predicts factual error. Only a run against real factual-error labels, with Mahalanobis,
PCA and max-softmax baselines at matched cost, can do that, and it has not been run.

---

## 8. Negative results

A first-class section. Every entry names what it was compared against.

### 8.1 Spectral summaries do not separate grounded from shuffled text

6 blocks, 10 token positions, one 61-token passage, against the same token multiset in
shuffled order. Separation as a fraction of the grounded value:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  sigma_max     [ 0.123 -0.012 -0.135 -0.123  0.233 -0.035]  |mean| 0.0085
  stable_rank   [ 0.346  0.165  0.196  0.091 -0.299  0.184]  |mean| 0.1138
  tail_alpha    [ 0.051 -0.088 -0.091 -0.037  0.067  0.227]  |mean| 0.0217
  logdet_abs    [ 0.805  0.187  0.002 -0.111 -0.282 -0.273]  |mean| 0.0548
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Three of the four flip sign across layers.** Only `stable_rank` holds sign, at 5 of 6
layers, median **+17.5%**, with block 4 reversing it. No summary separates the conditions
reliably.

**Control:** the shuffled-token condition, matched token multiset, same model, same positions.

This kills "J-space summary statistics as a hallucination detector" as stated, and it is
recorded as `K1` in [`CANDIDATES.md`](CANDIDATES.md) so it is not proposed again. The caveat
that keeps it open rather than closed is the one in §7.4: a shuffled-token control is an OOD
control. "J-space does not detect hallucination" is supported here only *against scrambling*.

### 8.2 The salvage: `tail_alpha` is structural, not semantic

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  tail_alpha, blocks 1-5
    grounded    0.2374 +/- 0.0293    cv 0.124
    shuffled    0.2342 +/- 0.0481    cv 0.205
    difference  0.0033  =  1.4% of grounded
    block 0     0.6168               excluded, see caveat
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Agreement to **1.4%** between coherent and scrambled text means the exponent is a property of
the **architecture** rather than of the content. That is precisely what disqualifies it as a
detector, and precisely what qualifies it as a **pruning budget**: for `σ_i ~ i^{-a}` the rank
needed for a stated relative error follows in closed form, so a measured exponent replaces a
tuned density constant. That is candidate `C1`. The negative result created the candidate.

**Caveat, unresolved, and load-bearing:** block 0 was excluded **after** inspecting the data.
That is a forking path. It must be re-derived on held-out text before it is reported as a
finding, and until then `C1` rests on an untested exclusion. `C1` additionally rests on
`tail_alpha` being width-invariant, which has **not** been tested; if the exponent drifts with
`D`, the candidate dies.

### 8.3 The 6-step block cocycle

Reported in §7.3. Discarded because its drift exceeded its own value.

### 8.4 The point cloud, ruled out before this repository started

Persistence over a point cloud of hidden states does not predict forward: **R² 0.537** for
lexical distinct-type count against **0.087** for the equal-dimension linear channel, measured
in `sigmoid/examples/why_tokens_fail.py`. This is the reason the programme works in Jacobian
space at all, and it is not re-run here.

### 8.5 Two environment facts, both found by running rather than reading

- `torch.func.jvp` raises `NotImplementedError` on
  `_scaled_dot_product_efficient_attention`. Forward-mode AD is unimplemented for the fused
  SDPA kernel that `transformers` selects by default. Any JVP-based method must load with
  `attn_implementation="eager"`. Reverse mode is unaffected. Recorded as `D1` in
  [`CANDIDATES.md`](CANDIDATES.md), deferred rather than admitted because the natural upstream
  target is off-limits.
- `transformers >= 5` returns a bare tensor from a block where earlier versions returned a
  tuple. Indexing `[0]` on the new return **silently takes the batch dimension** instead of
  the hidden states, producing a wrong-shaped result rather than an error.
  `caustic.jacobian.block_map` handles both.

---

## 9. Prior art

Where this sits, and — more importantly — where the alternatives are better. Numbers in the
AUROC column come from `sigmoid/examples/gate_ood_benchmark.py`, an **OOD** benchmark on the
author's own prior gate, not a factual-error benchmark. Cells with no measurement say so.

| Approach | Which linearization | Object it reads | Forward-causal by construction | Mean AUROC, author's OOD gate benchmark | Where it beats this work |
|---|---|---|---|---|---|
| Persistent homology as ML features | global | point cloud of hidden states | no — an observable of the past | not run | mature tooling; reaches H₁/H₂, where this repository computes none |
| Semantic entropy for hallucination detection | none — sampling, not geometry | distribution over resampled generations | n/a | not run | targets factual error directly and needs no white-box access to the Jacobian |
| Mahalanobis distance on the final hidden state | local — one Gaussian | one hidden vector | no | **0.899** | **cheaper *and* higher AUROC. This is the arm to beat, and it has not been beaten** |
| PCA reconstruction error | global linear subspace | one hidden vector | no | **0.888** | cheaper; also beats the author's own topological gate |
| Dynamical isometry / mean-field signal propagation | local | Jacobian singular spectrum, at initialization | yes | not run | same object, established theory, and it produces architecture-level design rules — but it is an initialization-time and ensemble-level statement, not a per-token measurement on a trained model at inference |
| Intrinsic-dimension estimation | across scale | point cloud of hidden states | no | not run | one scalar, very cheap, no autodiff required |
| The author's prior topological gate (`sigmoid`) | global | barcode of an activation window | no | 0.846 | — it loses to both baselines above, which is why this repository exists |
| **`caustic`** | all three, composed along a trajectory (3) | `J_l(t)`, the transport operator | **yes** | **not yet run** | — nothing yet. §7 is a separation against a *scrambling* control only |

**The honest summary of this table:** the geometric-detector arm has not yet beaten
Mahalanobis. The author's own previous attempt scored **0.846** against Mahalanobis **0.899**
and PCA **0.888** — both of which are cheaper. The working assumption until measured otherwise
is that the same ordering holds here, and any J-space score that fails to beat Mahalanobis on
**both** AUROC and cost per token will be written up as a negative result rather than dropped.

Not benchmarked, and deliberately not estimated: semantic entropy, persistent-homology
feature pipelines, and intrinsic-dimension estimators are not installed and not run on this
host. Inventing a figure for them would be worse than the gap.

---

## 10. Validation

**30 tests, every one against a closed-form answer rather than against self-consistency.**

The reason for that standard is that the expensive failures in a topological pipeline are
silent. A wrong filtration, a transposed index, or a scale divided out where it should have
been kept does not raise: it produces a well-formed diagram with healthy variance and no
information, the downstream fit still trains, and the benchmark still reports a number. A
self-consistency test passes on every one of them. Only an assertion with an independently
known answer fails.

| Assertion | Tolerance |
|---|---|
| Jacobian of a position-wise linear block equals its weight matrix | `1e-10` |
| Power iteration matches exact `svdvals` | `1e-6` |
| `tail_alpha` recovers synthetic exponents 0.25 / 0.5 / 1.0 / 2.0 | `1e-9` |
| `log_volume` equals `torch.linalg.slogdet` | `1e-8` |
| Flat spectrum returns exponent 0 *(negative control)* | `1e-9` |
| Diagonal cocycle returns `log a` | `1e-8` |
| Orthogonal cocycle returns 0 *(negative control)* | `1e-8` |
| Exponents sum to mean `log|det|` | `1e-8` |
| Scaling every Jacobian by `c` shifts every exponent by `log c` | `1e-9` |
| Rank-deficient step drives one exponent below `−100`, others finite | — |

The last row is the folding case from §2 and §4.4: it must be a supported input that produces
a readable answer, not a `NaN` and not an exception.

```bash
python -m pytest tests/ -q          # 30 tests, no model download, no GPU required
```

---

## 11. Quick start

```bash
git clone https://github.com/teerthsharma/caustic.git && cd caustic
pip install -e .                    # numpy + torch
pip install -e ".[experiments,dev]" # adds transformers and pytest
```

```bash
python -m pytest tests/ -q                       # 30 closed-form assertions
python -m caustic.experiments.probe_cost         # the cost table of §6
python -m caustic.experiments.layer_sweep        # the negative result of §8.1
python -m caustic.experiments.lyapunov_llm       # both cocycles of §7
```

The three experiments download `distilgpt2` on first run and take a few minutes each on the
hardware in §12. `probe_cost` needs no `transformers`-side configuration beyond the eager
attention requirement below.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from caustic import exact_jacobian, singular_values, summarize
from caustic.cocycle import finite_time_spectrum, lyapunov_spectrum

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    dtype=torch.float32,
    attn_implementation="eager",     # required; see §12
).eval()
for p in model.parameters():
    p.requires_grad_(False)

ids = tok("The capital of France is Paris.", return_tensors="pt").input_ids
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states

# One Jacobian, and the four summaries of equations (5) and (6).
J = exact_jacobian(model.transformer.h[3], hs[3].detach(), pos=-1)
print(summarize(singular_values(J).cpu().numpy()))

# The token cocycle at a fixed block, and its convergence trace.
Js = [exact_jacobian(model.transformer.h[3], hs[3].detach(), t).cpu()
      for t in range(ids.shape[1])]
lam = lyapunov_spectrum(Js)
trace = finite_time_spectrum(Js)                 # (n, D) -- check it settled

print(f"lam1 {lam[0]:+.4f}  sum {lam.sum():+.2f}  positive {(lam > 0).sum()}/{len(lam)}")
print(f"last-step drift {abs(trace[-1, 0] - trace[-2, 0]):.4f}")
```

**Read the drift before quoting `lam1`.** That single line is what separates §7.1 from §7.3.

---

## 12. Requirements

| | |
|---|---|
| **Python** | `>= 3.10` |
| **Core** | `numpy >= 1.24`, `torch >= 2.4` |
| **Experiments** | `transformers >= 4.40` |
| **Dev** | `pytest >= 8` |

**Measured on:** NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB. Intel64. Windows 11.
Python 3.11.9, PyTorch 2.5.1+cu121, transformers 5.3.0, float32, seed 0.
Model: distilgpt2, `D = 768`, 6 blocks. Every number in this README was produced on that
host; none is projected to any other.

**`attn_implementation="eager"` is required.** `torch.func.jvp` raises `NotImplementedError`
on `_scaled_dot_product_efficient_attention`, the fused SDPA kernel `transformers` selects by
default — forward-mode AD is simply not implemented for it. `top_singular_values` uses JVP and
will fail without eager attention. `exact_jacobian` uses reverse mode and is unaffected, but
the experiments load eager throughout so that both paths are comparable on the same model.

**Determinism.** Anything feeding an A/B comparison runs under
`torch.use_deterministic_algorithms(True)`. Non-determinism in the measurement invalidates the
measurement.

**8 GiB is enough** for everything reported here, because a `768 × 768` float32 Jacobian is
2.25 MiB and the cocycles hold at most a few dozen of them on CPU. Nothing in this repository
has been run at a width where that stops being true.

---

## 13. Limitations

**The detector question is open, and the honest prior is that it fails.** No run against real
factual-error labels exists. Every separation reported here is against a shuffled-token
control, which is an out-of-distribution control. Mahalanobis at 0.899 and PCA at 0.888 are
cheaper than anything in this repository and have not been beaten by it.

**`n = 1` throughout.** One passage, one model, one seed, no error bars across texts. The
Lyapunov separations of §7.2 are six numbers from a single trajectory pair. They agree in sign,
which is why they are reported; they are not a statistical result.

**One model, one width.** distilgpt2 at `D = 768`. The cost inversion of §6, the `9.69` stable
rank, and the `0.2374` exponent are all measured at that single width. The crossover width at
which the Krylov estimator starts to win is **not measured and must not be assumed**. The
`BULK = (10, 400)` window is absolute and would have to be made relative to `D` before any
cross-width comparison is meaningful.

**One post-hoc decision is outstanding.** Block 0 was excluded from the `tail_alpha` fit after
inspecting the data. It has not been re-derived on held-out text, so §8.2 is a forking path
until it is.

**The Oseledets bridge is untested on real data.** Equation (3) is implemented only as far as
the exponents; the interval encoding of the filtration is not in this repository. If the
exponents of a real model are all distinct, the encoding carries nothing the sorted spectrum
did not, and that outcome must be reported rather than hidden.

**The block cocycle is structurally too short.** Six blocks is six steps. No amount of care
makes a six-step average asymptotic, so the block cocycle cannot be fixed by better numerics —
it needs a deeper model.

**Only the diagonal block is computed.** `J_l(t)` holds every other token position at its
observed value. The full `(TD × TD)` Jacobian, including the cross-position terms that
attention creates, is not computed anywhere here.

**H₀ only, and in fact no homology at all yet.** The topological half of the thesis is present
as the structural argument of §4.3 and as tooling inherited from the author's other work. This
repository computes no barcodes.

---

## 14. What is not built yet

Stated so the file table above is not mistaken for a roadmap. Full task breakdowns, with the
tests written before the implementations, are in [`PLAN.md`](PLAN.md).

| Planned | What it would settle |
|---|---|
| `caustic/oseledets.py` | The interval encoding of the filtration (3), and whether multiplicities on a real model ever exceed 1. If they do not, the bridge buys nothing and that is the falsification of the spine's second half |
| `caustic/detect.py` + `experiments/hallucination_auroc.py` | The only experiment that can confirm or kill the detector arm: real factual-error labels, with `max_softmax`, `mean_entropy`, Mahalanobis and PCA baselines implemented **first**, so they are not written to lose |
| `experiments/width_invariance.py` | Whether `tail_alpha` is width-invariant across `D = 768 / 1024 / 1280`, and a re-derivation of the block-0 exclusion on held-out text |
| `experiments/prune_budget.py` | Whether the exponent buys wall-clock, against random selection, locality-only, and oracle top-`k` at **matched density**. The position between random and oracle is the result; a speedup number alone is not |

The evidence standard those tasks inherit is one sentence, and it is the one that survived the
author's previous build: **a claim needs a control.** Four previously shipped claims in
`sigmoid` were wrong, and every one of them was wrong because a control was missing — never
because the mathematics failed.

---

## License

MIT, as declared in [`pyproject.toml`](pyproject.toml). No `LICENSE` file is committed yet.

<p align="center">
  <strong>MIT © <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong><br>
  <em>A caustic is the image of the critical set — the locus where a smooth map folds<br>
  and distinct preimages merge. The name is a hypothesis about where to look.</em>
</p>
