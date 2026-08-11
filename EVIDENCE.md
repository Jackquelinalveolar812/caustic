# Evidence

Every number measured in-process on this machine. Numbers that came out badly are
reported as they came out. Each entry names what it was compared against; an entry
without a control is not evidence and is marked as such.

**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB. Intel64. Windows 11.
**Software:** Python 3.11.9, PyTorch 2.5.1+cu121, transformers 5.3.0, float32, seed 0.
**Model:** distilgpt2, D=768, 6 blocks, `attn_implementation="eager"`.

---

## 1. Cost of working in Jacobian space

distilgpt2, layer 3, one token position. Median of 30 calls after 3 warmups, CUDA synced.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block forward                 0.588 ms
  one JVP                       6.144 ms      10.45x forward
  top-8 power, 20 iters      1428.938 ms    2429.5x forward
  full 768x768 jacrev          53.694 ms      91.3x forward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**The exact Jacobian is 26.6x cheaper than the Krylov estimator of its own top
eight singular values.** The estimator is not wrong — it agrees with exact
`svdvals` to `1.685e-04` max relative error on the top 8 — it simply loses at
D=768, because batched reverse-mode AD vectorizes across all 768 output
components while `k`-column power iteration runs `k * iters` sequential passes.

**Control:** the exact computation, same input, same hardware.
**Not measured:** the width at which the ordering inverts. It is not assumed.

**Consequence for the programme:** the premise that Jacobian space needs a cheap
bound to be tractable is false at this scale. Bounds matter for large D only.

## 2. Two environment facts, both found by running rather than reading

- `torch.func.jvp` raises `NotImplementedError` on
  `_scaled_dot_product_efficient_attention`. Forward-mode AD is unimplemented for
  the fused SDPA kernel that transformers selects by default. Any JVP-based method
  must load with `attn_implementation="eager"`. Reverse mode is unaffected.
- transformers >= 5 returns a bare tensor from a block where earlier versions
  returned a tuple. Indexing `[0]` on the new return silently takes the batch
  dimension instead of the hidden states, producing a wrong-shaped result rather
  than an error.

---

## 3. Spectral summaries across blocks — NEGATIVE

6 blocks, 10 token positions, one 61-token passage, against the same token
multiset in shuffled order. The shuffle preserves embedding statistics and
destroys only grounded structure.

Separation (grounded − shuffled) as a fraction of the grounded value:

```
sigma_max     per-layer [ 0.123 -0.012 -0.135 -0.123  0.233 -0.035]  |mean| 0.0085
stable_rank   per-layer [ 0.346  0.165  0.196  0.091 -0.299  0.184]  |mean| 0.1138
tail_alpha    per-layer [ 0.051 -0.088 -0.091 -0.037  0.067  0.227]  |mean| 0.0217
logdet_abs    per-layer [ 0.805  0.187  0.002 -0.111 -0.282 -0.273]  |mean| 0.0548
```

**Three of four flip sign across layers.** Only `stable_rank` holds sign, at 5 of
6 layers, median +17.5%, with block 4 reversing. No summary separates the
conditions reliably.

### The salvage: `tail_alpha` is structural, not semantic

```
tail_alpha, blocks 1-5
  grounded    0.2374 +/- 0.0293    cv 0.124
  shuffled    0.2342 +/- 0.0481    cv 0.205
  difference  0.0033  =  1.4% of grounded
  block 0     0.6168               excluded, see caveat
```

Agreement to 1.4% between coherent and scrambled text means the exponent is a
property of the architecture rather than the content. That kills it as a detector
and is exactly what qualifies it as a pruning budget (candidate C1).

**Caveat, unresolved:** block 0 was excluded *after* inspecting the data. That is
a forking path. It must be re-derived on held-out text before it is reported as a
finding. Plan Task 5 does this.

---

## 4. Finite-time Lyapunov exponents — the first consistent separation

Two cocycles, both against the shuffled control.

### Token cocycle, block 3, 46 steps — CONVERGED, trustworthy

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  lam1        sum       positive     last-step drift
  grounded     +0.1653    -226.74      139/768          0.0012
  shuffled     +0.1852    -170.37      151/768          0.0003
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Drift of 0.0012 on a value of 0.1653 is 0.7%, and 0.0003 on 0.1852 is 0.2%. Both
had settled.

### Block cocycle, 6 steps — NOT CONVERGED, discarded

```
  grounded     +0.0224    -276.65       79/768          0.0255
  shuffled     +0.1416    -250.56       92/768          0.0083
```

The grounded leading exponent traces `0.364, 0.027, -0.016, -0.002, -0.003, 0.022`
and is still oscillating at the last step. **The drift, 0.0255, is larger than the
value, 0.0224.** That number is noise and no claim rests on it. Six steps is
nowhere near the asymptotic regime Oseledets' theorem describes, and a 6-step
average is not a Lyapunov exponent.

### What the converged numbers say

**The model is locally chaotic and strongly dissipative.** `lam1 > 0` in every
condition means perturbations grow along at least one direction. `sum(lam)` near
−227 over 768 dimensions means the map contracts volume enormously per step, with
only 139 of 768 directions expanding.

Positive leading exponent with a strongly negative sum is the signature of a
dissipative chaotic system on a low-dimensional attractor. **Volume contraction is
the necessary precondition for folding**, which is the mechanism the programme is
testing. It is necessary and not sufficient: a map can contract volume and remain
injective.

### Separation — all six same sign, unlike §3

```
  block  lam1 -5.3274   sum -0.0943   n_positive -0.1646
  token  lam1 -0.1202   sum -0.2486   n_positive -0.0863
```

Every one is negative. Grounded text is **less chaotic and contracts more volume**
than the scrambled control. The largest converged separation is `sum` on the token
cocycle at 25%.

The block `lam1` figure of −5.33 is an artefact of dividing by a near-zero
grounded value (0.0224); the absolute difference is −0.119, and the underlying
number is not converged anyway.

**This is the first quantity that separated with consistent sign.** In §3 three of
four flipped.

**What it is not:** n = 1 passage, 1 model, 1 seed, no error bars across texts.
The shuffle is an out-of-distribution control, not a hallucination control. It
cannot settle whether J-space predicts factual error; only Plan Task 4, against
real factual-error labels with Mahalanobis, PCA and max-softmax baselines, can do
that.

---

## 5. The Oseledets filtration — NEGATIVE, and it falsifies half the spine

The programme's organising claim has two halves. The first is that the Jacobian is
the causal object where a point cloud is not; that half stands. The second is that
the filtration Oseledets' theorem induces is the same shape of object a
persistence module decomposes into, so barcode tooling applies to it. **The second
half does not survive measurement on distilgpt2.**

Token cocycle, block 3, 46 steps, D = 768, `log(D) = 6.6438`. Bar count and
multiplicity entropy across the full monotone tolerance sweep:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       tol        cond   n_bars  max_mult  entropy  entropy/logD
  1.308e-05   grounded      763         2   6.6348        0.9986
  1.059e-05   shuffled      759         3   6.6269        0.9975
  1.308e-04   grounded      681         3   6.4793        0.9752
  4.137e-04   grounded      531         7   6.1414        0.9244
  1.308e-03   grounded      289        32   5.1567        0.7762
  4.137e-03   grounded      107       395   2.5358        0.3817
  1.308e-02   grounded       42       683   0.7172        0.1080
  1.308e-01   grounded        6       760   0.0723        0.0109
  1.308e+00   grounded        1       768  -0.0000       -0.0000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Three findings, all negative:

**1. At any tolerance fine enough to be honest, the filtration is D singletons.**
`entropy/logD = 0.9986` at `tol = 1.3e-05`. The encoding is the sorted spectrum
with extra steps and carries no additional information.

**2. There is no plateau.** The bar count slides smoothly 763 → 739 → 681 → 531 →
289 → 107 → 42 → 16 → 6 → 2 → 1 with no tolerance range where it holds steady. A
spectrum with genuine subspace structure would show a plateau, because a real
degeneracy survives a range of tolerances. Every tolerance here gives a different
answer, which means **the grouping is measuring the tolerance, not the operator.**

**3. Grounded and shuffled are indistinguishable at every tolerance.** 763 vs 759,
681 vs 674, 531 vs 532, 289 vs 279, 107 vs 113, 42 vs 31, 6 vs 7. At the median
adjacent gap both give 384 bars, with 148 against 152 of multiplicity > 1, and
`entropy/logD` of **0.8495 against 0.8505** — a difference of 0.001.

**What this means.** The Lyapunov spectrum of a transformer Jacobian cocycle is
*generic*: no repeated exponents, no degenerate Oseledets subspaces of dimension
above one. That is a substantive fact and not merely an absence — it says the
dynamics at this level carry no symmetry that would force an invariant subspace.
The persistence-barcode analogy is structurally suggestive and, here,
computationally empty.

**The salvage.** `tolerance_sweep` and `filtration_entropy` are a general test for
whether *any* claimed filtration structure is real, and the diagnostic is the
plateau rather than the value. A grouping that produces a different answer at
every tolerance is reporting its own parameter. That test is reusable against any
paper claiming discovered subspace or cluster structure from a thresholded
spectrum, and it costs one sweep.

**Limits.** One model, one width, one passage, one block, one seed, 46 cocycle
steps. A degeneracy could exist at larger width, in a model with tied or
structured weights, or over a longer trajectory. Nothing here rules that out; it
rules out distilgpt2 at D = 768.

## 6. Correctness of the instruments

30 tests, all against closed-form answers rather than self-consistency.

| assertion | tolerance |
|---|---|
| Jacobian of a position-wise linear block equals its weight matrix | 1e-10 |
| power iteration matches exact `svdvals` | 1e-6 |
| `tail_alpha` recovers synthetic exponents 0.25 / 0.5 / 1.0 / 2.0 | 1e-9 |
| `log_volume` equals `torch.linalg.slogdet` | 1e-8 |
| flat spectrum returns exponent 0 (negative control) | 1e-9 |
| diagonal cocycle returns `log a` | 1e-8 |
| orthogonal cocycle returns 0 (negative control) | 1e-8 |
| exponents sum to mean `log|det|` | 1e-8 |
| scaling every Jacobian by `c` shifts every exponent by `log c` | 1e-9 |
| rank-deficient step drives one exponent below −100, others finite | — |
