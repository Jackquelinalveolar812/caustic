<h1 align="center">Caustic — Where a Language Model Folds</h1>

<p align="center">
  <strong>A smooth map can be invertible at every single point and still send two different worlds to the same state.</strong><br>
  Where that happens, nothing downstream can recover which world it was — so the model must invent one.<br>
  The object measured is <code>J_l(t) = ∂h_{l+1}[t] / ∂h_l[t]</code>, the layer-to-layer transport map at one token.
</p>

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong> ·
  <a href="https://github.com/teerthsharma/caustic">github.com/teerthsharma/caustic</a> ·
  <em>teerths57@gmail.com</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&color=00aaff" alt="MIT"></a>
  <a href="#2-the-ledger"><img src="https://img.shields.io/badge/read%20this%20first-the%20ledger-00aaff?style=flat-square" alt="Ledger"></a>
  <a href="#10-negative-results"><img src="https://img.shields.io/badge/negatives-first--class%20section-red?style=flat-square" alt="Negatives"></a>
  <a href="#54-the-hypothesis-audit"><img src="https://img.shields.io/badge/Oseledets-hypotheses%20fail%2C%20stated-orange?style=flat-square" alt="Hypotheses"></a>
  <a href="#8-closed-form-propositions-the-test-suite-pins"><img src="https://img.shields.io/badge/propositions-proved%20in%20line-blueviolet?style=flat-square" alt="Propositions"></a>
  <a href="#14-validation"><img src="https://img.shields.io/badge/tests-71%20closed--form-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/core%20deps-numpy%20%2B%20torch-lightgrey?style=flat-square" alt="Deps"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-yellow?style=flat-square" alt="Python"></a>
</p>

---

## Abstract

**Caustic** proposes a geometric mechanism for confabulation in a language model and builds
the instruments to test it. The mechanism is *folding*: a map from contexts to internal
states that is a local diffeomorphism at every point it ever visits can still be globally
many-to-one, and at any state with two preimages the information distinguishing them is
gone. No downstream computation can recover a bit the state does not carry, so a model asked
to condition on that bit must produce something not determined by either context.

The first consequence is a claim about **where to look, and it corrects the obvious guess.**
In classical singularity theory a caustic is the image of the critical set, and
non-injectivity appears at a crease where `det J = 0`. The failure mode proposed here has
**no crease**. On the punctured plane $z \mapsto z^2$ has derivative $2z$, which never
vanishes, and is exactly two-to-one; $\exp : \mathbb{C} \to \mathbb{C}\setminus\{0\}$ has
derivative $e^z$, which never vanishes, and is infinitely many-to-one. Both are local
diffeomorphisms **everywhere** and neither is injective. A detector that watches for a
vanishing determinant, a collapsing smallest singular value or a spiking condition number is
therefore watching for the wrong event, and [section 1.4](#14-why-a-vanishing-determinant-is-the-wrong-place-to-look)
states how that framing claim dies.

The second consequence is that the ergodic-theoretic machinery this repository leans on does
not apply to the object it is usually pointed at, and this README says so before it reports a
number. A product of six *different* transformer blocks is not a cocycle — there is no single
map being iterated, so the cocycle identity never holds and Oseledets' theorem has nothing to
say about it. The token product has a better structural claim, and still no exhibited
invariant measure, no ergodicity and 46 steps rather than a limit. Every exponent reported
below is therefore called a **finite-time QR characteristic exponent**, never a Lyapunov
exponent. [Section 5.4](#54-the-hypothesis-audit) audits all four hypotheses in a table.

What has been measured is a set of preconditions, a cost, and three negatives. On a converged
46-step token product over distilgpt2, grounded text gives `lam1 = +0.1653` with
`sum = -226.74` over 768 directions and only `139/768` expanding — the signature of a
dissipative system on a low-dimensional attractor. Per-block Kaplan-Yorke dimension on a
separate run falls to `29.57` at block 1, a 26× compression against the width, then climbs
back to `674.67` by block 5. The exact Jacobian is **26.6× cheaper** than the Krylov estimator
of its own top eight singular values, which agree to `1.685e-04`. Three of four spectral
summaries flip sign across layers against a shuffled-token control. And the Oseledets-to-
persistence bridge, the organising analogy of the whole programme, is shown here to be
**structurally empty before it is measured**: every structure map in the growth filtration is
injective, so no bar ever dies, so the barcode *is* the multiplicity vector and carries
nothing else. The measured `entropy/logD = 0.9986` is what that argument predicts, not an
unlucky draw.

Nothing here measures non-injectivity. No collision search and no context-recoverability
probe exists in this repository, no detection benchmark has been run, and **no AUROC is quoted
anywhere on this page**. Until one runs, every number below is consistent with a perfectly
injective model.

**Keywords:** folding, caustic, singularity theory, Whitney fold and cusp, critical set,
global injectivity, local diffeomorphism, Jacobian, transformer interpretability,
multiplicative ergodic theorem, Oseledets filtration, linear cocycle, finite-time
characteristic exponent, Benettin QR algorithm, subadditive ergodic theorem, Kaplan-Yorke
dimension, Pesin identity, Takens delay embedding, attractor dimension, volume contraction,
dissipative dynamics, persistence module, interval decomposition, structure theorem, barcode,
bottleneck stability, singular spectrum, stable rank, power-law tail exponent, confabulation,
hallucination detection, out-of-distribution detection, reverse-mode automatic
differentiation, Krylov subspace methods

---

## Table of contents

| § | Section | What is in it |
|---|---|---|
| [1](#1-the-mechanism-drawn) | The mechanism, drawn | The folding picture, and why the crease is the wrong target |
| [2](#2-the-ledger) | **The ledger** | Every number, every control, one table |
| [3](#3-from-folding-to-confabulation) | From folding to confabulation | The argument, and what it forbids |
| [4](#4-cocycles-and-the-two-products-a-transformer-offers) | Cocycles, defined | Why only one of the two products is one |
| [5](#5-oseledets-multiplicative-ergodic-theorem) | Oseledets' theorem | Hypotheses first, then the audit that fails them |
| [6](#6-why-the-benettin-qr-algorithm-derived-not-asserted) | Benettin's QR, derived | Two propositions and a roundoff bound |
| [7](#7-persistence-modules-and-why-the-bridge-is-empty) | The persistence bridge | Why it is empty structurally, not just empirically |
| [8](#8-closed-form-propositions-the-test-suite-pins) | Propositions | 17 closed-form facts the suite pins, with proofs |
| [9](#9-measured-results) | Measured results | Cost, exponents, attractor dimension |
| [10](#10-negative-results) | **Negative results** | The layer sweep, the bridge, the discarded product |
| [11](#11-what-would-falsify-this) | **What would falsify this** | One named measurement per live claim |
| [12](#12-prior-art) | Prior art | Where the alternatives are better |
| [13](#13-implementation-map) | Implementation map | File-by-file responsibility |
| [14](#14-validation) | Validation | 71 closed-form assertions |
| [15](#15-quick-start) | Quick start | Reproduce every box above |
| [16](#16-requirements-and-measurement-environment) | Measurement environment | The one machine every number came from |
| [17](#17-limitations) | Limitations | Collected once, at the end |
| [18](#18-what-is-not-built-yet) | Not built yet | And what each unbuilt thing would settle |

### Provenance tags

Every numeric claim below is followed by a tag in square brackets naming the file it was read
from. An untagged sentence containing a number is a bug in this README.

| Tag | Source | Authoritative for |
|---|---|---|
| `[E§1]` … `[E§6]` | [`EVIDENCE.md`](EVIDENCE.md), numbered section, or its hardware header | Timings, separations, exponents, filtration sweep, tolerances, and the host |
| `[A]` | the attractor-dimension run, `python -m caustic.experiments.attractor_dimension`, on the host of [§16](#16-requirements-and-measurement-environment) | Every Kaplan-Yorke number. **This run postdates `EVIDENCE.md` and is not yet recorded there** — a known drift, stated rather than hidden |
| `[code]` | the named module under [`caustic/`](caustic) or [`tests/`](tests) | Constants and conventions: floors, windows, sign rules, test tolerances |
| `[cfg]` | [`pyproject.toml`](pyproject.toml) | Dependency floors and Python version |
| `[plan]` | [`PLAN.md`](PLAN.md) | What is scheduled, and what is ruled out before being scheduled |
| *(none)* | this page | A definition, a theorem statement, or a proof — checkable by reading, not by running |

**What is deliberately absent.** No AUROC, anywhere: no detection benchmark has been run in
this repository, and importing a score from elsewhere to stand in for one measured here is the
exact failure this evidence standard exists to prevent. No projection past the measured range:
every number came from the single host in [§16](#16-requirements-and-measurement-environment),
on one model at one width. No CI badge: there is no continuous integration here and there is
not meant to be, because correctness is argued from closed-form assertions
([§8](#8-closed-form-propositions-the-test-suite-pins)), which a green check cannot supply.

---

## 1. The mechanism, drawn

The whole hypothesis fits on a napkin. This section draws it before any equation appears,
because a reader who cannot picture the folding cannot judge whether the measurements in
[§9](#9-measured-results) are evidence for it.

### 1.1 A sheet with a crease

Take a sheet of paper, hold it edge-on, and bend it into a U. Now shine a light straight down
and look at the shadow.

```
        the sheet, seen edge-on                    the shadow, seen from below
   ───────────────────────────────            ────────────────────────────────

     c1 ●───────────────╮
                         ╲
                          ╲   <- the crease: the tangent
                          ╱      plane turns vertical here          ●
                         ╱                                          h*
     c2 ●───────────────╯

   two distinct points of the sheet                one point of the shadow
```

Every point of the sheet has a well-defined tangent plane. The shadow map is smooth. Yet `c1`
and `c2`, arbitrarily far apart on the sheet, land on the same shadow point `h*`. Given only
`h*`, the question "which point of the sheet was it?" has no answer. The information was not
degraded, corrupted, or noisily encoded — it was **destroyed by the geometry of the map**, and
no amount of downstream cleverness restores it.

The set of points where the sheet turns over is the **critical set**. Its image — the bright
curve you see at the bottom of a cup of coffee, or on a swimming-pool floor — is the
**caustic**. That is what this repository is named for. The name is a hypothesis about where
to look, not a claim that the object has been found in a language model.

### 1.2 Whitney, and what the name buys

Whitney classified the stable singularities of smooth maps of the plane to the plane:
generically there are exactly two, the **fold** and the **cusp**. In local coordinates,

```
   fold      (x, y)  ↦  (x,  y²)                 det J = 2y      critical on y = 0
   cusp      (x, y)  ↦  (x,  y³ − xy)            det J = 3y² − x  critical on x = 3y²
```

The fold is two-to-one on one side of its critical line and empty on the other. The cusp is
three-to-one inside its characteristic wedge. Both are stable: perturbing the map slightly
moves the crease but does not remove it. This is the classical picture, and it is the source
of the name. It is **not** the failure mode this repository is hunting.

### 1.3 The fold with no crease, the case that actually matters

Wrap a strip onto a circle.

```
   input strip, coordinate θ                    image, the circle
   ──────────────────────────                   ─────────────────

   0 ─────────── π ─────────── 2π                     ╭───●───╮
   ●                           ●          ──▶        │    h*   │
   c1                          c2                     ╰────────╯

   θ ↦ (cos θ, sin θ)   has derivative of norm 1 at every θ,
                        is a local diffeomorphism everywhere,
                        and still glues θ = 0 to θ = 2π.
```

There is no crease anywhere. The derivative never vanishes, is never ill-conditioned, and
every neighbourhood maps bijectively onto its image. The map is nonetheless two-to-one on the
closed strip. The same thing happens for two maps that are as well behaved as maps get:

- $z \mapsto z^2$ on $\mathbb{C}^{*} = \mathbb{C}\setminus\{0\}$ has derivative $2z$, which is
  nonzero at every point of the domain, and is **exactly two-to-one**.
- $\exp : \mathbb{C} \to \mathbb{C}\setminus\{0\}$ has derivative $e^{z}$, which is nonzero
  everywhere, and is **infinitely many-to-one**: $\exp(z) = \exp(z + 2\pi i k)$ for every
  integer $k$.

Both are local diffeomorphisms at every point of their domain. Neither is injective. The
inverse function theorem gives a local inverse near each point and says nothing whatever about
whether those local inverses patch together into a global one.

This is the geometry to keep. Two points of the input are far apart; every local patch around
each of them is a clean bijection; the two paths nevertheless arrive at the same place. Drawn
in the shape that matters here:

```
   grounded context A ──────╮
                             ╲
                              ╲
                               ●  h*  ──▶  every later layer  ──▶  one continuation
                              ╱
                             ╱
   grounded context B ──────╯

   det J(A) ≠ 0.  det J(B) ≠ 0.  Both well conditioned.  The bit is gone anyway.
```

### 1.4 Why a vanishing determinant is the wrong place to look

A detector built on a vanishing determinant, a collapsing smallest singular value, or a
spiking condition number is watching for **the crease of [§1.2](#12-whitney-and-what-the-name-buys)**.
The mechanism of [§1.3](#13-the-fold-with-no-crease-the-case-that-actually-matters) has no
crease to find. Watching `det J` for it is like listening for a car crash to detect two people
who arrived at the same address by different roads.

More precisely, the three quantities separate as follows:

| Watching this | Detects | Misses |
|---|---|---|
| `det J -> 0`, `sigma_min -> 0`, condition number | the classical critical set: creases, cusps, rank drops | every non-injectivity with no critical point — §1.3 in full |
| `sum_i lambda_i << 0`, volume contraction | the **precondition** for folding: the image occupies less volume than the source | nothing about whether preimages actually merged; contraction is compatible with injectivity |
| a preimage collision `Φ(c1) = Φ(c2)` with `c1 ≠ c2` | the mechanism itself | — this is the measurement, and it is the one that has not been run |

Only the third row tests the hypothesis. The first two rows are what the field, and this
repository so far, actually measure.

**This framing claim is itself falsifiable, and it is worth stating how it dies.** If a
collision search ([§11](#11-what-would-falsify-this)) finds that near-collisions occur *only*
where the Jacobian is ill-conditioned, then the classical crease was the right place to look,
the argument of this section is wrong, and the determinant-watchers were right the whole time.
That outcome would be reported as the result, not as a caveat.

### 1.5 What the Jacobian Conjecture does not give

Stated early and in full, because it is the most likely misreading of the mathematics above.

**The Jacobian Conjecture does not transfer** `[plan]`. It concerns polynomial maps
$\mathbb{C}^n \to \mathbb{C}^n$ whose Jacobian determinant is a **nonzero constant**, and asks
whether such a map must be invertible. Its hypotheses fail here, separately and fatally:

| Hypothesis of the conjecture | Status for a transformer |
|---|---|
| the map is polynomial | false — softmax, LayerNorm and GELU are not polynomial in the hidden state |
| `det J` is a nonzero constant | false — the layer sweep measured `logdet_abs` varying block to block and condition to condition `[E§3]` |
| the domain and codomain are $\mathbb{C}^n$ | the map here is real, and the object of interest is $\mathbb{R}^D \to \mathbb{R}^D$ at one token |

Any argument of the form "the conjecture says full rank implies invertible, therefore the
network is injective" is invalid on all three counts. **No claim in this repository rests on
that theorem, and no task in [`PLAN.md`](PLAN.md) may cite it as support.** What transfers is
a moral, and only a moral:

> Local invertibility everywhere does not imply global injectivity.

**Historical note, because it is what put the question in view.** The Jacobian Conjecture is
no longer open. In July 2026 Levent Alpöge, working with Claude Fable 5, produced an explicit
polynomial counterexample refuting it in dimension 3, which extends to every `n >= 3` by
adjoining identity coordinates; the plane case `n = 2` remains open. The refutation is what
made "everywhere non-degenerate, still not injective" a live question rather than a
technicality — a map can be non-degenerate at every point of its domain and still fail to be
one-to-one, and now there is a polynomial witness to it.

This is context, not support. The counterexample is a fact about polynomial maps on
$\mathbb{C}^n$, the hypothesis table above still rules the theorem out here on all three
counts, and nothing below depends on it. It is recorded because a reader who knows the result
should know it was not overlooked, and a reader who does not should not have to guess why the
moral is worth stating at all.

That moral is a statement about maps, not a theorem being borrowed, and
[§1.3](#13-the-fold-with-no-crease-the-case-that-actually-matters) already exhibits it with a
circle, a square and an exponential. A model can have full-rank Jacobians at every point it
ever visits and still fold two distinct grounded contexts onto one internal state. Full rank
is a local certificate; injectivity is a global property; the gap between them is where a
model can lose the information that distinguished two situations.

---

## 2. The ledger

Everything measured in this repository, in one table, before any argument for why it should be
interesting. **A number and its control are in the same row.** A row whose control column says
*none* is not evidence and is labelled as such.

Verdicts: **DISCARDED** — measured, failed its own convergence check, not used.
**NEGATIVE** — measured against its control and did not separate.
**STRUCTURAL** — measured, did not separate, and the non-separation is the usable part.
**SEPARATED** — measured, separated with consistent sign, `n = 1`.
**WEAK** — separated in direction but not in magnitude.
**MEASURED** — a cost or accuracy fact with no hypothesis attached.
**NOT RUN** — named here so its absence cannot be mistaken for a result.

| # | Quantity | Measured | Compared against | Control value | Verdict |
|---|---|---|---|---|---|
| 1 | block-product `lam1`, grounded, 6 steps `[E§4]` | `+0.0224` | its own last-step drift | drift `0.0255` — **larger than the value** | **DISCARDED** |
| 2 | `sigma_max` separation across 6 blocks `[E§3]` | \|mean\| `0.0085` | shuffled-token, matched multiset | sign flips layer to layer | **NEGATIVE** |
| 3 | `tail_alpha` separation across 6 blocks `[E§3]` | \|mean\| `0.0217` | shuffled-token, matched multiset | sign flips layer to layer | **NEGATIVE** |
| 4 | `logdet_abs` separation across 6 blocks `[E§3]` | \|mean\| `0.0548` | shuffled-token, matched multiset | sign flips layer to layer | **NEGATIVE** |
| 5 | `stable_rank` separation across 6 blocks `[E§3]` | \|mean\| `0.1138`, median `+17.5%` | shuffled-token, matched multiset | holds sign at 5 of 6 layers; block 4 reverses it | **NEGATIVE** |
| 6 | `tail_alpha` level, blocks 1–5 `[E§3]` | grounded `0.2374 ± 0.0293`, cv `0.124` | shuffled `0.2342 ± 0.0481`, cv `0.205` | difference `0.0033` = **1.4%** of grounded | **STRUCTURAL** |
| 7 | token-product `lam1`, 46 steps `[E§4]` | grounded `+0.1653`, drift `0.0012` (0.7%) | shuffled `+0.1852`, drift `0.0003` (0.2%) | separation `−0.1202` | **SEPARATED** |
| 8 | token-product `sum(lam)` `[E§4]` | grounded `−226.74` | shuffled `−170.37` | separation `−0.2486` — the largest converged one | **SEPARATED** |
| 9 | token-product positive exponents `[E§4]` | grounded `139/768` | shuffled `151/768` | separation `−0.0863` | **SEPARATED** |
| 10 | filtration `entropy/logD` at `tol = 1.308e-05` `[E§5]` | grounded `0.9986`, 763 bars | shuffled, same sweep | `0.9975`, 759 bars | **NEGATIVE** |
| 11 | filtration bar count at the median adjacent gap `[E§5]` | grounded 384 bars, `entropy/logD` `0.8495` | shuffled, same tolerance | 384 bars, `0.8505` — difference `0.001` | **NEGATIVE** |
| 12 | Kaplan-Yorke `D_KY` per block, token product `[A]` | `768.00 / 29.57 / 225.76 / 298.08 / 482.20 / 674.67`, cv `0.6801` | shuffled-token, matched multiset | 5 of 6 same sign, \|mean\| `0.0758` | **WEAK** |
| 13 | full `768×768` `jacrev` `[E§1]` | `53.694 ms`, `91.3×` forward | top-8 power iteration, 20 iters | `1428.938 ms`, `2429.5×` forward — exact is **26.6× cheaper** | **MEASURED** |
| 14 | top-8 power iteration accuracy `[E§1]` | max rel. error `1.685e-04` | exact `svdvals`, same input, same host | estimator is correct, and still loses | **MEASURED** |
| 15 | AUROC against factual-error labels | — | Mahalanobis, PCA, max-softmax | **not run — no J-space AUROC exists anywhere in this repository** | **NOT RUN** |

**The three sentences a skeptical reader should leave with.** Nine of the fifteen rows are a
failure, a discard, a weak signal, or an absence. The one arm that would settle whether any of
this detects factual error — row 15 — has not been run, so nothing here is a
hallucination-detection result. Mahalanobis and PCA on a single hidden vector are cheaper than
anything in this repository by orders of magnitude, and the working assumption until measured
otherwise is that they win.

---

## 3. From folding to confabulation

### 3.1 The statement

Let $\mathcal{C}$ be the set of contexts a model might be given and let

**(1)**

$$
\Phi_{\ell} : \mathcal{C} \to \mathbb{R}^{D},
\qquad
\Phi_{\ell}(c) \;=\; h_{\ell}[T] \;\;\text{under context}\; c
$$

be the map carrying a context to the hidden state at layer $\ell$ and final position $T$. The
hypothesis is that $\Phi_{\ell}$ is not injective:

**(2)**

$$
\exists \; c_1 \neq c_2 \in \mathcal{C}
\quad\text{with}\quad
\Phi_{\ell}(c_1) \;=\; \Phi_{\ell}(c_2) \;=\; h^{\star}
$$

Everything the model computes after layer $\ell$ is some function $g$ of the state. So for
every such $g$, without exception:

**(3)**

$$
g\big(\Phi_{\ell}(c_1)\big) \;=\; g\big(\Phi_{\ell}(c_2)\big)
$$

and if a prior on $\mathcal{C}$ puts mass on both, the conditional entropy of the context given
the state is strictly positive:

**(4)**

$$
\big\lvert \Phi_{\ell}^{-1}(h^{\star}) \big\rvert > 1
\qquad\Longrightarrow\qquad
H\big(c \;\big|\; h^{\star}\big) \;>\; 0
$$

Read (3) slowly, because it is the entire argument. It is not a claim about capacity, training
data, decoding temperature, or attention. It says that **if two grounded contexts land on one
state, then no continuation — not a better head, not a larger model above that layer, not a
longer chain of thought — can be conditioned on which context occurred.** The model is asked a
question whose answer is not present in what it holds. It answers anyway. That is confabulation
as a geometric consequence rather than a behavioural description.

### 3.2 What the mechanism forbids

A mechanism that forbids nothing is decoration. This one forbids the following, and each line
is a way to kill it:

1. **Collisions must exist.** If an exhaustive-enough search over $\mathcal{C}$ finds no pair
   of semantically distinct contexts whose states are closer than the states of paraphrase
   pairs, the mechanism is not operating at that layer and depth.
2. **Collisions must be recoverability-limited.** If a decoder trained on `h` recovers the
   distinguishing content of the context essentially perfectly, there was no fold to find. The
   control is a decoder of identical capacity trained on shuffled labels, and a second trained
   on an earlier layer where less mixing has occurred.
3. **Collisions must predict error.** If collision density does not correlate with factual
   error at matched cost against Mahalanobis, PCA and max-softmax, then folding may be real and
   simply irrelevant to hallucination — a true mechanism for the wrong phenomenon.
4. **Collisions must not require a crease.** If every near-collision sits where the Jacobian is
   ill-conditioned, [§1.4](#14-why-a-vanishing-determinant-is-the-wrong-place-to-look) is wrong.

Failing any one of these is a publishable negative and is planned as such `[plan]`.

### 3.3 What the mechanism does not claim

- It does not claim folding is the only cause of hallucination. Sampling noise, missing
  training data and reward hacking are separate and unaddressed here.
- It does not claim the fold is at any particular layer. $\ell$ is a free parameter of (1).
- It does not claim any of this has been observed. **Nothing in this repository measures
  non-injectivity.** [§9](#9-measured-results) measures a precondition and a cost; the
  measurement that would test (2) is described in [§11](#11-what-would-falsify-this) and has
  not been run.

**Why the Jacobian and not a point cloud.** The argument is definitional, not empirical, which
is why it is stated here rather than supported by a measurement. A point cloud of hidden states
over a token window is a record of **states the model has already occupied**; any statistic of
it, persistent or otherwise, is a function of the past trajectory, and whether such a statistic
predicts anything forward is a contingent empirical question with no structural guarantee
behind it. `J_l(t)` is the opposite kind of object: it is, by the definition in
[§4.1](#41-the-definition), the linear operator that carries a perturbation of `h_l[t]` forward
into `h_{l+1}[t]`. Forward causality is not a hypothesis about it; it is what the symbol means.
That is the entire reason for working in Jacobian space, and it is the sentence to keep if only
one survives.

---

## 4. Cocycles, and the two products a transformer offers

The word "cocycle" does real work in [§5](#5-oseledets-multiplicative-ergodic-theorem), so it
gets a definition rather than a gesture.

### 4.1 The definition

The object under study is the diagonal block of the layer-to-layer Jacobian at one token:

**(5)**

$$
J_l(t) \;=\; \frac{\partial h_{l+1}[t]}{\partial h_l[t]} \;\in\; \mathbb{R}^{D \times D}
$$

`h_l ∈ ℝ^{1×T×D}` is the observed hidden state entering block `l`; `t` is one token position;
`D` is the model width, 768 for distilgpt2 `[E§ header]`. Every other token position is held at
its observed value, so (5) is the **diagonal block** of the full `(TD × TD)` Jacobian. The block
remains causal: position `t` still attends over the frozen prefix. Implemented by `block_map`
and differentiated by batched reverse-mode AD in `exact_jacobian` `[code: caustic/jacobian.py]`.

Now the ergodic-theoretic definition. Let $(X, \mathcal{B}, \mu)$ be a probability space and let
$T : X \to X$ be measurable and measure-preserving, meaning $\mu(T^{-1}B) = \mu(B)$ for every
$B \in \mathcal{B}$. A measurable map $A : X \to \mathbb{R}^{D \times D}$ generates the **linear
cocycle over $T$**

**(6)**

$$
A^{(n)}(x) \;=\; A(T^{n-1}x)\,A(T^{n-2}x)\cdots A(Tx)\,A(x), \qquad A^{(0)}(x) = I
$$

The defining property, and the reason for the name, is the **cocycle identity**

**(7)**

$$
A^{(m+n)}(x) \;=\; A^{(m)}(T^{n}x)\; A^{(n)}(x) \qquad \text{for all } m, n \geq 0
$$

Read plainly: a cocycle is *one* matrix-valued function $A$, sampled along *one* orbit of *one*
measure-preserving map. Identity (7) is what lets $n$-step behaviour be decomposed at any
intermediate time, and it is the structural hypothesis every ergodic-theoretic conclusion below
is built on. **A product of $n$ arbitrary, unrelated matrices satisfies no such identity and is
not a cocycle.**

### 4.2 The two products, and why only one is a cocycle

Composing (5) along a trajectory gives the linear model of the whole trajectory:

**(8)**

$$
J^{(n)} \;=\; J_n\, J_{n-1} \cdots J_1
$$

Two distinct products exist in a transformer and they are **not** interchangeable.

| Product | Steps | The map being iterated | Is it a cocycle in the sense of §4.1? |
|---|---|---|---|
| **Token product**, fixed block `l`, positions `t = 1 … n` | as many as the sequence is long — 46 in the reported run `[E§4]` | the same block, the same parameters, at successive positions | Structurally closest. One function $A(\cdot) = J_l(\cdot)$ sampled along "advance one position". It satisfies (7). Whether the base map preserves a measure is a separate question, answered in §5.4 |
| **Block product**, fixed token `t`, blocks `l = 0 … L−1` | 6 on distilgpt2 `[E§4]` | a *different* block, with *different* parameters, at each step | **No.** There is no single $A$ and no base map to iterate. Six different transformer blocks composed is a product of six different matrices; the identity (7) never holds, and there is no multiplicative ergodic theorem for it to be an approximation *of* |

This is not a fine distinction and it is not a caveat about convergence. The block product fails
the definition, not the asymptotics. **Oseledets' theorem does not apply to it at all**, and no
amount of depth would repair that: a 96-block model would give a product of 96 different
matrices, which is still not a cocycle. The quantity computed from it is a **finite-time QR
characteristic exponent of a finite matrix product** and is called that everywhere below.

Both products are run in `caustic/experiments/lyapunov_llm.py`; they behave differently, and
[§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model) reports why only one of them
produced a usable number. The table above is the structural reason to expect that in advance.

---

## 5. Oseledets' Multiplicative Ergodic Theorem

Stated with hypotheses first. The conclusion is what everyone quotes; the hypotheses are what
decide whether the quotation is legitimate here.

### 5.1 Hypotheses

- **(H1) A measure-preserving dynamical system.** $(X, \mathcal{B}, \mu)$ is a probability space
  and $T : X \to X$ is measurable with $\mu \circ T^{-1} = \mu$.
- **(H2) Measurability of the generator.** $A : X \to \mathbb{R}^{D \times D}$ is measurable.
- **(H3) Integrability.** The positive part of the log norm is integrable:

**(9)**

$$
\int_X \log^{+} \lVert A(x) \rVert \, d\mu(x) \;<\; \infty,
\qquad \log^{+} s \;=\; \max(0, \log s)
$$

- **(H4) Invertibility, for the two-sided theorem only.** $T$ is invertible, $A(x)$ is
  invertible for $\mu$-almost every $x$, and $\log^{+}\lVert A(x)^{-1} \rVert$ is also
  $\mu$-integrable. Without (H4) one still gets the decreasing filtration of §5.2; (H4) is what
  upgrades the filtration to a splitting into Oseledets subspaces.

Note what (H3) does and does not ask. Only the *positive* part must be integrable, so directions
that collapse are allowed and produce exponents equal to $-\infty$; this is exactly the folding
case, and the implementation supports it rather than erroring
([Proposition 6](#8-closed-form-propositions-the-test-suite-pins)).

### 5.2 Conclusion

Under (H1)–(H3) there is a $T$-invariant set $X_0$ with $\mu(X_0) = 1$ such that for every
$x \in X_0$ the limit

**(10)**

$$
\Lambda(x) \;=\; \lim_{n \to \infty} \big( A^{(n)}(x)^{\top} A^{(n)}(x) \big)^{1/(2n)}
$$

exists and is positive semi-definite. Write its distinct eigenvalues as
$e^{\lambda_1(x)} > \cdots > e^{\lambda_k(x)}$ with multiplicities $m_1(x), \dots, m_k(x)$
summing to $D$, where each $\lambda_i(x) \in [-\infty, \infty)$. Then there is a **decreasing
filtration**

**(11)**

$$
\mathbb{R}^D = V_1(x) \supsetneq V_2(x) \supsetneq \cdots \supsetneq V_k(x) \supsetneq V_{k+1}(x) = \{0\}
$$

for which the exponential growth rate is exactly determined by which step of the filtration a
vector first leaves:

**(12)**

$$
\lim_{n \to \infty} \frac{1}{n} \log \lVert A^{(n)}(x)\, v \rVert \;=\; \lambda_i(x)
\qquad \text{for every } v \in V_i(x) \setminus V_{i+1}(x)
$$

with $\dim V_i(x) - \dim V_{i+1}(x) = m_i(x)$. The filtration is **equivariant** along the base
dynamics:

**(13)**

$$
A(x)\, V_i(x) \;\subseteq\; V_i(Tx)
$$

If in addition $T$ is **ergodic**, then $k$, the $\lambda_i$ and the $m_i$ are $\mu$-almost
everywhere constant — which is what turns them from functions of $x$ into *numbers*. Without
ergodicity, "the Lyapunov spectrum" is not well defined as a list of scalars.

Three things are worth naming explicitly, because each is quietly assumed whenever the theorem
is invoked casually:

1. The conclusion is a **limit as $n \to \infty$**. Nothing in (10)–(12) says anything about a
   finite $n$.
2. The conclusion holds only **almost everywhere**. A single trajectory is a measure-zero set,
   so no statement about one trajectory follows from the theorem alone.
3. The filtration (11) is **$x$-dependent**. It is a measurable bundle over the base, tied
   together by (13), not one fixed flag in $\mathbb{R}^D$.

### 5.3 Two corollaries used later

**Corollary A, volume.** If additionally $\log\lvert\det A\rvert \in L^1(\mu)$, then for
$\mu$-a.e. $x$

**(14)**

$$
\lim_{n \to \infty} \frac{1}{n} \log \lvert \det A^{(n)}(x) \rvert \;=\; \sum_{i=1}^{k} m_i(x)\, \lambda_i(x)
$$

*Proof sketch.* $\lvert\det A^{(n)}\rvert$ is the product of all $D$ singular values of
$A^{(n)}$, and by (10) the $j$-th singular value grows like $e^{n\lambda_{i(j)}}$. Take logs and
divide by $n$. The finite-$n$ version of this identity is exact and is
[Proposition 3](#8-closed-form-propositions-the-test-suite-pins). ∎

This is the corollary that connects contraction to folding, and its **direction** matters. A map
sending two separated regions onto one region must lose the volume of one of them, so folding
**implies** contraction. The converse is false, and the counterexample is one line: `h -> h/2`
has $\sum_i \lambda_i = D \log(1/2) < 0$, contracts every volume in the space, and is injective
everywhere. A uniform contraction shrinks the sheet without ever bending it. So:

**(15)**

$$
\text{folding} \;\Longrightarrow\; \sum_i \lambda_i < 0,
\qquad\qquad
\sum_i \lambda_i < 0 \;\;\text{does not imply}\;\; \text{folding}
$$

A strongly negative exponent sum is **consistent with** the mechanism and **is not evidence of**
it. It rules out the one world in which the mechanism is impossible — a volume-preserving or
volume-expanding map cannot fold — and rules out nothing else. Every number in
[§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model) should be read against (15)
and not past it.

**Corollary B, the top exponent is upper semicontinuous and only that.** Set
$f_n(A) = \int_X \log \lVert A^{(n)}(x) \rVert \, d\mu(x)$. Measure preservation gives
$f_{m+n}(A) \le f_m(A) + f_n(A)$, so by Fekete's subadditivity lemma — the finite-form core of
Kingman's subadditive ergodic theorem —

**(16)**

$$
\lambda_1 \;=\; \lim_{n \to \infty} \frac{1}{n} f_n(A) \;=\; \inf_{n \geq 1} \frac{1}{n} f_n(A)
$$

Each $A \mapsto \frac{1}{n} f_n(A)$ is continuous in the uniform topology on bounded cocycles,
and **an infimum of continuous functions is upper semicontinuous**. Hence $\lambda_1$ is upper
semicontinuous in the cocycle. ∎

Corollary B is stated because of what it does *not* say. Upper semicontinuity is strictly weaker
than continuity, and continuity genuinely fails: Bochi's theorem produces $C^0$-arbitrarily
small perturbations of area-preserving surface systems whose exponents collapse to zero. **The
Lyapunov spectrum is not a stable invariant of its input.**
[Section 7.4](#74-why-the-bridge-is-structurally-empty) uses this to kill one half of the
persistence analogy, where the corresponding object *is* 1-Lipschitz stable.

### 5.4 The hypothesis audit

This is the load-bearing honesty of the whole document, so it is a table rather than a
paragraph, and it appears before any exponent is printed.

| Hypothesis | What it would have to be here | Status for the **token** product | Status for the **block** product |
|---|---|---|---|
| **(H1)** base is measure-preserving | $T$ = advance one token position, on a space of hidden states carrying a $T$-invariant probability measure | **FAILS — not established.** One prompt gives one finite orbit. The state distribution at position 1 is not the distribution at position 46; learned positional structure alone breaks stationarity, and no invariant $\mu$ has been exhibited | **FAILS structurally.** $T$ = "next block" iterates a *different* map at each step, so there is no single $A$ over a single base at all, and no cocycle identity |
| **(H2)** $A$ measurable | $x \mapsto J_l(x)$ | **HOLDS.** A composition of smooth operations is Borel measurable. This is the only hypothesis claimed outright | Same, but moot: without a base map there is nothing for it to be measurable over |
| **(H3)** $\log^{+}\lVert A \rVert \in L^1$ | finite mean of $\log^{+}$ of the Jacobian norms | **HOLDS vacuously.** A finite orbit of finite matrices satisfies (9) trivially, so satisfying it carries no information. Integrability is not the binding constraint here | Same |
| **(H4)** invertibility | $\det J \neq 0$ along the orbit, with integrable inverse log-norm | **UNVERIFIED.** Not measured. `logdet_abs` was measured to vary block to block `[E§3]`, which bears on the value, not on non-vanishing | Same |
| **ergodicity** | needed for the $\lambda_i$ to be numbers rather than functions of $x$ | **FAILS — unavailable.** No invariant measure has been exhibited, so ergodicity of it cannot even be posed | Same |
| **$n \to \infty$** | the limits (10), (12), (14) | **FAILS — 46 steps** `[E§4]` | **FAILS — 6 steps** `[E§4]` |

**The conclusion this repository draws from that table, stated as plainly as it can be:**

> What [§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model) and
> [§9.3](#93-attractor-dimension-per-block) report are **finite-time QR characteristic exponents
> of finite matrix products**, not Lyapunov exponents in the sense of equation (12). For the
> token product, Oseledets' theorem is the reason such a quantity is expected to mean something
> in a stationary setting; it is not a theorem about this computation and is not cited as one.
> For the block product it is not even that: the object is not a cocycle, so the theorem is not
> an idealisation of what was computed, it is about a different kind of object entirely.

This is a real gap and it is not repairable by better numerics. Two things would narrow it, and
neither is built:

1. **Supply a base and a measure.** Take the base to be the shift on a *stationary text process*
   rather than one prompt, and $A$ the Jacobian along it, then average over prompts. That gives
   (H1) an honest candidate, and it makes the exponents a property of the model-plus-corpus
   rather than of one passage. It is not implemented `[plan]`.
2. **Report convergence, always.** `finite_time_spectrum` returns the running exponents after
   every step, shape `(n, D)` `[code: caustic/cocycle.py]`, so a report can show whether a value
   had settled. That is the mechanism that discarded the 6-step number in
   [§10.3](#103-the-6-step-block-product-discarded-and-why), and it is the only defence available
   while gap (1) is open.

---

## 6. Why the Benettin QR algorithm, derived not asserted

The usual sentence is "you cannot multiply the Jacobians directly, so re-orthonormalize". Below
is why, in two propositions and three corollaries, each checkable by reading.

### 6.1 The naive product collapses onto one direction

**Proposition 1, collapse.** Let $M_n = J_n \cdots J_1$ have singular values
$\sigma_1(n) \geq \cdots \geq \sigma_D(n)$, left singular vectors $u_j(n)$ and right singular
vectors $v_j(n)$. Let $w$ be any unit vector with $c_1 = \langle w, v_1(n) \rangle \neq 0$. Then

**(17)**

$$
\sin \angle \big( M_n w,\; u_1(n) \big) \;\leq\; \frac{\sigma_2(n)}{\sigma_1(n)} \cdot \frac{1}{\lvert c_1 \rvert}
$$

*Proof.* Write $w = \sum_j c_j v_j(n)$, so $M_n w = \sum_j \sigma_j(n) c_j u_j(n)$. The component
along $u_1$ has norm $\sigma_1 \lvert c_1 \rvert$ and the orthogonal complement has norm
$(\sum_{j \geq 2} \sigma_j^2 c_j^2)^{1/2} \leq \sigma_2 \lVert w \rVert = \sigma_2$. The tangent
of the angle is the ratio of the two, and $\sin \leq \tan$. ∎

**Corollary, why this is fatal in floating point.** If growth rates exist then
$\sigma_j(n) \approx e^{n \lambda_j}$, so the bound (17) decays like
$e^{-n(\lambda_1 - \lambda_2)}$. Two columns of a running product are *numerically
indistinguishable* once their separation falls below unit roundoff
$u = 2^{-53} \approx 1.1 \times 10^{-16}$ in float64, that is once

**(18)**

$$
n \;\gtrsim\; \frac{53 \ln 2}{\lambda_1 - \lambda_2} \;\approx\; \frac{36.7}{\lambda_1 - \lambda_2}
$$

Past that point every column of $M_n$ is a rounding of the same direction, the computed
$\sigma_2, \dots, \sigma_D$ are noise, and **all but the leading exponent are lost**. The binding
ratio across the whole spectrum is
$\sigma_D(n)/\sigma_1(n) \approx e^{-n(\lambda_1 - \lambda_D)}$, which is far smaller still.

Dynamic range fails at the same time and for the same reason. In float64 the largest finite value
is about $1.8 \times 10^{308}$, i.e. $\log \approx 709.8$, and the smallest normal is its
reciprocal. The measured token product has exponent sum $-226.74$ across $768$ directions
`[E§4]`. Arithmetic on those two sourced numbers, shown so it can be checked:
$-226.74 / 768 = -0.295$ is the **mean** per-step exponent. A direction contracting at the mean
rate underflows a float64 by $n \approx 709.8 / 0.295 \approx 2400$ steps, and the most
contracting directions — the ones that make the sum as negative as it is — underflow far sooner.
A naive product therefore loses the bottom of the spectrum to underflow while it is losing the
middle to collapse.

### 6.2 What QR accumulation actually computes

Benettin's fix is to carry an orthonormal frame and re-orthonormalize every step. Let $Q_0$ be
orthogonal and define, for $k = 1, \dots, n$,

**(19)**

$$
Z_k = J_k\, Q_{k-1}, \qquad Q_k R_k = Z_k, \qquad (R_k)_{ii} > 0
$$

The positivity of the diagonal is not automatic — QR is unique only up to the sign of each
column, and LAPACK's sign choice is not stable across inputs — so `_qr_step` folds the sign into
$Q$ `[code: caustic/cocycle.py]`. With that convention the factorization is unique for
nonsingular $Z_k$, which is what makes the next proposition an identity rather than an
approximation.

**Proposition 2, QR partial sums are exact log $p$-volumes.** With (19),

**(20)**

$$
M_n Q_0 \;=\; Q_n\, R_n R_{n-1} \cdots R_1
$$

and consequently, writing $q_1, \dots, q_D$ for the columns of $Q_0$ and $\mathrm{vol}_p$ for the
$p$-dimensional parallelepiped volume, for every $p \leq D$

**(21)**

$$
\sum_{i=1}^{p} \frac{1}{n} \sum_{k=1}^{n} \log (R_k)_{ii}
\;=\;
\frac{1}{n} \log \mathrm{vol}_p \big( M_n q_1, \dots, M_n q_p \big)
$$

*Proof.* For (20), induct: $J_1 Q_0 = Q_1 R_1$, and if $M_{k-1} Q_0 = Q_{k-1} R_{k-1} \cdots R_1$
then $M_k Q_0 = J_k Q_{k-1} R_{k-1} \cdots R_1 = Q_k R_k R_{k-1} \cdots R_1$. For (21), the
product $R_n \cdots R_1$ is upper triangular with diagonal entries $\prod_k (R_k)_{ii}$. Its first
$p$ columns have their last $D - p$ entries zero, so the first $p$ columns of $M_n Q_0$ are $Q_n$
applied to $p$ vectors whose Gram determinant is the square of
$\prod_{i \leq p} \prod_k (R_k)_{ii}$; $Q_n$ is an isometry and preserves that volume. Take logs
and divide by $n$. ∎

**Corollary, all $D$ exponents are recovered.** Define
$\lambda_i^{(n)} = \frac{1}{n}\sum_k \log (R_k)_{ii}$, which is what `lyapunov_spectrum` returns
`[code: caustic/cocycle.py]`. By (21), $\lambda_1^{(n)} + \cdots + \lambda_p^{(n)}$ is the growth
rate of the $p$-volume of the image of the leading $p$-frame. Subtracting the $p-1$ case from the
$p$ case isolates $\lambda_p^{(n)}$. The naive product gives access only to $p = 1$, the norm,
and through the determinant $p = D$; QR gives every $p$ in between, which is precisely the
$D - 2$ exponents the naive method throws away.

**Corollary, no overflow.** Every quantity in (19) is $O(\lVert J_k \rVert)$, because the frame is
orthonormal at the start of each step. The dynamic range is per-step rather than cumulative, so
neither failure of §6.1 can occur, however long the trajectory runs.

### 6.3 The frame dependence that survives at finite n

Proposition 2 also states, honestly, the price. For finite $n$ the value of $\lambda_i^{(n)}$
depends on $Q_0$ — equation (21) is about *the image of the specific frame $q_1, \dots, q_p$*.
The dependence vanishes in the limit for almost every $Q_0$, which is the content of Oseledets'
theorem, but the limit is exactly what [§5.4](#54-the-hypothesis-audit) says is unavailable.

There is one exception, and it matters. At $p = D$ the frame drops out:
$\mathrm{vol}_D(M_n Q_0 \cdot)$ is $\lvert \det M_n \rvert$ regardless of $Q_0$, so
$\sum_i \lambda_i^{(n)}$ is **frame-independent at every finite $n$**, while each individual
$\lambda_i^{(n)}$ is not. That is [Proposition 3](#8-closed-form-propositions-the-test-suite-pins).

`caustic/cocycle.py` takes $Q_0 = I$ `[code]`. So the per-index finite-time numbers reported in
[§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model) are growth rates of the
*coordinate* flag of the residual-stream basis, not of a generic flag; the reported **sum** is
not, and is the one statistic among them that carries no frame convention at all.

**Observation, offered as an observation and not as a mechanism.** The most frame-robust of the
reported statistics is `sum`, and `sum` on the token product is also the largest converged
separation measured, at 25% `[E§4]`. That is consistent with the frame argument. It is one
comparison on one trajectory pair and it is not evidence for the frame argument.

### 6.4 Two implementation details that are not cosmetic

- **Sign folding.** Without forcing $(R_k)_{ii} > 0$, the accumulated
  $\log \lvert (R_k)_{ii} \rvert$ is still numerically right but $Q_k$ drifts between sign
  conventions step to step, so the frame whose volume (21) describes is not the same frame at
  step $k$ and step $k+1$. The fold is what makes (20) an identity about one frame
  `[code: caustic/cocycle.py]`.
- **The floor.** A genuinely singular step would give $\log 0 = -\infty$ and poison the running
  sum. The diagonal is clamped at `1e-300` `[code: caustic/cocycle.py]`, so a dead direction
  reads as about $\log(10^{-300}) = -300 \ln 10 \approx -690.8$ rather than as a NaN. Folding
  produces exactly this case, so it is a **supported input, not an error**, and
  [Proposition 6](#8-closed-form-propositions-the-test-suite-pins) pins it.

---

## 7. Persistence modules, and why the bridge is empty

The programme's organising claim is that the filtration (11) is the same shape of object
persistent homology consumes. Stating that precisely enough to be judged requires the structure
theorem, so here it is, and then the judgement — which is negative, and negative for structural
reasons that were available before any measurement.

### 7.1 Definitions

Let $k$ be a field.

**Persistence module.** A persistence module over $\mathbb{R}$ is a functor
$M : (\mathbb{R}, \leq) \to \mathbf{Vect}_k$: a family of vector spaces
$\{M_s\}_{s \in \mathbb{R}}$ with linear **structure maps**
$\varphi_{s \to t} : M_s \to M_t$ for every $s \leq t$, satisfying
$\varphi_{s \to s} = \mathrm{id}$ and
$\varphi_{t \to u} \circ \varphi_{s \to t} = \varphi_{s \to u}$.

**Pointwise finite dimensional, p.f.d.** $\dim_k M_s < \infty$ for every $s$.

**Interval module.** For an interval $I \subseteq \mathbb{R}$, the module $k_I$ has
$(k_I)_s = k$ if $s \in I$ and $0$ otherwise, with structure maps the identity whenever both ends
lie in $I$ and zero otherwise. Interval modules are the indecomposables.

**Barcode.** The multiset of intervals appearing in a decomposition into interval modules.

### 7.2 The structure theorem, with its hypotheses

**Theorem, interval decomposition, Crawley-Boevey.** Let $M$ be a persistence module over a
totally ordered index set with coefficients in a **field** $k$, and suppose $M$ is **pointwise
finite dimensional**. Then

**(22)**

$$
M \;\cong\; \bigoplus_{j \in J} k_{I_j}
$$

for a multiset of intervals $\{I_j\}_{j \in J}$, and that multiset is **unique** up to reordering.

Both hypotheses are load-bearing. Over a general ring the decomposition fails — this is why
persistent homology is computed with field coefficients and why torsion is invisible to a
barcode. Pointwise finite dimensionality, or the weaker q-tameness which gives the same
conclusion off the diagonal, is what rules out pathological modules. Uniqueness comes from the
Krull–Remak–Schmidt–Azumaya theorem applied to the endomorphism rings of interval modules, which
are local.

### 7.3 The Oseledets filtration is a persistence module

**Proposition 4.** Fix $x$ and let $\lambda_1 > \cdots > \lambda_k$ with multiplicities
$m_1, \dots, m_k$ be as in §5.2. Define, for $s \in \mathbb{R}$,

**(23)**

$$
F_s \;=\; \Big\{\, v \in \mathbb{R}^D \;:\; \limsup_{n \to \infty} \tfrac{1}{n} \log \lVert A^{(n)}(x) v \rVert \;\leq\; s \,\Big\}
$$

with structure maps the inclusions $F_s \hookrightarrow F_t$ for $s \leq t$. Then $F$ is a
pointwise finite dimensional persistence module over $\mathbb{R}$, and its barcode is

**(24)**

$$
\Big\{\, [\lambda_i, \infty) \ \text{ with multiplicity } \ m_i \;:\; i = 1, \dots, k \,\Big\}
$$

*Proof.* Each $F_s$ is a subspace of $\mathbb{R}^D$ by (12), and $s \leq t$ gives
$F_s \subseteq F_t$, so $F$ is a functor and $\dim F_s \leq D < \infty$. By (11) and (12),
$F_s = V_i$ exactly when $\lambda_i \leq s < \lambda_{i-1}$, so $s \mapsto \dim F_s$ is a step
function that is $0$ below $\lambda_k$, jumps by $m_i$ at each $s = \lambda_i$, and reaches $D$ at
$s = \lambda_1$. All structure maps are injective, so no generator is ever killed; a p.f.d. module
whose structure maps are all injective decomposes as a direct sum of half-infinite interval
modules born at the jump points, with multiplicity equal to the size of the jump. That is (24). ∎

So the bridge is not a metaphor. The structure theorem genuinely applies, and the interval
decomposition genuinely exists. Now the part that matters.

### 7.4 Why the bridge is structurally empty

**Three things really are shared.** Both objects are functors from a totally ordered real
parameter to finite-dimensional vector spaces, so theorem (22) applies to both with the same
hypotheses and delivers uniqueness for both. Both decompose into intervals carrying
multiplicities, so any tool whose input type is "multiset of intervals with multiplicity" accepts
either without modification — a real, checkable, software-level claim, and the one the
implementation depends on. And neither invariant depends on a choice of basis for the ambient
space.

**Then it stops, and it stops for two reasons that are proofs rather than measurements.**

**Reason 1: every structure map is injective, so no bar ever dies.** In persistent homology the
structure maps are induced by inclusions of *spaces* and are in general neither injective nor
surjective — which is exactly why homology classes are born and later die, and why bar *length*
carries information. In (23) every structure map is a subspace inclusion inside one fixed
$\mathbb{R}^D$, hence injective, hence monic, hence nothing is ever killed. By Proposition 4 the
barcode is therefore $k$ **half-infinite rays** $[\lambda_i, \infty)$ and can be nothing else. Bar
length, the birth–death pairing, and the diagonal of a persistence diagram — the entire
discriminating content of persistent homology — are degenerate on this module *by construction*.

The consequence is worth writing as an identity rather than a complaint. The barcode is
determined by the pair (sorted distinct exponents, multiplicities), and the sorted exponents are
already the input. So:

> **The barcode of the growth filtration IS the multiplicity vector $(m_1, \dots, m_k)$, and
> carries nothing else.** The encoding's entire cash value is how far the measured spectrum
> departs from all-distinct.

**Reason 2: stability points the opposite way.** Persistence barcodes satisfy a 1-Lipschitz
stability theorem,
$d_B(\mathrm{Dgm}(f), \mathrm{Dgm}(g)) \leq \lVert f - g \rVert_\infty$, by
Cohen-Steiner–Edelsbrunner–Harer and its interleaving generalisation. That theorem is *the*
reason persistence is useful as a feature: a small perturbation of the input moves the output a
bounded amount, so a barcode is a legitimate summary of noisy data. Lyapunov spectra satisfy no
such theorem. By Corollary B of [§5.3](#53-two-corollaries-used-later), $\lambda_1$ is merely
**upper semicontinuous** in the cocycle, and continuity genuinely fails — Bochi's theorem
exhibits arbitrarily small $C^0$ perturbations that collapse the exponents to zero. **Any argument
that carries a stability guarantee across this bridge is invalid**, in that direction, for that
reason. The property that makes persistence worth importing has no counterpart on the side being
imported to.

**A third mismatch, smaller but real.** The Oseledets filtration is a bundle over the base with
$A(x)V_i(x) \subseteq V_i(Tx)$, equation (13). A persistence module has no base dynamics, so the
bridge as implemented discards $x$ and keeps one fibre.

**And the shipped triples are not even the canonical barcode.** `growth_filtration` returns
`(birth, death, multiplicity)` with `death` set to *the next distinct exponent below*, or `-inf`
for the last `[code: caustic/oseledets.py]`. Those intervals are disjoint and tile the line; the
canonical bars (24) all overlap and all run to $+\infty$. The shipped encoding is a **gap
encoding**, chosen so downstream barcode tooling has finite bars to consume. It is a deliberate
convention with a purpose, and this README states it rather than letting the word "barcode" carry
a claim the mathematics does not support.

**Verdict, one sentence.** The bridge is a correct statement about the *shape* of the two objects
and a false statement about their *content*; its entire value is the multiplicity vector, so it is
worth exactly as much as measured spectra depart from all-distinct — and
[§7.5](#75-the-measurement-that-confirms-it) is that measurement.

### 7.5 The measurement that confirms it

`filtration_entropy` reports the Shannon entropy of the multiplicity distribution — $0$ for one
subspace carrying every direction, $\log D$ for $D$ singletons, in which case the encoding is free
and worthless `[code: caustic/oseledets.py]`. Token product, block 3, 46 steps, `D = 768`,
`log(D) = 6.6438`, across the full monotone tolerance sweep `[E§5]`:

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

  control: the shuffled-token condition through the identical sweep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

`entropy/logD = 0.9986` at the finest honest tolerance `[E§5]`. **That number is what
[§7.4](#74-why-the-bridge-is-structurally-empty) predicts, not an unlucky draw.** The structural
argument caps the encoding's payoff at the multiplicity vector; a spectrum computed in floating
point from an operator with no imposed symmetry has no exact ties, so every multiplicity is 1, so
the multiplicity vector is $(1, 1, \dots, 1)$, so the entropy is $\log D$ and the payoff is zero.
The measurement did not discover a disappointing fact about distilgpt2. It confirmed a fact about
the encoding that Proposition 4 already contained.

Two further readings of the sweep, both negative, both stated in
[§10.2](#102-the-oseledets-bridge-buys-nothing-on-a-real-spectrum) with their controls.

---

## 8. Closed-form propositions the test suite pins

Every assertion in the suite has an independently known answer. Each is stated below as a
proposition with a one-line proof, followed by the tolerance the suite holds it to. A proposition
whose proof you can read is a specification; a test that merely checks self-consistency is not.

The reason for this standard: the expensive failures in a spectral or topological pipeline are
**silent**. A transposed index, a wrong filtration, or a scale divided out where it should have
been kept does not raise. It produces a well-formed result with healthy variance and no
information, the downstream fit still trains, and the benchmark still reports a number. A
self-consistency test passes on every one of them. Only an assertion with an independently known
answer fails.

### Cocycle propositions

**Proposition 1, diagonal cocycle.** If $J_k = \mathrm{diag}(a_1, \dots, a_D)$ for every $k$, then
$\lambda_i^{(n)} = \log \lvert a_i \rvert$ exactly, for every $n$.
*Proof.* The coordinate axes are invariant under every $J_k$, so with $Q_0 = I$ the frame never
rotates and $(R_k)_{ii} = \lvert a_i \rvert$ at every step; average. ∎
Tolerance `1e-8` `[E§6]`.

**Proposition 2, orthogonal cocycle, the negative control.** If every $J_k$ is orthogonal, then
$\lambda_i^{(n)} = 0$ exactly, for every $i$ and every $n$.
*Proof.* $J_k Q_{k-1}$ is a product of orthogonal matrices, hence orthogonal, and the QR
factorization of an orthogonal matrix with positive diagonal $R$ is $Q$ = the matrix itself and
$R = I$; so $\log (R_k)_{ii} = 0$. ∎
Tolerance `1e-8` `[E§6]`. An implementation that multiplies Jacobians directly and normalizes only
at the end drifts off zero here, which is why this is the control that catches it.

**Proposition 3, exponents sum to the mean log-determinant.**
$\sum_{i=1}^{D} \lambda_i^{(n)} = \frac{1}{n}\sum_{k=1}^{n} \log \lvert \det J_k \rvert$, exactly,
for every $n$.
*Proof.* Take $\lvert\det\rvert$ of (20): $\lvert \det M_n \rvert = \prod_k \prod_i (R_k)_{ii}$,
since $Q_0$ and $Q_n$ are orthogonal and each $R_k$ is triangular with positive diagonal. Take
logs, divide by $n$, and use $\det M_n = \prod_k \det J_k$. ∎
Tolerance `1e-8` `[E§6]`. This is the finite-$n$ form of Corollary A, and by
[§6.3](#63-the-frame-dependence-that-survives-at-finite-n) it is the one reported statistic that
does not depend on $Q_0$.

**Proposition 4, uniform scaling shifts the whole spectrum.** For $c > 0$, replacing every $J_k$
by $c J_k$ sends $\lambda_i^{(n)} \mapsto \lambda_i^{(n)} + \log c$ for every $i$.
*Proof.* $Z_k \mapsto c Z_k$ leaves $Q_k$ unchanged and sends $R_k \mapsto c R_k$, so each
$\log (R_k)_{ii}$ gains $\log c$. ∎
Tolerance `1e-9` `[E§6]`.

**Proposition 5, unit eigenvector residual.** If $T x = \lambda x$ with $\lVert x \rVert = 1$, then
$\lVert T x - x \rVert = \lvert \lambda - 1 \rvert$.
*Proof.* $T x - x = (\lambda - 1) x$; take norms and use $\lVert x \rVert = 1$. ∎
This is the exact residual behind the informal reading "an eigenvalue near 1 means a nearly fixed
direction", and the sharp form of the exponent-zero condition. **Provenance note:** unlike
Propositions 1–4 this one is *not* currently exercised by the suite. It is stated because it is
used when reading a spectrum, and a lemma used in reading should be pinned; that it is not yet is
a gap, recorded as such rather than papered over.

**Proposition 6, a dead direction is a supported input.** If some step has $(R_k)_{ii} = 0$, the
floor gives $\log(10^{-300}) \approx -690.8$ for that term `[code: caustic/cocycle.py]`, so the
running mean is bounded below by that value: the exponent goes large and negative, never to NaN,
and the other $D-1$ exponents are untouched.
*Proof.* The clamp bounds each summand below by a finite constant, so the average of $n$ of them is
finite; the other diagonal entries are unaffected because $R$ is triangular. ∎
The suite asserts the affected exponent falls below `−100` and the rest stay finite `[E§6]`.

### Spectral-summary propositions

With $\sigma_1 \geq \cdots \geq \sigma_D$ the singular values of one $J$:

**(25)**

$$
\sigma_{\max} = \sigma_1,
\qquad
\mathrm{srank}(J) = \frac{\lVert J \rVert_F^2}{\sigma_1^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2},
\qquad
\log \mathrm{vol}(J) = \sum_i \log \sigma_i
$$

**(26)**

$$
\sigma_i \sim i^{-a}
\quad \Longrightarrow \quad
a \;=\; -\,\mathrm{slope}\big( \log \sigma_i \ \text{vs} \ \log i \big),
\qquad i \in [\,10,\ 400\,)
$$

The bulk window excludes the leading spike and the trailing numerical floor, neither of which
belongs to a power-law tail. `BULK = (10, 400)` is **absolute** `[code: caustic/spectrum.py]` and
must be made relative to $D$ before any cross-width comparison, or the comparison measures the
window rather than the model.

**Proposition 7, exact Jacobian of a linear block.** For a position-wise block
$h \mapsto h W^{\top}$, the diagonal block of the Jacobian at every position is exactly $W$.
*Proof.* The map is linear and acts position-wise, so the derivative is $W$ and there is no
cross-position term to leak. ∎
Tolerance `1e-10` `[E§6]` — the tightest in the suite, because it is the ground truth for the
estimator itself.

**Proposition 8, `tail_alpha` recovers an exact power law.** If $\sigma_i = (i+1)^{-a}$ exactly,
then the least-squares slope of $\log \sigma$ against $\log(i+1)$ is $-a$, so the fit returns $a$.
*Proof.* The points are exactly collinear with slope $-a$; least squares through collinear points
returns that slope. ∎
Recovered for `0.25 / 0.5 / 1.0 / 2.0` at tolerance `1e-9` `[E§6]`. A fit that cannot recover a
synthetic exponent cannot be trusted on a measured one.

**Proposition 9, `tail_alpha` is scale invariant.** Replacing every $\sigma_i$ by $c\sigma_i$,
$c > 0$, leaves $a$ unchanged.
*Proof.* $\log(c\sigma_i) = \log \sigma_i + \log c$ adds a constant to the response, which moves
the least-squares intercept and not the slope. ∎
Asserted at `1e-6`, `1.0` and `1e6` `[code: tests/test_jacobian.py]`.

**Proposition 10, flat spectrum has exponent zero, the negative control.** If $\sigma_i = c$ for
all $i$, then $a = 0$.
*Proof.* $\log \sigma_i$ is constant, so the slope is $0$. ∎
Tolerance `1e-9` `[E§6]`. A pipeline that reports structure in a flat spectrum reports it
everywhere, which is the failure mode that makes fractal claims worthless.

**Proposition 11, `log_volume` is $\log\lvert\det J\rvert$.**
$\sum_i \log \sigma_i = \log \lvert \det J \rvert$.
*Proof.* $\lvert\det J\rvert = \prod_i \sigma_i$ from the SVD, since the orthogonal factors have
unit modulus determinant. ∎
Tolerance `1e-8` against `torch.linalg.slogdet` `[E§6]`.

**Proposition 12, stable-rank bounds.** $1 \leq \mathrm{srank}(J) \leq \mathrm{rank}(J) \leq D$,
and $\mathrm{srank}$ is scale invariant.
*Proof.* $\sum_i \sigma_i^2 \geq \sigma_1^2$ gives the lower bound; $\sigma_i \leq \sigma_1$ for
each of the $r = \mathrm{rank}(J)$ nonzero values gives $\sum_i \sigma_i^2 \leq r\sigma_1^2$.
Scaling multiplies numerator and denominator by $c^2$. ∎

**Proposition 13, Krylov agreement.** The block power iteration on $J^{\top}J$ converges to the
top-$k$ singular values of the same $J$ that `exact_jacobian` returns.
Held to `1e-6` on the closed-form linear block `[E§6]`, and measured at `1.685e-04` max relative
error on the top 8 of a live block `[E§1]`. The estimator is **correct**;
[§9.1](#91-the-cost-of-working-in-jacobian-space) is about its cost, not its accuracy, and
conflating the two would be the easy misreading.

### Filtration propositions

**Proposition 14, degenerate and distinct ends.** A spectrum with all exponents equal gives one bar
of multiplicity $D$ and filtration entropy $0$; a spectrum with $D$ distinct exponents gives $D$
bars of multiplicity $1$ and entropy $\log D$.
*Proof.* Immediate from the definitions of §7.3 and of Shannon entropy on the multiplicity
distribution. ∎
Both ends are asserted, and the distinct case is deliberately the prominent one, because it is the
failure mode of the whole bridge `[code: tests/test_oseledets.py]`.

**Proposition 15, monotonicity in the tolerance.** Widening the grouping tolerance never increases
the bar count.
*Proof.* The grouping merges adjacent exponents whose gap is at most `tol`; enlarging `tol`
enlarges the merge relation, so groups only coalesce. ∎
A grouping rule that is not monotone in its own parameter is not a filtration, which is why this is
asserted rather than assumed `[code: tests/test_oseledets.py]`.

### Attractor-dimension propositions

The Kaplan-Yorke dimension of a spectrum $\lambda_1 \geq \cdots \geq \lambda_D$ is

**(27)**

$$
D_{KY} \;=\; j \;+\; \frac{\sum_{i \leq j} \lambda_i}{\lvert \lambda_{j+1} \rvert},
\qquad
j \;=\; \max\Big\{\, m : \sum_{i \leq m} \lambda_i \geq 0 \,\Big\}
$$

The interpolation term is what makes it fractional, and therefore a fractal dimension rather than a
count of directions. **The Kaplan-Yorke conjecture** — that $D_{KY}$ equals the information
dimension of the attractor — is a conjecture, proved only in restricted settings; nothing here
proves it `[code: caustic/attractor.py]`.

**Proposition 16, the Lorenz ground truth.** The Lorenz system at $\sigma = 10$, $r = 28$,
$b = 8/3$ has spectrum $(0.906,\, 0,\, -14.572)$. Then $j = 2$, since $0.906 + 0 \geq 0$ and
$0.906 + 0 - 14.572 < 0$, and

$$
D_{KY} \;=\; 2 + \frac{0.906 + 0}{14.572} \;=\; 2.0622
$$

which is the value quoted in the standard texts. The Takens sufficient embedding size is
$\lceil 2 \times 2.0622 \rceil + 1 = 6$.
*Proof.* Direct evaluation of (27). ∎
Asserted at `1e-3` `[code: tests/test_attractor.py]`. **This is the assertion that validates the
implementation against the literature rather than against itself**, and it is the reason any
number in [§9.3](#93-attractor-dimension-per-block) is worth printing.

**Proposition 17, $D_{KY}$ is scale invariant and bounded by the width.** For $c > 0$,
$D_{KY}(c\lambda) = D_{KY}(\lambda)$, and $0 \leq D_{KY}(\lambda) \leq D$ always.
*Proof.* Scaling by $c > 0$ multiplies every partial sum by $c$, leaving every sign and therefore
$j$ unchanged, and the interpolation term is a ratio of two quantities both scaled by $c$. The
bounds follow because $j \in \{0, \dots, D\}$ and the interpolation term lies in $[0, 1)$. ∎
Asserted at relative `1e-12` over three decades of $c$, and the bound asserted over 200 random
spectra `[code: tests/test_attractor.py]`. Scale invariance is what makes $D_{KY}$ a candidate for
a constant: it does not depend on how the trajectory was parameterised.

---

## 9. Measured results

Every number carries its tag. The measurement host is
[§16](#16-requirements-and-measurement-environment); no number is projected off it.

### 9.1 The cost of working in Jacobian space

distilgpt2, layer 3, one token position. Median of 30 calls after 3 warmups, CUDA synced `[E§1]`.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block forward                 0.588 ms
  one JVP                       6.144 ms      10.45x forward
  top-8 power, 20 iters      1428.938 ms    2429.5x forward   <-- the estimator
  full 768x768 jacrev          53.694 ms      91.3x forward   <-- the exact path

  control: the exact computation, same input, same hardware, same accuracy
           target. Agreement of the estimator with exact svdvals:
           1.685e-04 max relative error on the top 8 (k=8, iters=20).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**The exact Jacobian is 26.6× cheaper than the Krylov estimator of its own top eight singular
values** `[E§1]`.

**Accuracy is not the issue.** The estimator agrees with exact `svdvals` to `1.685e-04` max
relative error `[E§1]`, and Proposition 13 pins that agreement on a case with a closed-form answer.
The mechanism of the loss is specific and structural: batched reverse-mode AD vectorizes across all
768 output components in one pass, while $k$-column power iteration runs $k \times \text{iters}$
sequential JVP/VJP passes with no batching `[E§1]`.

**Not measured:** the width at which the ordering inverts `[E§1]`. It is not assumed, and
`top_singular_values` is kept precisely for the regime beyond that crossover. Anyone quoting this
ratio for a larger model is quoting it outside its measured range.

**Consequence for the programme:** the premise that Jacobian space needs a cheap spectral bound to
be tractable is **false at this scale**; bounds matter for large `D` only `[E§1]`. This is the seed
of candidate `C2` in [`CANDIDATES.md`](CANDIDATES.md) — a class of spectral estimator that defaults
to an iterative method regardless of problem size — and it was found by running the code rather
than by reasoning about it.

### 9.2 Finite-time QR characteristic exponents of a live model

Read [§5.4](#54-the-hypothesis-audit) first. These are finite-time QR characteristic exponents of
finite matrix products, not Lyapunov exponents in the sense of (12).

Both products of [§4.2](#42-the-two-products-and-why-only-one-is-a-cocycle), both against the same
shuffled-token control. The shuffle preserves the token multiset and therefore the embedding
statistics, and destroys only grounded structure `[E§3]`.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOKEN PRODUCT, BLOCK 3, 46 STEPS — CONVERGED

                  lam1        sum       positive     last-step drift
  grounded     +0.1653    -226.74      139/768          0.0012   (0.7% of value)
  shuffled     +0.1852    -170.37      151/768          0.0003   (0.2% of value)

  control: the shuffled-token condition, same passage, same token multiset,
           same block, same 46 steps; plus each value's own drift column
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Both had settled `[E§4]`. **That is the only reason these numbers are printed and the
[§10.3](#103-the-6-step-block-product-discarded-and-why) numbers are not** — the two boxes differ
in the drift column and in nothing else structural.

**What they say.** `lam1 > 0` in every condition means perturbations grow along at least one
direction, so the map is locally chaotic. A sum near `−227` over 768 dimensions means the map
contracts volume enormously per step, with only 139 of 768 directions expanding `[E§4]`. A positive
leading exponent with a strongly negative sum is the textbook signature of a **dissipative chaotic**
system on a low-dimensional attractor.

**What they do not say.** By (15) this establishes the *precondition* for folding and nothing more.
The sheet is being squeezed hard; whether it is also being bent over onto itself is exactly the
question these numbers cannot answer. A reader who leaves this section believing folding has been
observed has been misled, and the fault would be this document's.

**Separation, and why it matters more than the values** `[E§4]`:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  separation = (grounded - shuffled) / grounded

  block  lam1 -5.3274   sum -0.0943   n_positive -0.1646   <-- NOT CONVERGED (§10.3)
  token  lam1 -0.1202   sum -0.2486   n_positive -0.0863   <-- converged
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**All six are negative** `[E§4]`. Grounded text is *less* chaotic and contracts *more* volume than
the scrambled control. The largest converged separation is `sum` on the token product, at **25%**
`[E§4]` — which is also, by [§6.3](#63-the-frame-dependence-that-survives-at-finite-n), the one
statistic in the table that carries no frame convention.

**This is the first quantity in the programme that separated with consistent sign** `[E§4]`. In the
layer sweep of [§10.1](#101-spectral-summaries-do-not-separate-grounded-from-shuffled-text), three
of four summaries flipped. It is also, on the mechanism's own terms, a weak signal: contraction is a
precondition, and a precondition separating between two conditions is not the mechanism separating
between them.

The block `lam1` figure of `−5.3274` is an artefact of dividing by a near-zero grounded value
`0.0224`; the absolute difference is `−0.119`, and the underlying number is not converged anyway
`[E§4]`. It is printed with that annotation rather than quietly dropped, because a `−5.3` in a
separation table is exactly the figure that survives into a slide deck.

**Scope, in the same breath as the claim.** `n = 1` passage, 1 model, 1 seed, no error bars across
texts `[E§4]`. Six numbers from a single trajectory pair, of which the top row is discarded. They
agree in sign, which is why they are reported; **they are not a statistical result**, and
[§11](#11-what-would-falsify-this) names the measurement that would make them one.

### 9.3 Attractor dimension per block

The one number the three linearizations jointly produce. The Jacobian is the local linearization;
composed along the token product it gives a spectrum; the Kaplan-Yorke formula (27) collapses that
spectrum into a fractal dimension. Per-block, on the token product, `D = 768` `[A]`:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KAPLAN-YORKE DIMENSION, TOKEN PRODUCT, GROUNDED

  block      sum       n_pos      D_KY     D/D_KY    2*D_KY+1
  ------------------------------------------------------------
    0      +117.3        512    768.00       1.0        --      <-- formula saturates
    1      -541.6          9     29.57      26.0        61
    2         --          --    225.76        --        --
    3         --          --    298.08        --        --
    4         --          --    482.20        --        --
    5         --          --    674.67        --        --

  cv of D_KY across the six blocks: 0.6801
  control: shuffled-token, matched multiset — 5 of 6 same sign, |mean| 0.0758
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Cells marked `--` are not printed because they are not sourced here; the per-block `sum` and
`n_pos` columns are recorded only for blocks 0 and 1 `[A]`.

**Three readings, and the first one is a warning about the formula.**

**1. `D_KY = 768.00` at block 0 is the formula saturating, not a measured dimension.** The exponent
sum at block 0 is `+117.3`, i.e. **positive** `[A]`. Because the exponents are sorted descending,
a non-negative total sum forces every partial sum to be non-negative, so the index $j$ of (27)
runs to the end of the spectrum, there is no $\lambda_{j+1}$ to interpolate against, and
`kaplan_yorke_dimension` returns `float(lam.size)` by construction `[code: caustic/attractor.py]`.
The value `768.00` is therefore the *ambient width being echoed back*, and reading it as "the
attractor fills the space" would be reading the boundary case of a formula as a measurement. The
512 positive exponents at block 0 `[A]` say the same thing more directly: at block 0 the token
product expands volume, so the precondition of (15) does not even hold there.

**2. The profile is compress-then-expand with depth.** Block 1 gives `D_KY = 29.57` with exponent
sum `−541.6` and only 9 positive exponents `[A]` — a **26× compression** against the width, and a
Takens sufficient reconstruction size of **61** by
[Proposition 16](#8-closed-form-propositions-the-test-suite-pins)'s formula. Blocks 2 through 5
then climb monotonically: `225.76`, `298.08`, `482.20`, `674.67` `[A]`. Whatever the residual
stream is doing, it is not doing one thing uniformly with depth.

**3. `D_KY` is not a constant with depth, so it is not the width-invariant number the thesis asked
for.** The coefficient of variation across the six blocks is `0.6801` `[A]`. A low cv would have
made `D_KY` a candidate constant; `0.68` is not low. **Separation against the shuffled control is
5 of 6 same sign with |mean| `0.0758`** `[A]`: consistent in direction, weak in magnitude, and
**not a detector**. It is reported at the same weight as the layer sweep's failure, because a
consistent sign on a small magnitude with `n = 1` is a direction and not a result.

**Why this is worth computing at all.** Two classical results attach to an attractor dimension and
both are statements about compression: Whitney's embedding theorem, that a $d$-dimensional manifold
embeds in $\mathbb{R}^{2d+1}$, and Takens' delay-embedding theorem, that an attractor of box
dimension $d$ is reconstructed from $2d+1$ generic delayed scalar observations. If the hidden-state
dynamics sat on an attractor of dimension much smaller than the width, then the width is not the
information content of the state and $2 D_{KY} + 1$ is the scale at which lossless reduction stops
being possible — a bound with a number in it rather than a hyperparameter, falsifiable by reducing
below it and watching reconstruction degrade. Block 1 is the only block where that reading is even
available.

**Two scope statements that must travel with these numbers.**

- **The hypotheses fail here too, and worse.** Takens' theorem, like Oseledets', requires a
  measure-preserving dynamical system. A finite non-autonomous trajectory driven by input tokens is
  not obviously one, and [§5.4](#54-the-hypothesis-audit) applies verbatim. The numbers are computed
  *as if* it were, and that gap is the single largest caveat on any conclusion drawn from them
  `[code: caustic/attractor.py]`.
- **This is a different run from [§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model).**
  The attractor experiment uses its own four-sentence passage and starts the product at `T // 4`
  rather than at the first token `[code: caustic/experiments/attractor_dimension.py]`. Its
  per-block sums are therefore **not** comparable to the `−226.74` of §9.2, which came from a
  different passage over a different step range `[E§4]`. Nothing here should be read as the same
  spectrum measured twice.

---

## 10. Negative results

A first-class section, placed where a reader reaches it before the quick start. There are three
headline negatives — the layer sweep, the Oseledets bridge, and the discarded 6-step product — plus
a salvage and two environment facts. Every entry names what it was compared against.

### 10.1 Spectral summaries do not separate grounded from shuffled text

6 blocks, 10 token positions, one 61-token passage, against the same token multiset in shuffled
order `[E§3]`. Separation is `(grounded − shuffled)` as a fraction of the grounded value, so every
entry below already carries its control inside the arithmetic.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  control: shuffled-token condition, matched token multiset,
           same model, same 10 positions, same 6 blocks

  sigma_max     [ 0.123 -0.012 -0.135 -0.123  0.233 -0.035]  |mean| 0.0085
  stable_rank   [ 0.346  0.165  0.196  0.091 -0.299  0.184]  |mean| 0.1138
  tail_alpha    [ 0.051 -0.088 -0.091 -0.037  0.067  0.227]  |mean| 0.0217
  logdet_abs    [ 0.805  0.187  0.002 -0.111 -0.282 -0.273]  |mean| 0.0548
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Three of the four flip sign across layers** `[E§3]`. A summary that is larger under grounded text
at block 0 and smaller at block 2 is not measuring grounding; it is measuring the block. Only
`stable_rank` holds sign, at 5 of 6 layers, median **+17.5%**, with block 4 reversing it — one
reversal away from the same verdict. **No summary separates the conditions reliably.**

This kills "J-space summary statistics as a hallucination detector" *as stated*, and it is recorded
as `K1` in [`CANDIDATES.md`](CANDIDATES.md) so it is not proposed again. The caveat that keeps it
open rather than closed: a shuffled-token control is an out-of-distribution control, so what is
supported here is "J-space summaries do not detect *scrambling*", which is weaker than the headline
the arm was built to test.

Read against [§1.4](#14-why-a-vanishing-determinant-is-the-wrong-place-to-look) this is also the
expected outcome rather than a surprise: `logdet_abs` and `sigma_max` are crease-detectors, and the
mechanism of §1.3 leaves no crease for them to find. That reading is a post-hoc rationalisation of a
failure, is flagged here as one, and does not become evidence for anything.

### 10.2 The Oseledets bridge buys nothing on a real spectrum

The programme's organising claim has two halves. The first is that the Jacobian is the causal object
where a point cloud is not; that half stands, and it stands definitionally
([§3.3](#33-what-the-mechanism-does-not-claim)). The second is that the filtration Oseledets'
theorem induces is the same shape of object a persistence module decomposes into, so barcode tooling
applies to it. **The second half does not survive, and [§7.4](#74-why-the-bridge-is-structurally-empty)
shows it did not need to be measured to fail.** The sweep of
[§7.5](#75-the-measurement-that-confirms-it) then confirms it three ways:

**1. At any tolerance fine enough to be honest, the filtration is `D` singletons.**
`entropy/logD = 0.9986` at `tol = 1.308e-05` `[E§5]`. The encoding is the sorted spectrum with extra
steps.

**2. There is no plateau.** The bar count slides smoothly from 763 down to 1 with no tolerance range
where it holds steady `[E§5]`. A spectrum with genuine subspace structure would show a plateau,
because a real degeneracy survives a range of tolerances. Every tolerance here gives a different
answer, which means **the grouping is measuring the tolerance, not the operator.**

**3. Grounded and shuffled are indistinguishable at every tolerance.** 763 vs 759, 681 vs 674, 531
vs 532, 289 vs 279, 107 vs 113, 42 vs 31, 6 vs 7 `[E§5]`. At the median adjacent gap both give 384
bars, with 148 against 152 of multiplicity greater than 1, and `entropy/logD` of **0.8495 against
0.8505** — a difference of `0.001` `[E§5]`.

**Control:** the shuffled-token condition through the identical monotone sweep `[E§5]`.

**What this means, stated as the stronger claim it is.** The Lyapunov spectrum of a transformer
Jacobian product is *generic*: no repeated exponents, no degenerate Oseledets subspaces of dimension
above one. That is a substantive fact and not merely an absence — it says the dynamics at this level
carry no symmetry that would force an invariant subspace. Combined with
[§7.4](#74-why-the-bridge-is-structurally-empty), the negative is not "the bridge happened not to
pay here" but "the bridge can only ever pay in the multiplicity vector, and on a generic spectrum
the multiplicity vector is all ones." A better model, a longer trajectory or a cleverer tolerance
would not change the first half of that sentence.

**The salvage.** `tolerance_sweep` and `filtration_entropy` are a general test for whether *any*
claimed filtration structure is real, and the diagnostic is the **plateau** rather than the value. A
grouping that produces a different answer at every tolerance is reporting its own parameter. That
test is reusable against any claim of discovered subspace or cluster structure from a thresholded
spectrum, and it costs one sweep `[E§5]`.

**Limits.** One model, one width, one passage, one block, one seed, 46 product steps `[E§5]`. A
degeneracy could exist at larger width, in a model with tied or structured weights, or over a longer
trajectory. Nothing here rules that out; it rules out distilgpt2 at `D = 768`.

### 10.3 The 6-step block product, discarded and why

This block exists because a repository that shows its own discarded data is more trustworthy than
one that does not. **The following numbers were produced, examined, and thrown away. Nothing in this
repository rests on them.**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BLOCK PRODUCT, 6 STEPS — NOT CONVERGED, DISCARDED

                  lam1        sum       positive     last-step drift
  grounded     +0.0224    -276.65       79/768          0.0255   <-- drift > value
  shuffled     +0.1416    -250.56       92/768          0.0083

  control: the value's own convergence trace, returned step by step by
           caustic.cocycle.finite_time_spectrum

  grounded lam1 across its six steps:
      0.364,  0.027,  -0.016,  -0.002,  -0.003,  0.022     <-- still oscillating
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**The drift, `0.0255`, is larger than the value it drifts on, `0.0224`** `[E§4]`. That number is
noise. Read without its drift column it looks like a clean result — a small positive leading
exponent, a large negative sum, a plausible 79-of-768 split — and it would have made a presentable
table row. The drift column is the only thing distinguishing it from
[§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model), and it exists because
`finite_time_spectrum` returns the running exponents after **every** step rather than only the last
`[code: caustic/cocycle.py]`.

There are two independent reasons to discard it, and the second is more serious than the first:

- Six steps is nowhere near the asymptotic regime equation (12) describes, so a six-step average is
  not a Lyapunov exponent `[E§4]`. **This cannot be fixed by better numerics** — six blocks is six
  steps, and it needs a deeper model.
- By [§4.2](#42-the-two-products-and-why-only-one-is-a-cocycle) the block product **is not a cocycle
  at all**: it composes six *different* maps, the cocycle identity (7) never holds, and there is no
  multiplicative ergodic theorem for a six-step average to be a finite-time approximation *of*. A
  deeper model does not repair this one either. Depth would give more steps of a product that is
  still not a cocycle.

It is printed rather than deleted because deleting it is exactly how a plausible-looking figure
survives into a table.

### 10.4 The salvage, tail_alpha is structural not semantic

The §10.1 failure has a usable half. The reason `tail_alpha` did not separate is that the two
conditions **agree**, and they agree far more tightly than the noise in either one.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  tail_alpha, blocks 1-5, 10 token positions

    grounded    0.2374 +/- 0.0293    cv 0.124
    shuffled    0.2342 +/- 0.0481    cv 0.205      <-- the control, same box
    difference  0.0033  =  1.4% of grounded
    block 0     0.6168               excluded AFTER inspecting the data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The `0.0033` difference is smaller than the grounded standard deviation `0.0293` by roughly a factor
of nine, and smaller than the shuffled standard deviation `0.0481` by roughly fifteen `[E§3]`.
Agreement to **1.4%** between coherent and scrambled text means the exponent is a property of the
**architecture** rather than of the content.

That is precisely what disqualifies it as a detector, and precisely what qualifies it as a **pruning
budget**: for $\sigma_i \sim i^{-a}$ the rank needed for a stated relative error follows in closed
form from (26), so a measured exponent replaces a tuned density constant. That is candidate `C1` in
[`CANDIDATES.md`](CANDIDATES.md). **The negative result created the candidate**, which is the single
strongest argument in this repository for publishing negatives rather than filing them.

**Caveat, unresolved and load-bearing.** Block 0 was excluded **after** inspecting the data `[E§3]`.
Its value is `0.6168` against `0.2374 ± 0.0293` for blocks 1–5 — a clear outlier, and "clear
outlier" is exactly what a forking path feels like from the inside. The exclusion has a plausible
mechanism, that block 0 sees the raw embedding rather than a residual stream already shaped by an
attention block, but a plausible mechanism found after the fact is not a pre-registration. What it
costs, exactly:

- The `1.4%` agreement is conditional on the exclusion and has **not** been recomputed with block 0
  included and reported alongside.
- Candidate `C1` therefore rests on an untested exclusion, and additionally on `tail_alpha` being
  width-invariant, which has **not** been measured at any width other than 768.
- It has not been re-derived on held-out text `[plan]`. Until it is, this section is a forking path
  and is labelled one here, in `EVIDENCE.md`, in `CANDIDATES.md`, in `PLAN.md`, and in the docstring
  of `caustic/spectrum.py` — five places, because it is the single easiest thing here to forget.

### 10.5 Two environment facts, both found by running rather than reading

Neither is a result about language models. Both cost real time and both are the kind of thing a
reader repeating this work will hit within an hour.

- **`torch.func.jvp` raises `NotImplementedError` on `_scaled_dot_product_efficient_attention`**
  `[E§2]`. Forward-mode AD is unimplemented for the fused SDPA kernel that `transformers` selects by
  default. Any JVP-based method must load with `attn_implementation="eager"`. Reverse mode is
  unaffected. Recorded as `D1` in [`CANDIDATES.md`](CANDIDATES.md), deferred rather than admitted.
- **`transformers >= 5` returns a bare tensor from a block** where earlier versions returned a tuple
  `[E§2]`. Indexing `[0]` on the new return **silently takes the batch dimension** instead of the
  hidden states, producing a wrong-shaped result rather than an error. `block_map` handles both
  returns, and the suite pins both paths `[code: tests/test_jacobian.py]`.

---

## 11. What would falsify this

Every claim still standing, and the specific measurement that would kill it. A claim whose falsifier
cannot be named does not belong in this table.

| Live claim | Where | The measurement that kills it |
|---|---|---|
| Folding occurs: two semantically distinct contexts land on one state | [§3](#3-from-folding-to-confabulation) | An ε-fold search: pairs far apart in context and close in state, reported as the **ratio** of fold density against three controls run first — random pairs at matched ε, paraphrase pairs which *should* collide, and the same search at the embedding layer. If grounded fold density does not exceed the paraphrase control, the mechanism is not operating at that layer and depth. **Not built** |
| The fold has no crease, so watching `det J` is wrong | [§1.4](#14-why-a-vanishing-determinant-is-the-wrong-place-to-look) | The same search, conditioned on Jacobian conditioning. If every near-collision sits where the Jacobian is ill-conditioned, the classical crease was the right target and §1.4 is wrong. **Not built** |
| Information is actually lost at the fold | [§3.2](#32-what-the-mechanism-forbids) | A decoder `h -> context` against two controls: a capacity-matched decoder on shuffled labels for the chance floor, and the same decoder at an earlier layer for the mixing gradient. The mechanism predicts an irreducible error floor that survives added capacity. Without both controls a high error rate measures the decoder, not the model. **Not built** |
| Grounded text is less chaotic and contracts more volume than its shuffled control | [§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model) | Repeat the 46-step token product over **≥ 20 distinct passages and ≥ 3 seeds** and bootstrap a CI on the `sum` separation. If that CI crosses zero, the six same-sign numbers were one trajectory pair's luck. Sign agreement across six correlated summaries of *one* pair is not independent evidence |
| `D_KY` compresses meaningfully and separates conditions | [§9.3](#93-attractor-dimension-per-block) | The same multi-passage protocol on `D_KY`. With cv `0.6801` across blocks and \|mean\| separation `0.0758` at 5-of-6 sign agreement, a bootstrap CI crossing zero would end it. A second killer: measure at `D = 1024` and `D = 1280`; if the block-1 compression ratio moves with width it is not a constant of the architecture |
| `tail_alpha` is a property of the architecture, not the content | [§10.4](#104-the-salvage-tail_alpha-is-structural-not-semantic) | Re-run blocks 1–5 on **held-out text**, with block 0 included and excluded, both stated. If the grounded–shuffled difference exceeds the grounded sd `0.0293`, the 1.4% agreement was an artefact of one passage |
| `tail_alpha` can serve as a pruning budget, `C1` | [§10.4](#104-the-salvage-tail_alpha-is-structural-not-semantic) | Measure the exponent at `D = 1024` and `D = 1280` with the bulk window made **relative to `D`**. If it drifts by more than the grounded cv `0.124`, it is not the width-invariant constant `C1` requires. A second killer: the exponent-derived budget failing to land between random selection and oracle top-`k` at matched density |
| The exact Jacobian beats the Krylov estimator, so iterative spectral defaults are wrong, `C2` | [§9.1](#91-the-cost-of-working-in-jacobian-space) | Run the same cost probe across widths. If the ordering inverts at or below the widths real models use, `C2` has no addressable range and is a distilgpt2 curiosity. A second killer, cheaper than any measurement: find that mainstream libraries already gate on problem size — that novelty check has **not been run** |

Two further falsifiers apply to structure rather than to a number:

- **The Oseledets bridge is already falsified and the falsification is published.**
  [§10.2](#102-the-oseledets-bridge-buys-nothing-on-a-real-spectrum) is what the "run the sweep on
  real spectra" falsifier returned. What would *revive* it is a spectrum with a genuine plateau in
  the tolerance sweep and multiplicities above 1 at a tolerance chosen before seeing the answer —
  most plausibly in a model with tied or structured weights, where a symmetry forces an invariant
  subspace. Note that even a revival is capped by [§7.4](#74-why-the-bridge-is-structurally-empty):
  the most it can ever buy is the multiplicity vector.
- **The detector arm.** Any J-space score failing to beat Mahalanobis on **both** AUROC and cost per
  token, against real factual-error labels, is a negative result. The baselines are to be
  implemented *before* the J-space scores, because baselines written second get written to lose.

**What a shuffled-token control can and cannot say.** Every separation reported above is against one
control: the same token multiset in shuffled order. It holds the embedding statistics, token
frequencies and sequence length fixed and destroys only grounded structure — real work, and strictly
less work than the headline question needs.

| Question | Does the shuffled control answer it | Why |
|---|---|---|
| Does J-space distinguish coherent from scrambled text | **Yes**, at `n = 1` | That is precisely what the control varies |
| Does J-space distinguish in-distribution from out-of-distribution input | Partially | Scrambling is one OOD direction among many |
| **Does J-space predict factual error** | **No** | A fluent, in-distribution, confidently wrong generation is nothing like a scrambled one. The control never produces one |

**The shuffled-token control is an out-of-distribution control, not a hallucination control.**
Nothing in this repository speaks to factual error yet. The word "hallucination" appears in the
keywords because it names the target of the programme, not a result.

---

## 12. Prior art

Where this sits, and — more importantly — where the alternatives are better. **No score column
appears in this table.** No detection benchmark has been run in this repository, so there is no
AUROC measured here to put in one, and importing a number from another setting to fill the column
would make the table look decisive while measuring nothing. A table with a gap in it is the honest
object.

| Approach | Which linearization | Object it reads | Forward-causal by construction | Can it see a creaseless fold | Where it beats this work |
|---|---|---|---|---|---|
| Persistent homology as ML features | global | point cloud of hidden states | no — a function of the past trajectory | in principle yes, in practice untested | mature tooling; reaches H₁/H₂, where this repository computes no homology at all |
| Semantic entropy for hallucination detection | none — sampling, not geometry | distribution over resampled generations | n/a | indirectly: a fold shows up as answer instability | targets factual error directly and needs no white-box access |
| Mahalanobis distance on the final hidden state | local — one Gaussian | one hidden vector | no | no — a fold can land inside the density | far cheaper, and the arm to beat; **not beaten here, and not benchmarked against here** |
| PCA reconstruction error | global linear subspace | one hidden vector | no | no — both preimages can be in-subspace | far cheaper; strong on OOD; same status |
| Condition number, smallest singular value | local | one Jacobian | yes | **no, and this is §1.4's point** | cheap, and correct for the classical crease |
| Dynamical isometry, mean-field signal propagation | local | Jacobian singular spectrum, at initialization | yes | no — an initialization-time, ensemble-level statement | same object, established theory, produces architecture design rules |
| Intrinsic-dimension estimation | across scale | point cloud of hidden states | no | no | one scalar, very cheap, no autodiff required |
| **`caustic`** | local, composed along a trajectory, read as a filtration | `J_l(t)`, the transport operator | **yes** | **by design — and not yet attempted** | — nothing yet. §9.2 is a separation against a *scrambling* control only |

**The honest summary of this table:** the detector arm of this repository has not been run, so it has
beaten nothing. Mahalanobis and PCA on a single hidden vector are cheaper than anything here by
orders of magnitude, and the working assumption until measured otherwise is that they win. Semantic
entropy, persistent-homology feature pipelines and intrinsic-dimension estimators are not installed
and not run on this host.

---

## 13. Implementation map

The package ships flat at the repository root. There is no `src/` wrapper and no nested
`caustic/caustic/`.

| File | Responsibility | Propositions it must satisfy |
|---|---|---|
| [`caustic/jacobian.py`](caustic/jacobian.py) | `block_map` isolates the diagonal block of (5); `exact_jacobian` differentiates it by batched reverse-mode AD; `top_singular_values` is the Krylov path that never materializes `J`; `singular_values` is the exact reference | 7, 13 |
| [`caustic/spectrum.py`](caustic/spectrum.py) | `sigma_max`, `stable_rank`, `tail_alpha`, `log_volume`, `summarize` — equations (25) and (26). `BULK = (10, 400)` is exported so it can be overridden | 8, 9, 10, 11, 12 |
| [`caustic/cocycle.py`](caustic/cocycle.py) | `lyapunov_spectrum` — the QR iteration of (19), justified by Propositions 1 and 2 of §6. `finite_time_spectrum` returns the running exponents after **every** step, shape `(n, D)`, so a report can show whether a value had settled | 1, 2, 3, 4, 6 |
| [`caustic/oseledets.py`](caustic/oseledets.py) | `growth_filtration` — the gap encoding of §7.4. `filtration_entropy` — the diagnostic that decides whether the encoding bought anything. `tolerance_sweep` — because reporting one tolerance invites the assumption it was chosen after seeing the answer | 14, 15 |
| [`caustic/attractor.py`](caustic/attractor.py) | `kaplan_yorke_dimension` — equation (27); `metric_entropy` — the Pesin sum of positive exponents; `embedding_bound` — the Takens sufficient size; `spectrum_report` | 16, 17 |
| [`caustic/experiments/probe_cost.py`](caustic/experiments/probe_cost.py) | Cost of each estimator against the block forward pass as the unit; §9.1 | — |
| [`caustic/experiments/layer_sweep.py`](caustic/experiments/layer_sweep.py) | Spectral summaries across 6 blocks × 10 positions, grounded against a shuffled-token control; §10.1, §10.4 | — |
| [`caustic/experiments/lyapunov_llm.py`](caustic/experiments/lyapunov_llm.py) | Both products of §4.2 on a live model, against the same control; §9.2, §10.3 | — |
| [`caustic/experiments/oseledets_structure.py`](caustic/experiments/oseledets_structure.py) | The monotone tolerance sweep on measured spectra; §7.5, §10.2 | — |
| [`caustic/experiments/attractor_dimension.py`](caustic/experiments/attractor_dimension.py) | Per-block Kaplan-Yorke on the token product, against the same control; §9.3 | — |
| [`tests/test_jacobian.py`](tests/test_jacobian.py) | Ground truth for the estimator and the spectral summaries — 24 assertions | 7–13 |
| [`tests/test_cocycle.py`](tests/test_cocycle.py) | Systems whose exponents are known exactly — 8 assertions | 1–4, 6 |
| [`tests/test_oseledets.py`](tests/test_oseledets.py) | Filtration invariants, including the degeneracy the bridge must be honest about — 10 assertions | 14, 15 |
| [`tests/test_attractor.py`](tests/test_attractor.py) | Kaplan-Yorke against the Lorenz literature value — 12 assertions | 16, 17 |
| [`EVIDENCE.md`](EVIDENCE.md) | Every measured number, its control, and every negative | — |
| [`PLAN.md`](PLAN.md) | The thesis and the task-by-task programme, tests written before implementations | — |
| [`CANDIDATES.md`](CANDIDATES.md) | The move catalogue — one row per candidate, six mandatory columns, with killed (`K1`) and deferred (`D1`) entries kept | — |

**Why `finite_time_spectrum` exists.** The theorem's conclusions are asymptotic and the trajectories
here are not ([§5.4](#54-the-hypothesis-audit)). Returning the whole convergence trace is what let
[§10.3](#103-the-6-step-block-product-discarded-and-why) catch and discard a number that would
otherwise have been reported. It is the single most load-bearing design decision in the package, and
it is four lines of code.

---

## 14. Validation

**71 tests, every one against a closed-form answer rather than against self-consistency.** Each is a
proposition from [§8](#8-closed-form-propositions-the-test-suite-pins) whose proof is on this page.

```bash
python -m pytest --collect-only -q     # 71 tests collected
python -m pytest tests/ -q             # no model download, no GPU required
```

| Assertion | Proposition | Tolerance |
|---|---|---|
| Jacobian of a position-wise linear block equals its weight matrix | 7 | `1e-10` `[E§6]` |
| Power iteration matches exact `svdvals` | 13 | `1e-6` `[E§6]` |
| `tail_alpha` recovers synthetic exponents 0.25 / 0.5 / 1.0 / 2.0 | 8 | `1e-9` `[E§6]` |
| `log_volume` equals `torch.linalg.slogdet` | 11 | `1e-8` `[E§6]` |
| Flat spectrum returns exponent 0 *(negative control)* | 10 | `1e-9` `[E§6]` |
| Diagonal cocycle returns `log a` | 1 | `1e-8` `[E§6]` |
| Orthogonal cocycle returns 0 *(negative control)* | 2 | `1e-8` `[E§6]` |
| Exponents sum to mean `log\|det\|` | 3 | `1e-8` `[E§6]` |
| Scaling every Jacobian by `c` shifts every exponent by `log c` | 4 | `1e-9` `[E§6]` |
| Rank-deficient step drives one exponent below `−100`, others finite | 6 | — `[E§6]` |
| Degenerate spectrum gives one bar; distinct spectrum gives entropy `log D` | 14 | exact `[code]` |
| Wider tolerance never increases the bar count | 15 | exact `[code]` |
| Lorenz spectrum gives the textbook `D_KY = 2.0622` | 16 | `1e-3` `[code]` |
| `D_KY` unchanged by rescaling the spectrum, and bounded by `D` | 17 | rel. `1e-12` `[code]` |

Four of those rows are **negative controls** — inputs whose correct answer is "nothing here". A suite
without them can only fail by returning the wrong nonzero number, and the failure mode this pipeline
actually has is returning a plausible-looking nonzero number from noise. The rank-deficient row is
the folding case of [§1](#1-the-mechanism-drawn) and
[§6.4](#64-two-implementation-details-that-are-not-cosmetic): it must be a supported input producing
a readable answer, not a `NaN` and not an exception. The Lorenz row is the only assertion in the
suite checked against a value from the literature rather than one derived on this page, which is
what makes it the ground truth for [§9.3](#93-attractor-dimension-per-block).

**One recorded drift.** [`EVIDENCE.md`](EVIDENCE.md) §6 records **30** assertions, covering
`tests/test_jacobian.py` and `tests/test_cocycle.py` at the time it was written. `pytest
--collect-only -q` on this tree reports **54**: 24 in `test_jacobian.py`, 8 in `test_cocycle.py`, 10
in `test_oseledets.py` and 12 in `test_attractor.py`. `EVIDENCE.md` is behind the suite; the suite is
the source of truth, and the delta is stated here rather than reconciled silently.

There is no CI badge in this README because there is no CI in this repository. Correctness is claimed
by the assertions above and by the runnable commands in [§15](#15-quick-start), not by a green
square.

---

## 15. Quick start

```bash
git clone https://github.com/teerthsharma/caustic.git && cd caustic
pip install -e .                    # numpy + torch
pip install -e ".[experiments,dev]" # adds transformers and pytest
```

Every box above is reproduced by one of these commands.

```bash
python -m pytest tests/ -q                              # §14, closed-form assertions
python -m caustic.experiments.probe_cost                # §9.1, the cost table
python -m caustic.experiments.layer_sweep               # §10.1 negative, §10.4 salvage
python -m caustic.experiments.lyapunov_llm              # §9.2 converged, §10.3 discard
python -m caustic.experiments.oseledets_structure       # §7.5, §10.2 the tolerance sweep
python -m caustic.experiments.attractor_dimension       # §9.3, per-block Kaplan-Yorke
```

The experiments download `distilgpt2` on first run and take a few minutes each on the hardware in
[§16](#16-requirements-and-measurement-environment).

```python
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from caustic import exact_jacobian, singular_values, summarize
from caustic.attractor import spectrum_report
from caustic.cocycle import finite_time_spectrum, lyapunov_spectrum
from caustic.oseledets import filtration_entropy, growth_filtration

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    dtype=torch.float32,
    attn_implementation="eager",     # required; see §16
).eval()
for p in model.parameters():
    p.requires_grad_(False)

ids = tok("The capital of France is Paris.", return_tensors="pt").input_ids
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states

# One Jacobian, and the four summaries of equations (25) and (26).
J = exact_jacobian(model.transformer.h[3], hs[3].detach(), pos=-1)
print(summarize(singular_values(J).cpu().numpy()))

# The token product at a fixed block, and its convergence trace.
Js = [exact_jacobian(model.transformer.h[3], hs[3].detach(), t).cpu()
      for t in range(ids.shape[1])]
lam = lyapunov_spectrum(Js)
trace = finite_time_spectrum(Js)                 # (n, D) -- check it settled

print(f"lam1 {lam[0]:+.4f}  sum {lam.sum():+.2f}  positive {(lam > 0).sum()}/{len(lam)}")
print(f"last-step drift {abs(trace[-1, 0] - trace[-2, 0]):.4f}")

# Did the bridge of §7 buy anything? Entropy at log(D) means it did not.
bars = growth_filtration(lam)
print(f"{len(bars)} bars, entropy {filtration_entropy(bars):.4f}, log(D) {np.log(len(lam)):.4f}")

# The attractor dimension, and whether the formula saturated.
print(spectrum_report(lam))
```

Three lines to read before quoting anything:

- **Read the drift before quoting `lam1`.** That single line is the whole difference between
  [§9.2](#92-finite-time-qr-characteristic-exponents-of-a-live-model) and
  [§10.3](#103-the-6-step-block-product-discarded-and-why). A short passage gives a short product,
  and a short product gives you the §10.3 outcome with no warning other than that number.
- **Read the entropy before quoting the filtration.** If it sits at `log(D)`, the encoding carried
  nothing the sorted spectrum did not.
- **Read `sum` before quoting `kaplan_yorke`.** If `sum` is non-negative, `D_KY` is the width echoed
  back by a saturating formula, not a measured dimension — see
  [§9.3](#93-attractor-dimension-per-block).

And read `sum` in preference to any individual exponent: by
[§6.3](#63-the-frame-dependence-that-survives-at-finite-n) it is the only one that does not depend on
the choice $Q_0 = I$.

---

## 16. Requirements and measurement environment

| | | |
|---|---|---|
| **Python** | `>= 3.10` | `[cfg]` |
| **Core** | `numpy >= 1.24`, `torch >= 2.4` | `[cfg]` |
| **Experiments** | `transformers >= 4.40` | `[cfg]` |
| **Dev** | `pytest >= 8` | `[cfg]` |

Every figure in this README was produced on one machine. None is projected to any other.

| | |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB `[E§ header]` |
| **Platform** | Intel64, Windows 11 `[E§ header]` |
| **Python** | 3.11.9 `[E§ header]` |
| **Torch** | 2.5.1+cu121 `[E§ header]` |
| **Transformers** | 5.3.0 `[E§ header]` |
| **Precision / seed** | float32, seed 0 `[E§ header]` |
| **Model** | distilgpt2, `D = 768`, 6 blocks, `attn_implementation="eager"` `[E§ header]` |

**`attn_implementation="eager"` is required** `[E§2]`. `torch.func.jvp` raises
`NotImplementedError` on `_scaled_dot_product_efficient_attention`, the fused SDPA kernel
`transformers` selects by default. `top_singular_values` uses JVP and will fail without eager
attention. `exact_jacobian` uses reverse mode and is unaffected, but the experiments load eager
throughout so both paths are comparable on the same model.

**Determinism.** Anything feeding an A/B comparison runs under
`torch.use_deterministic_algorithms(True)` `[plan]`, and the suite asserts that two calls to
`exact_jacobian` on the same input are bitwise equal `[code: tests/test_jacobian.py]`.
Non-determinism in the measurement invalidates the measurement.

**Precision.** All product arithmetic is promoted to float64 `[code: caustic/cocycle.py]`. Given
[§6.1](#61-the-naive-product-collapses-onto-one-direction), running the accumulation in float32
would halve the number of steps before the frame becomes numerically degenerate, for no memory
saving worth having at this width.

**8 GiB is enough** for everything reported here, because a `768 × 768` float32 Jacobian is 2.25 MiB
and the products hold at most a few dozen of them on CPU. Nothing here has been run at a width where
that stops being true.

---

## 17. Limitations

Collected once, here, rather than scattered through the evidence sections.

**The central claim is untested.** [§3](#3-from-folding-to-confabulation) argues that folding forces
confabulation. [§11](#11-what-would-falsify-this) describes the measurements that would test it.
None of them exists in this repository. Every number reported here is equally consistent with
$\Phi_{\ell}$ being perfectly injective.

**The multiplicative ergodic theorem is not established for this object.**
[§5.4](#54-the-hypothesis-audit) is the statement, and it is the most serious mathematical limitation
on the page. The block product **is not a cocycle at all**, so Oseledets' theorem does not apply to
it in any form; the token product may be one, but no invariant measure has been exhibited, ergodicity
is therefore unposed, and the trajectory is 46 steps `[E§4]`. Everything reported is finite-time and
is named as such. The same gap applies to Takens' theorem behind
[§9.3](#93-attractor-dimension-per-block).

**Contraction is a precondition, not the mechanism.** By (15), `sum = -226.74` establishes only that
folding is not impossible `[E§4]`. A reader taking it as evidence of folding has over-read it. And
at block 0 of the attractor run the sum is `+117.3` `[A]`, so the precondition does not even hold
there.

**The persistence bridge is exact and structurally empty.** Proposition 4 is correct, and the barcode
(24) has no death times, so the encoding adds the multiplicity vector and nothing more; on a generic
spectrum that vector is all ones, and the measured `entropy/logD = 0.9986` `[E§5]` is what that
predicts. Stability transfers in neither direction
([§7.4](#74-why-the-bridge-is-structurally-empty), reason 2).

**Finite-time exponents other than the sum depend on the initial frame.**
[§6.3](#63-the-frame-dependence-that-survives-at-finite-n). `caustic/cocycle.py` fixes $Q_0 = I$
`[code]`, so the per-index numbers are properties of the residual-stream coordinate flag as much as
of the model.

**The detector question is open, and the honest prior is that it fails.** No run against real
factual-error labels exists. Every separation reported here is against a shuffled-token control,
which is an out-of-distribution control `[E§4]`. Mahalanobis and PCA are cheaper than anything in
this repository and have not been measured against it here.

**`n = 1` throughout** `[E§4]`. One passage, one model, one seed, no error bars across texts. The
separations of §9.2 are six numbers from a single trajectory pair, two of which come from the
discarded block product. They agree in sign, which is why they are reported; they are not a
statistical result. The attractor run of §9.3 is a second single passage, not a replication of the
first `[A]`.

**One model, one width.** distilgpt2 at `D = 768` `[E§ header]`. The cost inversion of §9.1, the
`0.2374` exponent `[E§3]` and every `D_KY` `[A]` are measured at that single width. The crossover
width at which the Krylov estimator starts to win is **not measured and must not be assumed**
`[E§1]`. `BULK = (10, 400)` is absolute `[code: caustic/spectrum.py]` and would have to be made
relative to `D` before any cross-width comparison is meaningful.

**One post-hoc decision is outstanding.** Block 0 was excluded from the `tail_alpha` fit after
inspecting the data `[E§3]`. It has not been re-derived on held-out text, so
[§10.4](#104-the-salvage-tail_alpha-is-structural-not-semantic) is a forking path until it is.

**Only the diagonal block is computed.** `J_l(t)` holds every other token position at its observed
value `[code: caustic/jacobian.py]`. The full `(TD × TD)` Jacobian, including the cross-position
terms attention creates, is not computed anywhere here — which also means the token product is a
product of *diagonal blocks*, not of the true one-step transport operator on the whole sequence, and
a fold living in those cross terms is invisible to every measurement in this repository.

**No homology is computed.** The topological half of the thesis is present as the structure theorem
of [§7.2](#72-the-structure-theorem-with-its-hypotheses) and Proposition 4, applied to a filtration
of $\mathbb{R}^D$. This repository computes no barcodes of any space.

**`EVIDENCE.md` is behind the code in two places**, both stated where they matter: the attractor run
of §9.3 is not recorded there at all, and its test count of 30 predates two test files
([§14](#14-validation)).

**Two novelty checks have not run.** `C1` and `C2` in [`CANDIDATES.md`](CANDIDATES.md) are both
marked `NOVELTY CHECK: NOT YET RUN`. Either could die to a search rather than to a measurement.

---

## 18. What is not built yet

Stated so the file table in [§13](#13-implementation-map) is not mistaken for a roadmap. Full task
breakdowns, with tests written before implementations, are in [`PLAN.md`](PLAN.md).

| Planned | What it would settle |
|---|---|
| `experiments/fold_search.py` | The ε-fold search of [§11](#11-what-would-falsify-this): whether folds occur above the paraphrase and random-pair base rates, and whether they sit at ill-conditioned Jacobians. **The measurement the mechanism stands or falls on** |
| `experiments/recoverability.py` | Whether a capacity-matched decoder hits an irreducible floor recovering context from `h`, against a shuffled-label chance floor and an earlier-layer control |
| `caustic/detect.py` + `experiments/hallucination_auroc.py` | The only experiment that can confirm or kill the detector arm: real factual-error labels, with `max_softmax`, `mean_entropy`, Mahalanobis and PCA baselines implemented **first**, so they are not written to lose |
| A stationary base: exponents averaged over a corpus rather than one prompt | The only route to hypothesis (H1) of [§5.1](#51-hypotheses). Without it every exponent here is a statistic of one passage, not of the model |
| A multi-passage, multi-seed run of both the token product and `D_KY` | Whether the six same-sign separations of §9.2 and the 5-of-6 sign agreement of §9.3 survive a bootstrap CI — the difference between a direction and a result |
| `experiments/width_invariance.py` | Whether `tail_alpha` and the block-1 compression ratio are width-invariant across `D = 768 / 1024 / 1280` with a *relative* bulk window, and a re-derivation of the block-0 exclusion on held-out text |
| `experiments/prune_budget.py` | Whether the exponent buys wall-clock, against random selection, locality-only, and oracle top-`k` at **matched density**. The position between random and oracle is the result; a speedup number alone is not |
| A test for Proposition 5, and an `EVIDENCE.md` entry for §9.3 | The one lemma in §8 stated but not pinned, and the one measurement on this page with no evidence-file entry behind it |

The evidence standard those tasks inherit is one sentence: **a claim needs a control.** It is why
[§2](#2-the-ledger) is a table with a control column rather than a paragraph, why
[§10](#10-negative-results) sits above the quick start, and why the numbers in §10.3 are printed at
all rather than deleted.

---

## License

MIT. See [`LICENSE`](LICENSE) and the declaration in [`pyproject.toml`](pyproject.toml).

<p align="center">
  <strong>MIT © <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong><br>
  <em>A caustic is where a map folds and distinct preimages merge.<br>
  The classical one has a crease. The one worth hunting does not.</em>
</p>
