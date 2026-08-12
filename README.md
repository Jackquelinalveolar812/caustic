<h1 align="center">Caustic</h1>

<p align="center">
  <strong>A hallucination detector and repair for language models, built from the orbit partition of a relation, with five proved bounds behind it.</strong><br>
  A model can know a fact and still be unable to reach it. When it cannot, distinct entities collapse onto one answer.<br>
  That collapse is a topological invariant, it is measurable with no ground truth, and it is bounded from below by a theorem.
</p>

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong> ·
  <a href="https://github.com/teerthsharma/caustic">github.com/teerthsharma/caustic</a> ·
  <em>teerths57@gmail.com</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&color=00aaff" alt="MIT"></a>
  <a href="#4-the-five-theorems"><img src="https://img.shields.io/badge/theorems-5%20proved-blueviolet?style=flat-square" alt="Theorems"></a>
  <a href="#3-detection-without-ground-truth"><img src="https://img.shields.io/badge/detector-5%20forward%20passes%2C%20no%20Jacobian-00aaff?style=flat-square" alt="Detector"></a>
  <a href="#3-detection-without-ground-truth"><img src="https://img.shields.io/badge/equivariance%20AUROC-0.995-brightgreen?style=flat-square" alt="AUROC"></a>
  <a href="#14-limits"><img src="https://img.shields.io/badge/precondition-injective%20relations%20only-orange?style=flat-square" alt="Precondition"></a>
  <a href="#11-validation"><img src="https://img.shields.io/badge/tests-137%20closed--form-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/core%20deps-numpy%20%2B%20torch-lightgrey?style=flat-square" alt="Deps"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-yellow?style=flat-square" alt="Python"></a>
</p>

---

## Abstract

**Caustic** detects and repairs a specific, measurable failure of factual retrieval in a
language model, and proves what the detection certifies.

The failure is this. A model can hold a fact and still be unable to reach it, and whether it
can is decided by the character of the surrounding text rather than by its length. Holding a
prefix at exactly 128 tokens and varying only what those tokens are, accuracy on twenty
country–capital pairs is **1.000** when the prefix is coherent prose and **0.000** when it is
one token repeated 128 times. The knowledge does not move between those two rows. The regime
does.

The failure has a shape, and the shape is what makes it detectable. Under an unreachable fact
the model does not answer randomly — it answers *identically*, sending distinct entities to
one answer. The object that records this is the **orbit partition**: the partition of entities
by the answer they receive, which is `H0` of the answer-equivalence relation. It needs no
correct answer to compute, and once it resolves it is exactly preserved as context grows —
adjusted Rand index `1.0000` from 128 to 512 tokens, while 15% of the individual answers still
change.

Three things follow from the partition. It is **detectable without ground truth**: an
equivariance score computed from five forward passes, consulting no answer key and no
Jacobian, reaches **AUROC 0.995** on an injective relation. It is **bounded**: for an
injective relation on `n` entities producing `m` distinct answers, at least `n − m` answers
are provably wrong, and a pooled block of `k` entities caps *any* downstream recovery at
`1/k`. It is **repairable**: `repair_by_context` measures its own effect size — `0.550 →
1.000` on `capital` — and flags when a prefix makes matters worse, which is a real observed
outcome rather than a defensive branch.

Five theorems stand behind those claims, one per branch of mathematics, each with a proof and
an executable witness: topology, game theory, differential geometry, chaos theory, and a
partial-differential-symmetry no-go which shows that no pointwise function of the Jacobian —
determinant, smallest singular value, condition number, spectral decay — can detect pooling at
all.

**Keywords:** orbit partition, `H0`, answer-equivalence relation, equivariance, invariance,
group action on a fact, pooling equilibrium, signalling game, orbit error bound, certified
error, no-go theorem, local diffeomorphism, global injectivity, volume contraction,
characteristic exponent, Kaplan–Yorke dimension, dissipative dynamics, logit lens, linear
probe, adjusted Rand index, AUROC, hallucination detection, retrieval regime, context
coherence, inference-time repair

---

## Table of contents

| § | Section | What is in it |
|---|---|---|
| [1](#1-the-failure-measured) | The failure, measured | The same 128 tokens, accuracy 1.000 and 0.000 |
| [2](#2-the-orbit-partition) | The orbit partition | The shape of the failure, and that it is stable |
| [3](#3-detection-without-ground-truth) | Detection without ground truth | Equivariance, AUROC 0.995, and its precondition |
| [4](#4-the-five-theorems) | **The five theorems** | Full statements, proofs, and the chain between them |
| [5](#5-theoretical-results) | **Theoretical results** | What is certified, and what is not claimed |
| [6](#6-repair-and-its-own-effect-size) | Repair | The intervention, and the flag for when it backfires |
| [7](#7-what-the-model-holds-when-it-is-wrong) | Inside a wrong answer | The answer is present, ranked 3rd of 151,936 |
| [8](#8-dynamics-the-measured-input-to-theorem-4) | Dynamics | Contraction, attractor dimension, the input to Theorem 4 |
| [9](#9-cost) | Cost | Why the shipped detector is five forward passes |
| [10](#10-quick-start) | Quick start | Install, test, and a runnable example |
| [11](#11-validation) | Validation | 137 tests, each against a closed form |
| [12](#12-implementation-map) | Implementation map | File-by-file responsibility |
| [13](#13-measurement-environment) | Measurement environment | The one host every number came from |
| [14](#14-limits) | **Limits** | Collected once, at the end |

Every number on this page is measured, and appears in [`RESULTS.md`](RESULTS.md) with the
control it was compared against. There is no CI badge, because there is no CI: correctness is
argued from 137 closed-form assertions ([§11](#11-validation)), which a green check cannot
supply.

---

## 1. The failure, measured

Prefix length held fixed at **128 tokens**. Only the character of those tokens varies. The
prefix contains none of the answers and is identical across every entity, so it carries no
task information — the control is built into the design rather than argued for afterwards.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital, 20 entities        accuracy   distinct answers   largest orbit
    no prefix                    0.550                 15               4
    coherent prose, 128 tok      1.000                 20               1
    shuffled words, 128 tok      0.000                  1              20
    " the" x 128                 0.000                  1              20
    random token ids, 128 tok    0.100                  3              18

  language, 12 entities
    no prefix                    0.500                  8               5
    coherent prose, 128 tok      0.750                 12               1
    " the" x 128                 0.000                  2               7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Control:** identical token count in every row, so length is held constant by construction
and is not available as an explanation. **The same 128 tokens produce accuracy 1.000 or 0.000
depending only on whether they form language.**

Incoherent context does not merely fail to help. It merges all twenty countries onto a single
answer — a largest orbit of 20, worse than the orbit of 4 with no prefix at all. A prompt
whose prefix is the word `" the"` one hundred and twenty-eight times is a well-formed input, a
valid tokenisation, and an unanswerable question.

The model knows every one of these facts. Reaching them is a property of the surrounding text.

**Caveat.** `capital` and `language` disagree on the shuffled condition, 0.000 against 1.000.
The boundary between *coherent* and *merely lexically diverse* is not settled by this
measurement.

---

## 2. The orbit partition

Under the failure the model does not become noisy. It becomes constant. Distinct entities
receive one answer, and the object that records exactly that is the partition of entities
induced by the answer map `f`, whose blocks are the classes of

$$e_1 \sim e_2 \iff f(e_1) = f(e_2)$$

This is `H0` of the answer-equivalence relation "these two entities receive the same answer" —
its connected components. Three properties make it the right instrument:

- **It needs no ground truth.** The partition is computed from the model's own answers over
  all entities, whether those answers are right or wrong.
- **It cannot be confounded by accuracy.** Every entity is included regardless of correctness,
  so no class can collapse and no comparison inherits a moving base rate.
- **It is what the theorems constrain.** `n − m` and `1/k` in [§4](#4-the-five-theorems) are
  both statements about blocks of this partition.

`OrbitReport` carries it: `n_distinct` is the number of blocks, `largest_orbit` the size of
the biggest one, `collapse_ratio` that size over `n`, and `collapsed` is the boolean a caller
acts on.

### 2.1 The partition survives once it resolves

Adjusted Rand index between the entity partitions at different prefix lengths, computed over
all entities regardless of correctness.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital        ARI      answers changed
    0 -> 32   -0.0243                0.600
    0 -> 128   0.0000                0.450
  128 -> 512   1.0000                0.150
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Below the threshold the partition is destroyed and rebuilt. Above it the partition is
**exactly** preserved while individual answers still move — structure held, labels free. The
answer-churn column is what makes an ARI of `1.0000` evidence of stability rather than
evidence that nothing happened.

---

## 3. Detection without ground truth

A fact carries a group action, and a model that holds the fact must respect two halves of it:

```
  invariance     paraphrase the prompt  ->  the answer must NOT change
  equivariance   swap the entity        ->  the answer MUST change
```

A model outside its retrieval regime fails the second while passing the first: it is invariant
where it should be equivariant, returning the same answer whichever entity is named. Both
quantities are computed from the model's own outputs under transformations of its own input.
Neither consults a correct answer, which is the only condition under which a hallucination
detector is of any use.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  invariance (self-consistency)   collision (equivariance)
    capital              AUROC 0.859                 AUROC 0.995 [0.97, 1.00]
    language             AUROC 0.942                 AUROC 0.950 [0.83, 1.00]
    pooled, n = 32       AUROC 0.920                 AUROC 0.945 [0.85, 1.00]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

`symmetry_scores` returns both per entity, plus their difference. `invariance` is agreement
across paraphrases of the same fact, and high is good; `collision` is the fraction of *other*
entities receiving this entity's answer, and low is good. Five templates per relation means
five forward passes per entity, and no derivative of anything.

**Credit where due.** The invariance half is close to published self-consistency work and is
included as the baseline the other half has to beat. The equivariance half — penalising a
model for giving the *same* answer to different entities — is what reaches 0.995.

Adding a third symmetry does not help: an inversion test scores 0.5686 pooled and exactly
0.5000 on `capital`, and the unweighted sum of all three scores 0.9451, identical to
equivariance alone.

### 3.1 The precondition, and it is not optional

Equivariance requires the relation to be **injective**: distinct entities must genuinely
warrant distinct answers. On a many-to-one relation — many countries share a continent —
distinct entities *should* collide, and the signal inverts to **0.273**. Point it at such a
relation and you do not have a broken detector; you have a working detector with the sign
reversed, which is worse.

`RelationSpec.injective` records the precondition so it cannot be forgotten.
`OrbitReport.certified_errors` returns 0 whenever it is False, and `collapsed` is then always
False, because sharing an answer is not evidence of error there.

---

## 4. The five theorems

Five results, one per branch, each with a proof and an executable witness in
[`caustic/theorems.py`](caustic/theorems.py). Every statement is elementary, and that is
deliberate: the value is not in the difficulty of the proofs but in the fact that each one is
checkable against a measurement made in this repository, and that together they chain.

```
   differential geometry ──┐
                           ├──▶  topology  ──▶  game theory
   chaos theory ───────────┘

   partial-differential symmetry ──▶  why the differential route fails alone
```

Read: vanishing entity coupling *causes* pooling (differential geometry feeds topology);
volume contraction *drives* pooling (chaos theory feeds topology); pooling *bounds every
downstream reader* (topology feeds game theory); and no pointwise Jacobian statistic can
substitute for the topological measurement (the no-go).

**Notation.** `E` is a finite set of entities, `|E| = n`. A relation `R : E → A` is
*injective* when distinct entities have distinct correct answers. A model induces `f : E → A`,
and `P(f)` is the partition of `E` by the value of `f`, with `m = |P(f)|` blocks called
*orbits*.

### Theorem 1 (topology) — Orbit Error Bound

For injective `R`, the number of entities on which `f` errs, written `err(f)`, is at least
`n − m`:

$$\mathrm{err}(f) \;\geq\; n - m$$

**Proof.** The true map `R` is injective, so distinct entities carry distinct correct answers.
If `e1 ≠ e2` lie in one orbit then `f(e1) = f(e2)` while `R(e1) ≠ R(e2)`, so `f` is wrong on
at least one of them. An orbit of size `s` therefore contributes at least `s − 1` errors, and
summing over the `m` orbits gives `sum_i (s_i − 1) = n − m`. ∎

**Tight:** attained exactly when every orbit contains one correct answer.

**Measured instance.** `capital` with no prefix gave `n = 20`, `m = 15`, certifying at least
5 wrong answers with no ground truth consulted. Observed accuracy 0.550, so 9 were actually
wrong and the bound held with slack.

### Theorem 2 (game theory) — Pooling Recovery Bound

If `f` maps a block of `k` entities to one answer, then for **any** downstream function `h`,
the probability that `h(f(e))` recovers `e` under a uniform prior on that block is at most
`1/k`:

$$\Pr[\, h(f(e)) = e \,] \;\leq\; \frac{1}{k}$$

**Proof.** `h ∘ f` is constant on the block, so it takes one value there. It can therefore
agree with the identity on at most one of the `k` entities. Under a uniform prior the success
probability is at most `1/k`. ∎

This is the signalling-game reading: a block of size `k > 1` is a **pooling equilibrium**, and
pooling destroys the receiver's ability to infer the sender's type. No amount of downstream
capability recovers it — not a larger model above that layer, not a longer chain of thought,
not a better decoder — which is why orbit collapse is not merely an error but an unrecoverable
one.

**Measured instance.** The `" the" x 128` prefix pooled all 20 countries, bounding any
downstream recovery at **0.05**.

### Theorem 3 (differential geometry) — Zero Coupling Implies Pooling

Let `z_c(h)` be the logit of token `c` as a function of the entity representation `h`,
continuously differentiable on a domain containing a path `γ` from `h1` to `h2`. If the
directional derivative of `z_c` along `γ` vanishes identically, then `z_c(h1) = z_c(h2)`:

$$z_c(h_2) - z_c(h_1) \;=\; \int_{\gamma} \nabla z_c \cdot d\ell \;=\; 0$$

**Proof.** The fundamental theorem of calculus along `γ`, as displayed. ∎

**Consequence.** If this holds for every candidate `c`, the two entities receive identical
logit vectors, hence identical answers, hence share an orbit, and Theorem 1 applies.
**Differential geometry feeds topology.** Vanishing entity coupling is a *sufficient*
condition for pooling — which is also why the topological measurement is the one to take: the
partition observes the consequence whether or not the coupling itself is measurable.

`path_integral_change` evaluates the integral numerically along the straight path, and the
suite pins it against linear and quadratic closed forms to `1e-6`.

### Theorem 4 (chaos theory) — Dissipative Pooling

Let `T` be differentiable with characteristic exponents `λ_1 ≥ … ≥ λ_D` whose sum
`S = Σ_i λ_i` is negative. Then for any bounded set `A` of positive Lebesgue measure, the
volume of its image contracts:

$$\operatorname{vol}(T^{n}(A)) \;\sim\; e^{nS} \;\longrightarrow\; 0$$

**Proof.** The change-of-variables formula gives `vol(T^n(A)) = ∫_A |det D(T^n)|`, and the
exponents are defined so that `(1/n) log |det D(T^n)| → Σ_i λ_i = S`. With `S < 0` the
integrand decays like `exp(nS)`, so the volume does. ∎

**Consequence.** Once the image volume falls below the resolution separating decision regions,
distinct inputs must land in one region and pool. Volume contraction is therefore a *driver*
of the collapse Theorem 1 penalises, and depth is the clock. **Chaos theory feeds topology.**

**Measured instance.** The token product at block 3 gave `S = −226.74` over 768 dimensions
with only 139 expanding directions, and `D_KY = 29.57` at block 1 — a 26x compression against
the width ([§8](#8-dynamics-the-measured-input-to-theorem-4)). What prevents total pooling in
practice is that the trajectory is short, not that the map is volume-preserving.

### Theorem 5 (partial-differential symmetry) — No Local Criterion Detects Pooling

There exists a smooth map `F` whose Jacobian is nonsingular at every point of its domain and
which is not injective. Consequently **no function of the local Jacobian alone** —
determinant, smallest singular value, condition number, spectral decay, or any other pointwise
invariant — can decide injectivity.

$$F(x, y) \;=\; (e^{x}\cos y,\;\; e^{x}\sin y)$$

**Proof by witness.** `det DF = e^{2x} > 0` everywhere, yet `F(x, y) = F(x, y + 2π)`. The
Jacobians at the two preimages are related by a rotation and therefore share every spectral
invariant, so no pointwise function of the Jacobian distinguishes the injective case from this
one. ∎

This is the honest statement of a moral the refutation of the Jacobian Conjecture made vivid:
local invertibility everywhere does not imply global injectivity, and the failure is invisible
to local data. It is a **no-go** result, and it is the only theorem here that tells you what
not to build. A detector watching a determinant, a smallest singular value or a condition
number for orbit collapse is watching a quantity that provably cannot see it.

**The escape is global, and Theorem 1 is what takes it:** compare *two* entities instead of
examining one point. That is why the shipped detector is five forward passes and no Jacobian
at all.

---

## 5. Theoretical results

What the method certifies, stated as bounds rather than as claims about the world.

### 5.1 What Theorem 1 certifies

For an injective relation on `n` entities whose answers induce `m` distinct answers, **at
least `n − m` of those answers are provably wrong**, and the proof consults no ground truth.
The input is the partition; the answer key never enters the argument.

The measured instance, and its repaired counterpart:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital, 20 entities      n     m   certified wrong   observed accuracy
    no prefix              20    15                 5               0.550
    coherent prose         20    20                 0               1.000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Two readings of that table, both required. The certificate is **sound**: 5 were certified
wrong, 9 were in fact wrong, and the bound held with slack. The certificate is **not tight in
practice**: it saw 5 of the 9. Theorem 1 counts wrong answers; it does not name them.
Declining to consult an answer key has a price, and that is the price.

After repair the partition is fully separated, so the certificate correctly certifies nothing,
and accuracy is 1.000 — the bound going to zero and the truth going to perfect, with accuracy
never entering the computation.

### 5.2 What Theorem 2 certifies downstream

A pooled block of size `k` caps **any** downstream recovery of the entity at `1/k`. This is
not a statement about the current decoder. It quantifies over every function of the answer,
present or future.

For the fully collapsed 20-entity partition produced by the `" the" x 128` prefix, `k = 20`
and the ceiling is **0.05**. Nothing placed after that point — a bigger model, a verifier, a
second pass, a retrieval step reading the answer — raises it, because what distinguished the
twenty entities is not in the answer to be read.

### 5.3 What this is not

These are bounds on a **certificate**, measured on **one model and two relations**. They are
not a claim about hallucination rates in general, in this model or in any other. Theorem 1
says what follows from a partition; it does not say how often such partitions arise in
deployment, and nothing here measures that. The bounds are exact; their scope is
[§14](#14-limits).

---

## 6. Repair, and its own effect size

The detector finds collapsed orbits. `repair_by_context` applies the intervention the
measurements point at, and reports the effect size rather than asserting it.

The intervention is to supply the regime: prepend coherent prose that contains none of the
answers, is identical across every entity, and is unrelated in subject.

```python
from caustic import NEUTRAL_PREFIX, repair_by_context

report = repair_by_context(spec, answer_fn, prefix=NEUTRAL_PREFIX, gold=gold)
print(report)
# largest orbit 20 -> 1, distinct answers 1 -> 20 of 20 entities,
# accuracy 0.000 -> 1.000  REPAIRED
```

`RepairReport` carries three verdicts and one delta:

| field | meaning |
|---|---|
| `repaired` | True only for a genuine collapsed → fully separated transition |
| `worsened` | True when the prefix **merged** entities that were previously separate |
| `accuracy_delta` | reported only when `gold` is supplied; the verdict never consults it |
| `before/after_largest_orbit`, `before/after_n_distinct` | the raw partition on both sides |

**`worsened` is not a defensive check.** An incoherent prefix of the same length drove the
largest orbit from 4 to 20 in the measurements this module is built on
([§1](#1-the-failure-measured)). A prefix can make things much worse, and a repair function
that cannot say so is a repair function that will eventually lie.

**Why this is not prompt engineering in the pejorative sense.** The prefix carries no
information about the task: it contains none of the answers, it is identical across entities,
and it is unrelated in subject. It cannot be leaking an answer, because the *same 128 tokens
in shuffled order* drive accuracy to zero. What it supplies is distributional, not
informational. `NEUTRAL_PREFIX` is 128 tokens on mechanical calculators, ocean currents,
language and photosynthesis — the least interesting paragraph in the repository, and on
`capital` it is worth 0.550 → 1.000.

Two honest constraints ship with it. If a caller's relation concerns oceans, looms or
photosynthesis then `NEUTRAL_PREFIX` is no longer neutral for them, which is why the prefix is
a parameter. And `sweep_prefixes` returns a report for *every* candidate prefix, because
quoting the best of several without reporting the rest is selection on the outcome.

**What this is not.** It is not a pretraining method. The pretraining analogue — augmenting
facts across varied coherent contexts so they are extractable from any of them — is documented
in the literature with far larger effects and is not implemented here. This is an
inference-time wrapper whose effect this module measures on the caller's own model and
relation.

---

## 7. What the model holds when it is wrong

Measured on items the model answers **wrongly**, with grouped cross-validation so that no
template appears in both folds.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  entity still linearly recoverable from h_22        0.9624   (chance 0.0312)
  correct answer token recoverable from h_22         1.0000   (chance 0.0400)
  median rank of the correct answer            3 of 151,936
  correct answer within the top 10                 88 / 100
  correct answer within the top 1000              100 / 100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Recoverability on **wrong** items (0.9624) exceeds that on correct ones (0.8981).

The context survives. The answer survives, near the top. Nothing was destroyed — the right
answer lost a competition. This is the state Theorem 2 addresses: the loss happens where
distinct entities are mapped onto one answer, not in the representation feeding that map. It
is also why Theorem 5 matters operationally. There is no local degeneracy at the moment of
failure to go looking for, because nothing locally degenerate has happened.

### 7.1 Where the answer appears

Median rank of the correct answer, each layer read through the final norm and unembedding.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  layer      all      correct       wrong
     20    30552        29594       31626
     22        7            3          12
     24        2            1           3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The answer materialises across two layers, late and abruptly. It does not degrade afterwards.

**Caveat.** The lens applies the final norm to intermediate states that were not trained to be
read through it, so ranks before layer 24 are indicative. Layer 24 is exact — it reproduces
the model's own output.

---

## 8. Dynamics: the measured input to Theorem 4

Characteristic exponents of the token-position Jacobian product, block 3, 46 steps, `D = 768`,
against a shuffled-token control that preserves the token multiset.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                lambda_1      sum      expanding   last-step drift
  grounded       +0.1653  -226.74     139 / 768            0.0012
  shuffled       +0.1852  -170.37     151 / 768            0.0003
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Positive leading exponent with a strongly negative sum: dissipative dynamics on a
low-dimensional attractor. That is exactly the hypothesis `S < 0` of **Theorem 4**, measured
rather than assumed, and supplying it to the chain is the whole job of this section. The drift
column is what makes the values quotable: each is small against the value it drifts on.

Kaplan–Yorke dimension per block, validated against the textbook Lorenz spectrum
`(0.906, 0, −14.572)` giving `2.0622`, reproduced to `1e-3`:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block    D_KY     D / D_KY    expanding
      1   29.57         26.0        9 / 768
      3  298.08          2.6      139 / 768
      5  674.67          1.1      354 / 768
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Block 1 places 768 dimensions of transport on a 29.57-dimensional attractor. `D_KY` varies
across depth with `cv 0.6801`, so it is **not** a width-invariant constant, and the block-0
value saturates the formula rather than measuring a dimension. Both facts are stated because
the number is otherwise easy to over-read: contraction is the hypothesis Theorem 4 needs, and
`D_KY` describes how strong it is, not a second invariant.

---

## 9. Cost

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block forward                    0.588 ms
  full 768 x 768 exact Jacobian   53.694 ms      91.3x forward
  top-8 Krylov, 20 iterations   1428.938 ms    2429.5x forward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The exact Jacobian is **26.6x cheaper** than the Krylov estimator of its own top eight
singular values, which it matches to `1.685e-04`. The estimator costs 26.6x more than the
quantity it was brought in to approximate, and it is not a broken estimator: batched
reverse-mode AD vectorises across all outputs, while `k`-column power iteration runs
`k × iters` sequential passes. The crossover width is not measured and is not assumed.

None of this is on the detector's path. **The shipped detector costs five forward passes and
no Jacobian at all**, which Theorem 5 predicts in advance: the Jacobian could not have
supplied the missing signal at any price.

---

## 10. Quick start

```bash
git clone https://github.com/teerthsharma/caustic.git
cd caustic
pip install -e ".[dev]"          # numpy, torch, pytest
python -m pytest -q              # 137 passed, no model download, no GPU
```

The detector, the repair and the theorems need only `numpy`. A live model needs the extra:

```bash
pip install -e ".[experiments]"  # transformers
```

### 10.1 Detect and repair, end to end

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from caustic import NEUTRAL_PREFIX, RelationSpec, orbit_partition, repair_by_context

MODEL = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()


def top1(prompt: str) -> int:
    """Prompt -> top-1 next-token id. Must be deterministic: a sampled answer
    makes the partition noise."""
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        return int(model(ids).logits[0, -1].argmax())


facts = {"France": "Paris", "Japan": "Tokyo", "Peru": "Lima",
         "Kenya": "Nairobi", "Norway": "Oslo"}

spec = RelationSpec(
    templates=("The capital of {e} is", "{e}'s capital is"),
    entities=tuple(facts),
    injective=True,          # distinct countries have distinct capitals
)

report = orbit_partition(spec, top1)
print(report)                      # n entities -> m distinct answers, largest orbit s
print(report.certified_errors)     # Theorem 1: n - m, proved, no gold consulted
print(report.collapsed)            # the boolean to act on

gold = {e: tok(" " + a, add_special_tokens=False).input_ids[0]
        for e, a in facts.items()}

print(repair_by_context(spec, top1, prefix=NEUTRAL_PREFIX, gold=gold))
# largest orbit s -> 1, distinct answers m -> 5 of 5 entities,
# accuracy a -> b  REPAIRED
```

`orbit_partition` uses the first template. `symmetry_scores(spec, top1)` uses all of them and
needs at least two, since invariance is undefined on one. Set `injective=False` for a
many-to-one relation and both `certified_errors` and `collapsed` correctly report nothing
([§3.1](#31-the-precondition-and-it-is-not-optional)).

### 10.2 The theorems, directly

```python
from caustic import orbit_error_bound, pooling_recovery_bound, winding_witness

orbit_error_bound(20, 15)                      # 5    -- Theorem 1, measured instance
pooling_recovery_bound(20)                     # 0.05 -- Theorem 2, collapsed ceiling
image, jac, det = winding_witness(0.0, 0.0)    # Theorem 5, nonsingular and many-to-one
```

### 10.3 Reproducing the measurements

Each experiment prints its own table and its own control.

```bash
python -m caustic.experiments.coherence_vs_length     # §1
python -m caustic.experiments.orbit_invariant         # §2.1
python -m caustic.experiments.symmetry_break          # §3
python -m caustic.experiments.answer_presence         # §7
python -m caustic.experiments.attractor_dimension     # §8
python -m caustic.experiments.probe_cost              # §9
```

---

## 11. Validation

**137 tests, every one against a closed-form or independently computed answer.** Collected
with `python -m pytest --collect-only -q`.

| assertion | tolerance |
|---|---|
| Jacobian of a position-wise linear block equals its weight matrix | `1e-10` |
| Krylov estimate matches exact `svdvals` | `1e-6` |
| power-law exponent recovered from a synthetic spectrum | `1e-9` |
| log-volume equals `torch.linalg.slogdet` | `1e-8` |
| diagonal cocycle returns `log a`; orthogonal returns `0` | `1e-8` |
| exponents sum to the mean `log\|det\|` | `1e-8` |
| Kaplan–Yorke reproduces the Lorenz value `2.0622` | `1e-3` |
| Theorem 1 bound never exceeds true error count | 2000 random instances |
| Theorem 2 bound beaten by no constant decoder | exhaustive |
| Theorem 3 path integral matches linear and quadratic closed forms | `1e-6` |
| Theorem 4 exponent sum equals `log\|det A^n\|` | `1e-9` |
| Theorem 5 both preimages share every spectral invariant | `1e-9` |
| flat spectrum returns exponent zero (negative control) | `1e-9` |
| partition is bitwise identical across repeated calls | exact |

Three of those rows are negative controls — inputs whose correct answer is "nothing here" —
because the failure mode of a spectral pipeline is not an exception, it is a plausible number
from noise. A passing suite is therefore a statement about the mathematics, not about the last
time the code changed.

---

## 12. Implementation map

The package ships flat at the repository root.

| file | responsibility |
|---|---|
| [`caustic/regime.py`](caustic/regime.py) | `RelationSpec`, `OrbitReport`, `orbit_partition`, `symmetry_scores` — the detector |
| [`caustic/repair.py`](caustic/repair.py) | `NEUTRAL_PREFIX`, `repair_by_context`, `sweep_prefixes` — the intervention and its effect size |
| [`caustic/theorems.py`](caustic/theorems.py) | the five statements, their proofs in the module docstring, and one executable witness each |
| [`caustic/jacobian.py`](caustic/jacobian.py) | `block_map`, `exact_jacobian`, `singular_values`, `top_singular_values` |
| [`caustic/cocycle.py`](caustic/cocycle.py) | `lyapunov_spectrum`, `finite_time_spectrum` — QR characteristic exponents of a matrix product, with the running trace so convergence is visible |
| [`caustic/attractor.py`](caustic/attractor.py) | `kaplan_yorke_dimension`, `embedding_bound`, `metric_entropy`, `spectrum_report` |
| [`caustic/spectrum.py`](caustic/spectrum.py) | `sigma_max`, `log_volume`, `stable_rank`, `tail_alpha` — scalar summaries of a singular spectrum |
| [`caustic/oseledets.py`](caustic/oseledets.py) | `growth_filtration`, `filtration_entropy`, `tolerance_sweep` |
| [`caustic/detect.py`](caustic/detect.py) | `auroc`, `auroc_ci`, and the Mahalanobis and PCA baselines any score must beat |
| [`caustic/experiments/`](caustic/experiments) | one runnable file per measurement, each printing its own control |
| [`tests/`](tests) | 137 closed-form assertions |
| [`RESULTS.md`](RESULTS.md) | every number on this page, with its control |

---

## 13. Measurement environment

Every number came from one host. None is projected past the measured range.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Hardware   NVIDIA GeForce RTX 4060 Laptop, 8 GiB · Windows 11
  Software   Python 3.11.9 · PyTorch 2.5.1+cu121 · transformers 5.3.0
             float32 · seed 0
  Model      Qwen/Qwen2.5-0.5B — D = 896, 24 blocks, vocabulary 151,936
  Dynamics   token-position Jacobian product, block 3, 46 steps, D = 768
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Requirements: Python `>= 3.10`, `numpy >= 1.24`, `torch >= 2.4`; `transformers >= 4.40` for
the experiments and `pytest >= 8` for the suite.

---

## 14. Limits

Collected once, here, rather than scattered through the sections above.

- **One model at one width.** Whether coherence-gated retrieval is a general property of
  language models, or a behaviour of a 0.5B model outside its training regime, is **not**
  established here.
- **Two injective relations**, of 12 and 20 entities. One distractor passage per condition,
  one seed.
- **Top-1 token comparison.** Answers are compared by their first token, so a correct answer
  phrased differently counts as disagreement, and both the partition and the accuracy column
  inherit that convention.
- **Injectivity is a required precondition**, not a convenience. On a many-to-one relation the
  equivariance signal inverts to 0.273, `certified_errors` is meaningless, and `collapsed` is
  correctly always False. The requirement was diagnosed by measurement rather than predicted
  in advance.
- **The bounds are bounds on a certificate.** Theorems 1 and 2 are exact and they constrain a
  partition; they say nothing about how often such partitions arise in deployment
  ([§5.3](#53-what-this-is-not)).
- **The logit-lens ranks before layer 24 are indicative**, since the final norm is applied to
  states not trained to be read through it. Layer 24 is exact.
- **`D_KY` is not a width-invariant constant** (`cv 0.6801` across depth), and the block-0
  value saturates the formula rather than measuring a dimension.
- **The two relations disagree on the shuffled condition** (0.000 against 1.000), so the
  boundary between coherent text and merely lexically diverse text is not resolved by these
  measurements.

---

## License

MIT. See [`LICENSE`](LICENSE) and the declaration in [`pyproject.toml`](pyproject.toml).

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong><br>
  <a href="https://github.com/teerthsharma/caustic">github.com/teerthsharma/caustic</a> ·
  <em>teerths57@gmail.com</em><br>
  <em>A caustic is where a map folds and distinct preimages merge.<br>
  The partition is how you see it without being told where to look.</em>
</p>
