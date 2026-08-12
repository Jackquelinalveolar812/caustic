# Results

Every number measured in-process. Each states what it was compared against; a
figure without a control is not a result and does not appear here.

**Hardware** NVIDIA GeForce RTX 4060 Laptop, 8 GiB · Windows 11
**Software** Python 3.11.9 · PyTorch 2.5.1+cu121 · transformers 5.3.0 · float32 · seed 0
**Model** Qwen/Qwen2.5-0.5B — `D = 896`, 24 blocks, vocabulary 151,936

---

## 1. Coherence gates factual retrieval; length does not

The central measurement. Prefix length held fixed at **128 tokens**; only its
character varies. The prefix contains none of the answers and is identical across
every entity.

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

**Control:** identical token count across every row, so length is held constant by
construction. **The same 128 tokens produce accuracy 1.000 or 0.000 depending only
on whether they form language.**

Incoherent context does not merely fail to help. It merges all twenty countries
onto a single answer — a largest orbit of 20, worse than the orbit of 4 with no
prefix at all.

The model knows every one of these facts. Reaching them is a property of the
surrounding text.

**Caveat.** `capital` and `language` disagree on the shuffled condition, 0.000
against 1.000. The boundary between *coherent* and *merely lexically diverse* is
not settled by this measurement.

## 2. The partition survives once it resolves

Adjusted Rand index between the entity partitions at different prefix lengths,
computed over all entities regardless of correctness, so no class can collapse.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital        ARI      answers changed
    0 -> 32   -0.0243                0.600
    0 -> 128   0.0000                0.450
  128 -> 512   1.0000                0.150
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Below the threshold the partition is destroyed and rebuilt. Above it the partition
is **exactly** preserved while individual answers still move — structure held,
labels free.

## 3. Equivariance detects the failure with no ground truth

Two quantities computed only from the model's own outputs under transformations of
its own input. Neither consults a correct answer.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  invariance (self-consistency)   collision (equivariance)
    capital              AUROC 0.859                 AUROC 0.995 [0.97, 1.00]
    language             AUROC 0.942                 AUROC 0.950 [0.83, 1.00]
    pooled, n = 32       AUROC 0.920                 AUROC 0.945 [0.85, 1.00]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Precondition, and it is not optional.** Equivariance requires the relation to be
injective. On a many-to-one relation — many countries share a continent — distinct
entities *should* share an answer, and the signal inverts to 0.273. `RelationSpec`
records injectivity so the precondition cannot be forgotten.

**Credit where due.** The invariance half is close to published self-consistency
work and is included as the baseline. The equivariance half is what reaches 0.995.

Adding a third symmetry does not help: an inversion test scores 0.5686 pooled and
exactly 0.5000 on `capital`, and the unweighted sum of all three scores 0.9451 —
identical to equivariance alone.

## 4. Errors are not information loss

Measured on items the model answers **wrongly**.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  entity still linearly recoverable from h_22        0.9624   (chance 0.0312)
  correct answer token recoverable from h_22         1.0000   (chance 0.0400)
  median rank of the correct answer            3 of 151,936
  correct answer within the top 10                 88 / 100
  correct answer within the top 1000              100 / 100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Grouped cross-validation, so no template appears in both folds. Recoverability on
**wrong** items (0.9624) exceeds that on correct ones (0.8981).

The context survives. The answer survives, near the top. Nothing was destroyed —
the right answer lost a competition. This is what Theorem 2 addresses and what
Theorem 5 explains the absence of a local signal for.

## 5. Where the answer appears

Median rank of the correct answer, each layer read through the final norm and
unembedding.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  layer      all      correct       wrong
     20    30552        29594       31626
     22        7            3          12
     24        2            1           3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The answer materialises across two layers, late and abruptly. It does not
degrade afterwards.

**Caveat.** The lens applies the final norm to intermediate states that were not
trained to be read through it, so ranks before layer 24 are indicative. Layer 24
is exact — it reproduces the model's own output.

## 6. Dynamics

Characteristic exponents of the token-position Jacobian product, block 3, 46 steps,
`D = 768`, against a shuffled-token control.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                lambda_1      sum      expanding   last-step drift
  grounded       +0.1653  -226.74     139 / 768            0.0012
  shuffled       +0.1852  -170.37     151 / 768            0.0003
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Positive leading exponent with a strongly negative sum: dissipative dynamics on a
low-dimensional attractor. This is the measured input to **Theorem 4**.

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

Block 1 places 768 dimensions of transport on a 29.57-dimensional attractor.
`D_KY` varies across depth with `cv 0.6801`, so it is **not** a width-invariant
constant, and the block-0 value saturates the formula rather than measuring a
dimension.

## 7. Cost

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block forward                    0.588 ms
  full 768 x 768 exact Jacobian   53.694 ms      91.3x forward
  top-8 Krylov, 20 iterations   1428.938 ms    2429.5x forward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The exact Jacobian is **26.6x cheaper** than the Krylov estimator of its own top
eight singular values, which it matches to `1.685e-04`. Batched reverse-mode AD
vectorises across all outputs; `k`-column power iteration runs `k x iters`
sequential passes. The crossover width is not measured and is not assumed.

The shipped detector costs **five forward passes** and no Jacobian at all.

## 8. Validation

**137 tests, every one against a closed-form or independently computed answer.**

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

## 9. Theoretical results: what the method certifies

Theorem 1 turns the partition into a **lower bound on the error rate that requires
no ground truth**. For an injective relation on `n` entities producing `m` distinct
answers, the certified error floor is `(n - m) / n`.

Every row below is computed by `caustic.theorems.certified_error_floor`, and the
measured error is shown beside it. The theorem asserts floor <= measured, and it
holds on every row with slack.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  relation   condition       n    m   certified floor   measured error
  capital    no prefix      20   15             0.250            0.450
  capital    prose 128      20   20             0.000            0.000
  capital    " the" x128    20    1             0.950            1.000
  language   no prefix      12    8             0.333            0.500
  language   prose 128      12   12             0.000            0.250
  language   " the" x128    12    2             0.833            1.000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Certified reduction

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital,  no prefix -> prose            +0.250   certified error removed
  language, no prefix -> prose            +0.333   certified error removed
  capital,  " the" x128 -> prose          +0.950   full measured span
  capital,  no prefix -> " the" x128      -0.700   certified error ADDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

On `capital`, the choice of prefix moves the **provable** error floor across a
**95-point range** — from 0.950 under a degenerate prefix to 0.000 under coherent
prose, at identical token count. The fourth row is not a footnote: a prefix can add
0.700 to the certified floor, and `repair_by_context` reports that case as
`WORSENED` rather than quietly returning a number.

### Downstream ceiling, from Theorem 2

A pooled block of size `k` caps **any** downstream recovery of the entity at `1/k`.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  largest orbit 20 (" the" x128)   recovery ceiling  0.05
  largest orbit  1 (prose 128)     recovery ceiling  1.00      20x gain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

No downstream component recovers what a pooled block destroyed, however capable it
is. That is what makes collapse worth detecting rather than merely scoring.

### What these numbers are, and are not

They are bounds on a **certificate**, measured on one model and two injective
relations. `+0.250` means a quarter of the answers were provably wrong before the
intervention and none are provably wrong after — not that the true error rate fell
by exactly that much. On `capital` the true error happened to fall further, 0.450
to 0.000; on `language` it fell from 0.500 to 0.250 while the certified floor went
to zero, which is the bound behaving exactly as a bound should.

The certificate is one-sided by construction. It can prove a model wrong. It can
never prove a model right.

## Limits

One model at one width. Two injective relations, 12 and 20 entities. One
distractor passage per condition, one seed. Answers compared by top-1 token, so a
correct answer phrased differently counts as disagreement. Injectivity is a
required precondition and was diagnosed after observing the failure on a
many-to-one relation, not predicted in advance.

Whether coherence-gated retrieval is a general property of language models or a
behaviour of a 0.5B model outside its training regime is **not** established here.
