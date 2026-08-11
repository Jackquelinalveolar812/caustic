# Candidates

Target: **100 screened candidates**, not 100 confirmed benefits. The measured
conversion record this is sized against is 601 pull requests authored, 109
merged, 16 merged into repositories not owned by the author, 7 of those into
major labs. A 100-candidate pool converting at that rate yields single-digit
confirmed wins, and this file is written to expect that.

## The composition every candidate must use

Each candidate applies **the three geometries plus one computational
mathematics** to a specific branch of computation. All five fields are required.
A candidate that cannot fill one of them is not a candidate.

| field | what it must name |
|---|---|
| **topology** | the global linearization — which connectivity, order, or interval structure is present. Homology is a functor to vector spaces; this column names the space |
| **differential geometry** | the local linearization — the tangent, Jacobian, metric, or curvature object |
| **fractal geometry** | the scale linearization — the exponent, self-similarity, or power law, and what stays invariant across scale |
| **computational mathematics** | the fourth pillar: complexity analysis, numerical linear algebra, graph algorithms, approximation theory, or information theory |
| **branch optimized** | the exact hot path, in an exact file, that gets faster or smaller |

## Admission gate

Six columns, all mandatory, or the candidate is not admitted:

1. **hot path** — exact repo, file and function, verified to exist today by fetching it
2. **latent structure** — order, connectivity, interval, or scaling exponent, named precisely
3. **current cost** — the quadratic or redundant scan, complexity stated exactly
4. **predicted win** — an exact formula, never an adjective
5. **control** — what it is measured against at equal budget
6. **novelty check** — the search that found no prior art, or the prior art it must beat

## Screening

Fresh-context adversarial verification, per the math-olympiad discipline: the
verifier never sees the argument that generated the candidate. Asymmetric voting
— four confirmations to admit, two refutations to kill. Calibrated abstention is
preferred over a guess.

## Off-limits upstream targets

Already contributed to or explicitly declined: pytorch/pytorch, mujoco,
mujoco_warp, triton, XNNPACK, vllm, rtp-llm, NVIDIA topograph / NeMo-Relay /
TensorRT-LLM / cosmos, penzai, Graphormer, pytorch_geometric, cugraph,
tensorflow, highway, zstd, fairchem, alphafold3, openxla/xla. Verify against the
live pull request list before proposing anything.

---

## Admitted

*(none yet — a candidate reaches this section only after four independent
confirmations and a measured control)*

## Under screening

### C2 — the estimator that is slower than the exact computation it replaces

**Status:** measured on one case, not yet generalized. This is the strongest seed
because the anomaly was found by running the code rather than by reasoning about
it.

| field | content |
|---|---|
| topology | the spectrum's rank structure — stable rank 9.69 of 768 means the operator is effectively low rank, so the "large matrix" premise behind iterative methods is false here |
| differential geometry | the object is the layer-to-layer Jacobian `J_l(t) = d h_{l+1}[t] / d h_l[t]` |
| fractal geometry | the spectrum decays as a power law with exponent 0.2374 +/- 0.0293 over blocks 1-5, which is what makes the effective rank small |
| computational mathematics | complexity crossover analysis — batched reverse-mode AD is `O(D)` backward passes with perfect vectorization, while `k`-column power iteration is `O(k * iters)` sequential passes with no batching |
| branch optimized | any spectral estimator that defaults to an iterative method regardless of problem size |

**Measured, RTX 4060 Laptop, distilgpt2 layer 3, float32:**

```
block forward             0.588 ms
full 768x768 jacrev      53.694 ms   (91.3x forward)
top-8 power, 20 iters  1428.938 ms   (2429.5x forward)
```

The exact Jacobian is **26.6x cheaper** than the Krylov estimator of its own top
eight singular values. The estimator is correct — it agrees with exact `svdvals`
to `1.685e-04` max relative error — it simply loses at this width.

**Predicted win:** for `D` below the crossover, replacing an iterative top-`k`
solver with a direct one is a constant-factor speedup of the ratio measured
above. **The crossover has not been measured and must not be assumed.**

**Control:** the exact computation, at identical accuracy, on identical hardware.

**Novelty check:** NOT YET RUN. Libraries plausibly already gate on size; this
must be checked before the candidate is admitted, and if they do, the candidate
dies here.

**Fits the merge pattern:** yes — a correction to an existing hot path, zero new
public API, win stated as an exact ratio, determinism preserved because the
direct path is deterministic where the randomized one is not.

---

### C1 — exponent-derived adaptive rank for attention block selection

**Status:** rests on `tail_alpha` being width-invariant, which is **not yet
tested**. Plan Phase 1 Task 5 is the test. If the exponent drifts with `D`, this
candidate dies.

| field | content |
|---|---|
| topology | 0D persistence over key-block centroids to a causal CSR schedule, the construction inherited from `triton-lang/kernels#22` |
| differential geometry | the Jacobian spectrum at the selection point is the local linear model the budget must preserve |
| fractal geometry | `tail_alpha`, measured at 0.2374 +/- 0.0293 across blocks 1-5, cv 0.124 |
| computational mathematics | approximation theory — for `sigma_i ~ i^-a`, the rank needed for relative error `eps` follows in closed form, giving a budget rather than a hyperparameter |
| branch optimized | sparse attention scheduling: replace a tuned constant density with a density derived from the measured exponent |

**Measured basis:** `tail_alpha` over blocks 1-5, grounded `0.2374 +/- 0.0293`,
shuffled control `0.2342 +/- 0.0481`, differing by 1.4%. Block 0 is an outlier at
`0.6168` and was excluded **after** inspecting the data — a forking path that
Plan Task 5 must re-derive on held-out text before this is reported as a finding.

**Control:** random selection at matched density, locality-only at matched
density, and oracle top-`k` from dense scores. The position between random and
oracle is the result; a speedup number alone is not.

**Novelty check:** NOT YET RUN.

---

### C3 — single-linkage clustering materializes O(n²) to compute an MST

**Status:** strongest candidate in the ledger. The identity is textbook, the hot
path is real, and the replacement algorithm is already present in the same
package. `scipy` is not on the off-limits list.

**Hot path:** `scipy/cluster/_hierarchy.pyx`, `mst_single_linkage` / `nn_chain`.

| field | content |
|---|---|
| topology | single-linkage hierarchical clustering **is** 0-dimensional persistent homology: merge heights are H₀ death times and the dendrogram is the barcode. The structure is an ultrametric, equivalently a merge partial order |
| differential geometry | the metric on the point cloud is what the MST is taken with respect to; the result is invariant to any isometry of it |
| fractal geometry | merge heights across scales are the H₀ barcode, whose decay is the cloud's scaling structure |
| computational mathematics | graph algorithms — the single-linkage dendrogram equals the Euclidean minimum spanning tree, computable by dual-tree Borůvka on a KD-tree in roughly O(n log n) rather than by scanning a materialized condensed distance matrix |
| branch optimized | hierarchical clustering on large point clouds |

**Current cost:** consumes a fully materialized condensed distance matrix —
O(n²) time and O(n²) memory. This is what exhausts memory above roughly n = 50,000.

**Predicted win:** memory O(n²) → O(n); the dendrogram is **identical**, not
approximate, because the MST determines single-linkage exactly. `scipy.spatial.cKDTree`
already exists in the same package, so no new dependency is required.

**Control:** the existing implementation, asserting dendrogram equality — not
similarity — on every input where the old path still fits in memory.

**Determinism:** preserved. Ties in the MST must be broken by the same rule the
current code uses, and that is the one place this can silently diverge. A
tie-heavy regression fixture is mandatory. This is the exact failure the author's
own `R11` found: a parity claim verified on continuous inputs only, which then
failed 123 of 200 tie-heavy cases.

**Novelty check:** NOT YET RUN. `hdbscan` and `sklearn` already use dual-tree
Borůvka, so the algorithm is not novel — the candidate is that `scipy` still does
not. Verify the current source before proposing.

**Fits the merge pattern:** yes, precisely — a correction to an existing hot path,
zero new public API, win as an exact complexity change, identical output.

---

### C4 — physics islands rebuilt from scratch every step

**Status:** highest confidence of any entry, because the identical change already
merged as `google-deepmind/mujoco#3396`. This is the same move in a different
engine.

**Hot path:** `jrouwe/JoltPhysics`, `PhysicsSystem::BuildIslands`.

| field | content |
|---|---|
| topology | simulation islands are the connected components of the contact graph — H₀ of the contact complex, exactly the object mujoco#3396 replaced a dense n×n scratch buffer with |
| differential geometry | contacts are determined by the collision manifold; the graph is its combinatorial shadow |
| fractal geometry | contact-graph component sizes are heavy-tailed in granular piles, so most steps change very few components — which is what makes incremental work pay |
| computational mathematics | disjoint-set forest with union by rank and path compression, giving inverse-Ackermann amortized cost, plus a rollback structure for contact removal |
| branch optimized | rigid-body simulation step, island discovery |

**Current cost:** islands rebuilt from scratch every step, O(contacts) per step
regardless of how little changed.

**Predicted win:** incremental union-find with rollback on contact removal. The
formula to state is the fraction of steps in which the component partition is
unchanged; on typical stacking scenes that fraction is high, and it is the number
a reviewer will ask for.

**Control:** the existing rebuild, asserting identical island assignment every
step over a long trajectory, plus wall-clock at matched scene and seed.

**Determinism:** JoltPhysics guarantees deterministic simulation, so island
*ordering* — not just membership — must be preserved. This is a merge-blocker if
unanswered, and it is where mujoco#3396's framing is worth copying directly.

**Novelty check:** NOT YET RUN. Verify `BuildIslands` still rebuilds
unconditionally in current source before writing anything.

---

### C5 — quantization type chosen by filename, never by the tensor

**Hot path:** `ggml-org/llama.cpp`, `llama_tensor_get_type()` in `src/llama-quant.cpp`.

| field | content |
|---|---|
| topology | none directly — this candidate uses two of the three geometries, and that is stated rather than padded |
| differential geometry | the singular-value spectrum of the weight matrix is the local linear structure the quantizer must preserve |
| fractal geometry | singular-value decay is a power law; the retained-energy fraction `E_r = Σ_{i≤r} σ_i² / Σ_i σ_i²` is the scale-invariant quantity that should set precision |
| computational mathematics | numerical linear algebra — randomized SVD to get the spectrum at a cost amortized over a one-time quantization pass |
| branch optimized | model quantization: per-tensor precision selection |

**Current cost:** the type is selected by a hardcoded name-and-layer-index recipe,
`if (name.find("attn_v.weight") != std::string::npos) ...`, which never measures
a property of the tensor it is quantizing.

**Predicted win:** replace the recipe with a measured gate on the tensor's own
spectrum, falling back to higher precision when the gate rejects. Quantization is
a one-time offline pass, so an SVD per tensor is affordable where it would not be
at inference.

**Control:** the existing recipe, at matched total model size, measuring
perplexity. Matched *size* is the whole point — a gate that improves quality by
spending more bits has proved nothing.

**Novelty check:** NOT YET RUN.

---

### C6 — spectral clustering materializes a dense affinity before any solver runs

**Hot path:** `scikit-learn/scikit-learn`, `sklearn/manifold/_spectral_embedding.py`
and `sklearn/cluster/_spectral.py`.

| field | content |
|---|---|
| topology | the connectivity relation of the k-nearest-neighbour graph, which is what the embedding actually depends on |
| differential geometry | the normalized Laplacian `D^{-1/2} A D^{-1/2}` is the discrete Laplace–Beltrami operator of the underlying manifold |
| fractal geometry | eigenvalue decay sets how many eigenvectors are needed; the spectral gap is the scale-invariant stopping rule |
| computational mathematics | bounded subspace iteration with Rayleigh–Ritz on a sparse operator, instead of a dense decomposition |
| branch optimized | spectral clustering and spectral embedding |

**Current cost:** `SpectralClustering(affinity='rbf')` materializes a dense n×n
affinity via `pairwise_kernels` **before** any solver runs, so memory is O(n²)
regardless of which `eigen_solver` is selected.

**Predicted win:** sparse kNN adjacency plus subspace iteration returns the same
eigenvectors without the dense matrix.

**Control:** existing implementation, asserting subspace agreement to a stated
tolerance rather than eigenvector equality — eigenvectors are only defined up to
sign and up to rotation within degenerate eigenspaces, and a test that asserts
elementwise equality will fail for a correct implementation.

**Novelty check:** NOT YET RUN. `affinity='nearest_neighbors'` already exists;
the candidate is specifically the dense default path.

---

## Killed

### K1 — J-space summary statistics as a hallucination detector

Killed by its own control at the first measurement, and recorded here so it is
not proposed again.

Six blocks, ten token positions, one 61-token passage against its shuffled-token
control. Separation as a fraction of the grounded value:

```
sigma_max     per-layer [ 0.123 -0.012 -0.135 -0.123  0.233 -0.035]  |mean| 0.0085
stable_rank   per-layer [ 0.346  0.165  0.196  0.091 -0.299  0.184]  |mean| 0.1138
tail_alpha    per-layer [ 0.051 -0.088 -0.091 -0.037  0.067  0.227]  |mean| 0.0217
logdet_abs    per-layer [ 0.805  0.187  0.002 -0.111 -0.282 -0.273]  |mean| 0.0548
```

Signs flip layer to layer for three of the four. Only `stable_rank` holds sign at
5 of 6 layers, median +17.5%, and block 4 reverses it.

**Salvage:** `tail_alpha` agreeing to 1.4% between grounded and shuffled text is
what makes it a *structural* constant rather than a content signal. That failure
is precisely what qualifies it for C1. The negative result created the candidate.

**Caveat that keeps this open rather than closed:** a shuffled-token control is an
out-of-distribution control, not a hallucination control. Plan Phase 1 Task 4
runs the real test against factual-error labels, with Mahalanobis, PCA and
max-softmax baselines. Until that runs, "J-space does not detect hallucination"
is supported only against scrambling.

---

## Deferred

### D1 — forward-mode AD is unimplemented for fused attention

`torch.func.jvp` raises `NotImplementedError` on
`_scaled_dot_product_efficient_attention`, so any JVP-based method must load the
model with `attn_implementation="eager"` and give up the fused kernel. Reverse
mode is unaffected.

Real, measured, and a genuine obstacle to scaling J-space methods to large
models. Deferred rather than admitted because the natural upstream target is
off-limits.
