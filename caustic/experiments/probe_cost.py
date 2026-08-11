"""Feasibility probe for J-space work on a live small LM.

Measures the real cost of the three Jacobian estimators the research plan needs,
on distilgpt2 (D=768, 6 blocks), so the plan is sized against wall-clock rather
than against a guess:

  1. one forward pass through a single block          -- the unit of cost
  2. one JVP (forward-mode, one tangent direction)    -- cost of one column
  3. top-k singular values by power iteration on JVP/VJP
  4. full D x D Jacobian by jacrev                    -- the thing we cannot afford

The object under study is the layer-to-layer transport map at one token position:

    J_l(t) = d h_{l+1}[t] / d h_l[t]        in R^{D x D}

which is the causal operator: it is what carries a perturbation forward, as
opposed to a point cloud of hidden states, which is an observable of the past.

Everything runs offline against the locally cached model.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "distilgpt2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
LAYER = 3
PROMPT = "The capital of France is Paris, and the capital of Italy is"


def timed(fn, n: int, warmup: int = 3) -> float:
    """Median seconds per call over n calls, after warmup, with CUDA synced."""
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    # eager attention is required: forward-mode AD (jvp) is not implemented for
    # the fused _scaled_dot_product_efficient_attention kernel that transformers
    # selects by default. Reverse mode works either way.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, attn_implementation="eager"
    ).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
    T = ids.shape[1]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states
    D = hs[0].shape[-1]
    block = model.transformer.h[LAYER]
    h_in = hs[LAYER].detach()  # (1, T, D)

    print(f"model={MODEL} device={DEV} D={D} T={T} blocks={len(model.transformer.h)} layer={LAYER}")

    def block_fwd(h):
        # transformers >=5 returns a bare tensor here; older versions a tuple.
        out = block(h)
        return out[0] if isinstance(out, tuple) else out

    # Restrict to the last token's row of the block output: R^D -> R^D.
    # The block is causal, so position T-1 depends on all of h, but the square
    # Jacobian we want is the diagonal block d h_out[T-1] / d h_in[T-1].
    def f_last(h_row):
        h = h_in.clone()
        h[0, -1, :] = h_row
        return block_fwd(h)[0, -1, :]

    h_row = h_in[0, -1, :].clone()

    # --- 1. block forward -------------------------------------------------
    t_fwd = timed(lambda: block_fwd(h_in), n=30)

    # --- 2. single JVP ----------------------------------------------------
    v = torch.randn(D, device=DEV, dtype=DTYPE)
    v = v / v.norm()

    def one_jvp():
        return torch.func.jvp(f_last, (h_row,), (v,))[1]

    t_jvp = timed(one_jvp, n=30)

    # --- 3. top-k singular values by power iteration on J^T J -------------
    def jvp(u):
        return torch.func.jvp(f_last, (h_row,), (u,))[1]

    def vjp(w):
        _, pull = torch.func.vjp(f_last, h_row)
        return pull(w)[0]

    def top_singular(k: int, iters: int = 20) -> np.ndarray:
        """Block power iteration on J^T J with re-orthonormalization."""
        Q = torch.linalg.qr(torch.randn(D, k, device=DEV, dtype=DTYPE))[0]
        for _ in range(iters):
            W = torch.stack([jvp(Q[:, j]) for j in range(k)], dim=1)
            Z = torch.stack([vjp(W[:, j]) for j in range(k)], dim=1)
            Q = torch.linalg.qr(Z)[0]
        W = torch.stack([jvp(Q[:, j]) for j in range(k)], dim=1)
        return torch.linalg.svdvals(W).detach().cpu().numpy()

    K, ITERS = 8, 20
    t0 = time.perf_counter()
    sv_iter = top_singular(K, ITERS)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t_pow = time.perf_counter() - t0

    # --- 4. full Jacobian by jacrev (ground truth + cost ceiling) ---------
    t0 = time.perf_counter()
    J = torch.func.jacrev(f_last)(h_row)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t_full = time.perf_counter() - t0
    sv_true = torch.linalg.svdvals(J.float()).detach().cpu().numpy()

    # --- report ------------------------------------------------------------
    print()
    print(f"block forward           {t_fwd * 1e3:8.3f} ms")
    print(f"one JVP                 {t_jvp * 1e3:8.3f} ms   ({t_jvp / t_fwd:.2f}x forward)")
    print(f"top-{K} power ({ITERS} it)  {t_pow * 1e3:8.3f} ms   ({t_pow / t_fwd:.1f}x forward)")
    print(f"full {D}x{D} jacrev    {t_full * 1e3:8.3f} ms   ({t_full / t_fwd:.1f}x forward)")
    print()
    print(f"top-{K} sv, power iter : {np.array2string(sv_iter, precision=4)}")
    print(f"top-{K} sv, exact      : {np.array2string(sv_true[:K], precision=4)}")
    rel = np.abs(sv_iter - sv_true[:K]) / np.maximum(sv_true[:K], 1e-12)
    print(f"max rel err on top-{K}  : {rel.max():.3e}")
    print()
    print(f"sigma_max               {sv_true[0]:.4f}   (>1 means this block expands)")
    print(f"sigma_min               {sv_true[-1]:.6e}")
    print(f"cond                    {sv_true[0] / max(sv_true[-1], 1e-30):.4e}")
    print(f"stable rank             {(sv_true**2).sum() / sv_true[0]**2:.2f}  of {D}")
    eff = sv_true / sv_true.sum()
    print(f"spectral entropy        {-(eff * np.log(np.maximum(eff, 1e-30))).sum():.4f}  (ln D = {np.log(D):.4f})")
    n_tiny = int((sv_true < 1e-6 * sv_true[0]).sum())
    print(f"sv below 1e-6*sigma_max {n_tiny} of {D}   <- numerical rank deficiency = folding candidate")

    np.save("jspace_probe_spectrum.npy", sv_true)
    print("\nfull spectrum saved to jspace_probe_spectrum.npy")


if __name__ == "__main__":
    main()
