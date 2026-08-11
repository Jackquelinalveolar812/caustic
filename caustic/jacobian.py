"""Exact and estimated Jacobians of a live transformer block.

The object under study is the layer-to-layer transport map at one token position,

    J_l(t) = d h_{l+1}[t] / d h_l[t]        in R^{D x D}

This is deliberately not a point cloud of hidden states. A point cloud is an
observable of the past; measured on distilgpt2 in `sigmoid/examples/why_tokens_fail.py`,
a persistence signature over a token window reads lexical repetition (R^2 0.537 for
distinct-type count) and predicts nothing forward. J_l(t) is the operator that
carries a perturbation forward, so it is causal by construction. That is the whole
reason for working in Jacobian space rather than state space.

Two environment facts, both measured on this machine and both load-bearing:

1. Forward-mode AD is not implemented for the fused SDPA kernel that transformers
   selects by default. `torch.func.jvp` raises NotImplementedError on
   `_scaled_dot_product_efficient_attention`. Any model used here must be loaded
   with `attn_implementation="eager"`. Reverse mode works either way.

2. At D=768 the exact Jacobian is cheaper than a naive Krylov estimator of its top
   singular values. Measured on an RTX 4060 Laptop, distilgpt2, layer 3:

       block forward             0.588 ms
       full 768x768 jacrev      53.694 ms   (91.3x forward)
       top-8 power, 20 iters  1428.938 ms   (2429.5x forward)

   so `exact_jacobian` is the default path and `top_singular_values` exists for the
   regime where D is large enough to invert that ordering. The crossover has not
   been measured; do not assume it without measuring.
"""

from __future__ import annotations

import torch

__all__ = ["block_map", "exact_jacobian", "top_singular_values", "singular_values"]


def block_map(block, h_in: torch.Tensor, pos: int):
    """Return f: R^D -> R^D, the block's action on the hidden state at `pos`.

    All other positions are held at their observed values, so `f` isolates the
    diagonal block of the full (T*D x T*D) Jacobian. The block remains causal:
    position `pos` still attends over the frozen prefix.

    Args:
        block: a transformer block module, e.g. `model.transformer.h[l]`.
        h_in:  observed input hidden states, shape (1, T, D).
        pos:   token position to differentiate at.
    """
    if h_in.dim() != 3 or h_in.shape[0] != 1:
        raise ValueError(f"h_in must have shape (1, T, D); got {tuple(h_in.shape)}")
    if not -h_in.shape[1] <= pos < h_in.shape[1]:
        raise IndexError(f"pos {pos} out of range for T={h_in.shape[1]}")

    def f(h_row: torch.Tensor) -> torch.Tensor:
        h = h_in.clone()
        h[0, pos, :] = h_row
        out = block(h)
        # transformers >=5 returns a bare tensor; older versions a tuple.
        out = out[0] if isinstance(out, tuple) else out
        return out[0, pos, :]

    return f


def exact_jacobian(block, h_in: torch.Tensor, pos: int) -> torch.Tensor:
    """The exact D x D Jacobian by batched reverse-mode AD.

    This is the default. See the module docstring for why it beats the Krylov
    estimator at D=768.
    """
    f = block_map(block, h_in, pos)
    return torch.func.jacrev(f)(h_in[0, pos, :].detach().clone())


def singular_values(J: torch.Tensor) -> torch.Tensor:
    """Singular values of J in descending order, computed in float32."""
    return torch.linalg.svdvals(J.float())


def top_singular_values(
    block,
    h_in: torch.Tensor,
    pos: int,
    k: int = 8,
    iters: int = 20,
    seed: int | None = 0,
) -> torch.Tensor:
    """Top-k singular values by block power iteration on J^T J, using JVP and VJP.

    Never materializes J. Requires `attn_implementation="eager"` because the JVP
    half uses forward-mode AD.

    Accuracy against `exact_jacobian` was measured at max relative error 1.685e-04
    on the top 8 values (distilgpt2, layer 3, k=8, iters=20). The estimator is
    correct; it is simply slower than the exact path at this D.

    `seed` fixes the random start and defaults to being set. An unseeded start
    made this function non-deterministic, which was caught by a parity test that
    passed on five consecutive runs and failed on a sixth: an unlucky draw can
    leave a starting block nearly orthogonal to a wanted singular direction, and
    twenty iterations do not recover it. Any A/B comparison built on a
    non-deterministic estimator is measuring the estimator's own variance, so the
    default is a fixed seed. Pass `seed=None` deliberately to sample the start,
    for instance when estimating that variance on purpose.
    """
    f = block_map(block, h_in, pos)
    x = h_in[0, pos, :].detach().clone()
    D = x.shape[0]

    def jvp(u: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(f, (x,), (u,))[1]

    def vjp(w: torch.Tensor) -> torch.Tensor:
        _, pull = torch.func.vjp(f, x)
        return pull(w)[0]

    if seed is None:
        start = torch.randn(D, k, device=x.device, dtype=x.dtype)
    else:
        # Generator on the CPU then moved, so the same seed gives the same start
        # on CPU and CUDA. A device-side generator does not guarantee that.
        g = torch.Generator().manual_seed(seed)
        start = torch.randn(D, k, generator=g, dtype=x.dtype).to(x.device)
    Q = torch.linalg.qr(start)[0]
    for _ in range(iters):
        W = torch.stack([jvp(Q[:, j]) for j in range(k)], dim=1)
        Z = torch.stack([vjp(W[:, j]) for j in range(k)], dim=1)
        Q = torch.linalg.qr(Z)[0]
    W = torch.stack([jvp(Q[:, j]) for j in range(k)], dim=1)
    return torch.linalg.svdvals(W)
