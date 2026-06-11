"""Hourglass-shaped feed-forward block (wide-narrow-wide FFN stack).

Reference
---------
- Liao, Chen, Yi, Shiu, *Revisiting the Shape Convention of Transformer
  Language Models* (arXiv:2602.06471) — Transformer FFN as a stack of ``K``
  residual hourglass sub-MLPs.
- Chen, Lee, Liao, Shiu, *Rethinking the Shape Convention of an MLP*
  (arXiv:2510.01796, OpenReview ``bUtLHJn90a``) — the original wide-narrow-wide
  formulation with an optional **fixed random** input projection.

The conventional Transformer FFN widens once (``d_model -> d_ff``, with
``d_ff > d_model``) and narrows back. The hourglass FFN instead **narrows**
into a bottleneck ``d_h < d_model`` and refines through ``K`` residual
sub-MLPs, spending the parameters saved on depth rather than width.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NormKind = Literal["layernorm", "rmsnorm", "none"]


def _make_norm(kind: NormKind, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "rmsnorm":
        return nn.RMSNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r}")


class _HourglassSubMLP(nn.Module):
    """A single SwiGLU bottleneck sub-MLP: ``W_u (SiLU(W_d1 x̄) ⊙ (W_d2 x̄))``."""

    def __init__(self, dim: int, hidden_dim: int, norm: NormKind, bias: bool) -> None:
        super().__init__()
        self.norm = _make_norm(norm, dim)
        self.w_d1 = nn.Linear(dim, hidden_dim, bias=bias)   # down-projection (gate)
        self.w_d2 = nn.Linear(dim, hidden_dim, bias=bias)   # down-projection (value)
        self.w_u = nn.Linear(hidden_dim, dim, bias=bias)    # up-projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self.norm(x)
        return self.w_u(F.silu(self.w_d1(xb)) * self.w_d2(xb))


class HourglassFFN(nn.Module):
    r"""Stack of ``K`` residual hourglass sub-MLPs (drop-in ViT/LM FFN).

    .. math::

        h_0 = x, \qquad
        h_{i+1} = h_i + W_u^{(i)}\!\big(
            \operatorname{SiLU}(W_{d1}^{(i)} \bar h_i)
            \odot (W_{d2}^{(i)} \bar h_i)\big),
        \qquad \bar h_i = \operatorname{norm}(h_i)

    with bottleneck ``d_h < d_model`` and output ``h_K``.

    Parameters
    ----------
    dim : int
        Model dimension ``d_model`` (input == output).
    bottleneck_ratio : float
        ``d_h / d_model``. Recommended ``0.4``–``0.6`` (the hourglass regime).
        Values ``>= 1`` are allowed but defeat the purpose.
    depth : int
        Number of stacked sub-MLPs ``K``. Recommended ``2``–``4``.
    norm : {"layernorm", "rmsnorm", "none"}
        Per-sub-MLP pre-normalization. Default ``"layernorm"``.
    bias : bool
        Whether the linear layers use bias. Default ``False``.
    hidden_dim : int, optional
        Explicit bottleneck size, overriding ``bottleneck_ratio``.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_ratio: float = 0.5,
        depth: int = 4,
        norm: NormKind = "layernorm",
        bias: bool = False,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if hidden_dim is None:
            hidden_dim = max(1, int(round(dim * bottleneck_ratio)))

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            _HourglassSubMLP(dim, hidden_dim, norm=norm, bias=bias)
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = h + block(h)
        return h

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}, depth={self.depth}"
