"""Mixture-of-Hidden-Dimensions (MoHD) feed-forward block.

Reference
---------
Chen et al., *Mixture of Hidden-Dimensions: Not All Hidden-States' Dimensions
are Needed in Transformer* (ICML 2025, arXiv:2412.05644).

Where Mixture-of-Experts splits an FFN into expert *sub-networks*, MoHD splits
the **hidden dimension** into ``N`` sub-dimension groups and conditionally
activates them per token: a fixed fraction is *shared* (always on) and the rest
are *specialized* (top-``k`` routed). Sparsity-induced information loss is
restored by (a) a per-token scaling factor ``α`` and (b) a Grouped Fusion
layer.

Implementation note
-------------------
The paper's Grouped Fusion uses a Monarch matrix for ``O(d/r)`` cost. Here it is
realised as a **block-diagonal grouped linear** over the sub-dimension groups —
the same cheap, structured re-mixing role, without the Monarch permutation
machinery. Pass ``fusion=False`` to drop it entirely.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._aux_loss import AuxLossModule


class _GroupedFusion(nn.Module):
    """Block-diagonal linear: each of ``groups`` blocks mixed within itself."""

    def __init__(self, dim: int, groups: int, bias: bool = True) -> None:
        super().__init__()
        if dim % groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by groups ({groups})")
        self.groups = groups
        self.block = dim // groups
        self.weight = nn.Parameter(torch.empty(groups, self.block, self.block))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, d = x.shape
        xg = x.reshape(*lead, self.groups, self.block)
        out = torch.einsum("...gb,gbc->...gc", xg, self.weight).reshape(*lead, d)
        if self.bias is not None:
            out = out + self.bias
        return out


class MixtureOfHiddenDimensions(AuxLossModule):
    r"""FFN whose hidden dimension is conditionally activated by sub-dim groups.

    The intermediate width ``d_ff`` is split into ``N`` sub-dimension groups of
    size ``d_e = d_ff / N``. A router scores each group from the token; the
    first ``shared`` groups are always kept, and ``(active - shared)`` of the
    remaining groups are top-``k`` selected. Gated groups are scaled by ``α``
    and re-mixed by the grouped fusion before the activation and down-projection.

    Parameters
    ----------
    dim : int
        Model dimension (input == output).
    hidden_dim : int, optional
        Intermediate width ``d_ff``. Defaults to ``expansion * dim``.
    expansion : float
        Width multiplier when ``hidden_dim`` is not given. Default ``4.0``.
    num_subdims : int
        Number of sub-dimension groups ``N``. Default ``16`` (paper's best).
        Must divide ``hidden_dim``.
    active_ratio : float
        Fraction ``δ`` of groups active per token (shared + specialized).
        Default ``0.5``.
    shared_ratio : float
        Fraction ``φ`` of groups that are always-on shared groups. Must satisfy
        ``0 <= shared_ratio <= active_ratio``. Default ``0.375`` (≈ 3/4 of the
        active budget shared, the paper's recommendation).
    fusion : bool
        Whether to apply the grouped fusion layer. Default ``True``.
    aux_weight : float
        Weight of the load-balancing loss exposed via the ``reg_loss`` property.
    bias : bool
        Whether the projections use bias. Default ``False``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        expansion: float = 4.0,
        num_subdims: int = 16,
        active_ratio: float = 0.5,
        shared_ratio: float = 0.375,
        fusion: bool = True,
        aux_weight: float = 1e-2,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(round(expansion * dim))
        if hidden_dim % num_subdims != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_subdims ({num_subdims})"
            )
        if not (0.0 < active_ratio <= 1.0):
            raise ValueError(f"active_ratio must be in (0, 1], got {active_ratio}")
        if not (0.0 <= shared_ratio <= active_ratio):
            raise ValueError(
                f"shared_ratio must be in [0, active_ratio], got {shared_ratio}"
            )

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_subdims = num_subdims
        self.subdim = hidden_dim // num_subdims
        self.aux_weight = float(aux_weight)

        self.n_shared = int(round(shared_ratio * num_subdims))
        self.n_active = max(self.n_shared, int(round(active_ratio * num_subdims)))
        self.n_specialized = self.n_active - self.n_shared

        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        # Routing centroids φ_i: token -> affinity per sub-dimension group.
        self.centroids = nn.Parameter(torch.empty(num_subdims, dim))
        nn.init.normal_(self.centroids, std=dim ** -0.5)

        self.fusion = _GroupedFusion(hidden_dim, groups=num_subdims) if fusion else None

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-group gate ``g`` of shape ``(B, N, num_subdims)``."""
        s = F.softmax(x @ self.centroids.t(), dim=-1)        # (B, N, N_sub)

        gate = torch.zeros_like(s)
        # Shared groups: always on.
        if self.n_shared > 0:
            gate[..., : self.n_shared] = s[..., : self.n_shared]
        # Specialized groups: top-k within the remaining groups.
        if self.n_specialized > 0:
            spec = s[..., self.n_shared :]
            topv, topi = torch.topk(spec, self.n_specialized, dim=-1)
            gate[..., self.n_shared :].scatter_(-1, topi, topv)

        # Load-balancing loss over specialized groups (Switch-style).
        if self.training and self.aux_weight != 0.0 and self.n_specialized > 0:
            spec = s[..., self.n_shared :]
            chosen = (gate[..., self.n_shared :] > 0).to(s.dtype)
            f = chosen.mean(dim=(0, 1))
            p = spec.mean(dim=(0, 1))
            n_spec_groups = self.num_subdims - self.n_shared
            self.reg_loss = self.aux_weight * n_spec_groups * (f * p).sum()
        else:
            self.reg_loss = 0.0

        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        gate = self._compute_gate(x)                          # (B, N, N_sub)

        h = self.w_up(x).reshape(B, N, self.num_subdims, self.subdim)
        h = gate.unsqueeze(-1) * h                            # gated sub-dim groups

        # Activation-flow scaling α = (Σ g_i) · N, restoring activated magnitude.
        alpha = gate.sum(dim=-1, keepdim=True) * self.num_subdims  # (B, N, 1)
        h = h.reshape(B, N, self.hidden_dim) * alpha

        if self.fusion is not None:
            h = self.fusion(h)

        return self.w_down(F.silu(h))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden_dim={self.hidden_dim}, "
            f"num_subdims={self.num_subdims}, "
            f"active={self.n_active}, shared={self.n_shared}"
        )
