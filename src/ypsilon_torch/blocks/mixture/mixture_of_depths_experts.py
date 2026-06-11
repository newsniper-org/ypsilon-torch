"""Mixture-of-Depths-and-Experts (MoDE) — depth routing combined with MoE.

Reference
---------
Raposo et al., *Mixture-of-Depths* (arXiv:2404.02258), §4. Two variants:

- **integrated**: a single token-choice router over ``num_experts`` real
  experts plus one extra **no-op (identity) expert**. Routing a token to the
  no-op slot is exactly an MoD bypass — depth and expert selection share one
  routing operation. (The paper finds this variant preferable.)
- **staged**: an MoD router first decides which tokens are processed at all,
  then a separate MoE router assigns each surviving token to an expert.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._aux_loss import AuxLossModule
from ._swiglu import SwiGLUFeedForward
from .mixture_of_depths import MixtureOfDepths


class _Top1MoE(AuxLossModule):
    """Token-choice top-1 MoE with an optional no-op expert and Switch loss."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        expert_hidden: Optional[int],
        expansion: float,
        aux_weight: float,
        include_noop: bool,
        residual: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.include_noop = include_noop
        self.residual = residual
        self.aux_weight = float(aux_weight)

        self.experts = nn.ModuleList(
            SwiGLUFeedForward(dim, hidden_dim=expert_hidden, expansion=expansion)
            for _ in range(num_experts)
        )
        self.n_slots = num_experts + (1 if include_noop else 0)
        self.router = nn.Linear(dim, self.n_slots, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.router(x), dim=-1)        # (B, N, n_slots)
        top_p, top_i = probs.max(dim=-1)                 # (B, N)

        # Switch-style load-balancing loss over all slots (incl. no-op).
        if self.training and self.aux_weight != 0.0:
            one_hot = F.one_hot(top_i, self.n_slots).to(probs.dtype)
            f = one_hot.mean(dim=(0, 1))                 # dispatch fraction per slot
            p = probs.mean(dim=(0, 1))                   # mean prob per slot
            self.reg_loss = self.aux_weight * self.n_slots * (f * p).sum()
        else:
            self.reg_loss = 0.0

        transformed = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = top_i == e                            # (B, N) -> tokens for expert e
            if mask.any():
                xe = x[mask]                             # (M, D)
                ye = self.experts[e](xe)
                transformed[mask] = top_p[mask].unsqueeze(-1) * ye
        # tokens routed to the no-op slot keep ``transformed == 0``.

        return x + transformed if self.residual else transformed


class MixtureOfDepthsAndExperts(AuxLossModule):
    r"""MoD combined with a sparse MoE (integrated or staged).

    Parameters
    ----------
    dim : int
        Token dimension.
    num_experts : int
        Number of real experts (the integrated variant adds one no-op slot).
    variant : {"integrated", "staged"}
        See module docstring. Default ``"integrated"`` (paper-preferred).
    capacity_ratio : float
        Token fraction kept by the MoD stage (``"staged"`` only). Default
        ``0.125``.
    expert_hidden : int, optional
        Expert FFN width. Defaults to ``expansion * dim``.
    expansion : float
        Expert width multiplier when ``expert_hidden`` is not given.
    aux_weight : float
        Weight of the MoE load-balancing loss (and, for ``"staged"``, the MoD
        predictor loss). Exposed through the ``reg_loss`` property.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        variant: Literal["integrated", "staged"] = "integrated",
        capacity_ratio: float = 0.125,
        expert_hidden: Optional[int] = None,
        expansion: float = 4.0,
        aux_weight: float = 1e-2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.variant = variant

        if variant == "integrated":
            self.moe = _Top1MoE(
                dim, num_experts, expert_hidden, expansion,
                aux_weight=aux_weight, include_noop=True, residual=True,
            )
            self.mod = None
        elif variant == "staged":
            moe = _Top1MoE(
                dim, num_experts, expert_hidden, expansion,
                aux_weight=aux_weight, include_noop=False, residual=False,
            )
            self.moe = moe
            self.mod = MixtureOfDepths(
                block=moe, dim=dim, capacity_ratio=capacity_ratio,
                aux_weight=aux_weight,
            )
        else:
            raise ValueError(f"unknown variant: {variant!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "integrated":
            return self.moe(x)
        return self.mod(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, variant={self.variant}"
