"""Shared SwiGLU feed-forward used as the expert / sub-block of mixture blocks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """A standard SwiGLU FFN: ``W_down (SiLU(W_gate x) ⊙ (W_up x))``.

    Maps ``(..., dim) -> (..., dim)``.

    Parameters
    ----------
    dim : int
        Input / output dimension.
    hidden_dim : int, optional
        Expansion width. Defaults to ``round(expansion * dim)``.
    expansion : float
        Width multiplier when ``hidden_dim`` is not given. Default ``4.0``.
    bias : bool
        Whether the linear layers use bias. Default ``False``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        expansion: float = 4.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(round(expansion * dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
