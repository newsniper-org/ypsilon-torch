"""Spatial modeling components for AFBO-style bilinear MLPs.

A spatial modeling (SM) block is a per-channel operation applied along the
token-sequence axis of an input of shape ``(B, N, C)``. It captures local
spatial dependencies between neighbouring tokens without mixing channels.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


SMKind = Literal["dwconv", "pool", "none"]


class SpatialModeling(nn.Module):
    """Per-channel token-sequence modeling.

    Parameters
    ----------
    dim:
        Number of channels ``C`` in the input.
    kind:
        Which spatial operator to apply:
        - ``"dwconv"``: depthwise 1D convolution with ``groups=dim``.
        - ``"pool"``: 1D average pooling (stride 1).
        - ``"none"``: identity.
    kernel_size:
        Kernel size for ``"dwconv"`` / ``"pool"``. Ignored for ``"none"``.
    bias:
        Whether the depthwise convolution uses a bias term.
    """

    def __init__(
        self,
        dim: int,
        kind: SMKind = "dwconv",
        kernel_size: int = 3,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.dim = dim
        self.kind = kind
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        if kind == "dwconv":
            self.op: nn.Module = nn.Conv1d(
                dim, dim, kernel_size=kernel_size,
                padding=padding, groups=dim, bias=bias,
            )
        elif kind == "pool":
            self.op = nn.AvgPool1d(
                kernel_size=kernel_size, stride=1, padding=padding,
            )
        elif kind == "none":
            self.op = nn.Identity()
        else:
            raise ValueError(
                f"Unknown spatial modeling kind {kind!r}; "
                f"expected one of 'dwconv', 'pool', 'none'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial modeling.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, N, C)``.
        """
        if self.kind == "none":
            return x
        # (B, N, C) -> (B, C, N) for Conv1d / Pool1d, then back.
        x_t = x.transpose(1, 2)
        y_t = self.op(x_t)
        return y_t.transpose(1, 2).contiguous()
