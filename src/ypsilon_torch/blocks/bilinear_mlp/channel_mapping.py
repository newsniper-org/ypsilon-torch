"""Asymmetric channel-mapping (CM) components for AFBO-style bilinear MLPs.

Two structured-sparsity channel-mapping strategies are provided:

- :class:`GroupedCrossChannelMapping` (GCCM): non-overlapping channel groups
  + cyclic channel shuffle for cross-group information flow.
- :class:`OverlappedCycleChannelMapping` (OCCM): cyclically overlapping channel
  groups so each group sees its neighbours' channels directly.

Both modules act as drop-in replacements for ``nn.Linear(in_channels,
out_channels)`` on the last dimension of their input.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _check_divisible(name: str, value: int, divisor: int) -> None:
    if value % divisor != 0:
        raise ValueError(
            f"{name} (={value}) must be divisible by num_groups (={divisor})."
        )


class GroupedCrossChannelMapping(nn.Module):
    """Grouped 1x1 linear mapping followed by a cyclic channel shuffle.

    For ``G`` groups, the input channels are partitioned into ``G`` disjoint
    slices of size ``C_in // G`` and each slice is mapped independently to
    ``C_out // G`` output channels. A cyclic channel shuffle then permutes the
    output channels so that subsequent layers (or the paired bilinear branch)
    can mix information across groups.

    Parameters
    ----------
    in_channels:
        Size of the input feature dimension ``C_in``.
    out_channels:
        Size of the output feature dimension ``C_out``.
    num_groups:
        Number of channel groups ``G``. Must divide both ``in_channels`` and
        ``out_channels``.
    bias:
        Whether to include an additive bias after the grouped mapping.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        _check_divisible("in_channels", in_channels, num_groups)
        _check_divisible("out_channels", out_channels, num_groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.group_in = in_channels // num_groups
        self.group_out = out_channels // num_groups

        # Weight shape (G, group_in, group_out); einsum-friendly.
        self.weight = nn.Parameter(
            torch.empty(num_groups, self.group_in, self.group_out)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming-uniform matching nn.Linear's default (fan_in = group_in).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.group_in
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the grouped-cross channel mapping.

        Parameters
        ----------
        x:
            Tensor of shape ``(..., C_in)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., C_out)``.
        """
        *lead, c_in = x.shape
        if c_in != self.in_channels:
            raise ValueError(
                f"Expected last dim {self.in_channels}, got {c_in}."
            )
        x_g = x.reshape(*lead, self.num_groups, self.group_in)
        # (..., G, group_in) x (G, group_in, group_out) -> (..., G, group_out)
        y_g = torch.einsum("...gi,gio->...go", x_g, self.weight)
        # Cyclic channel shuffle: transpose group / channel axes so that
        # output channel k of group g is mapped to a new position that
        # interleaves across groups. Identical to ShuffleNet's channel
        # shuffle operation.
        y_g = y_g.transpose(-2, -1).contiguous()
        y = y_g.reshape(*lead, self.out_channels)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"num_groups={self.num_groups}, bias={self.bias is not None}"
        )


class OverlappedCycleChannelMapping(nn.Module):
    """Grouped linear mapping with cyclically overlapping input windows.

    Each of the ``G`` output groups of size ``C_out // G`` reads from a
    widened input slice of size ``group_in + 2 * overlap`` taken cyclically
    from the input channels. The overlap introduces local cross-group
    connectivity directly at the weight level, complementing GCCM.

    Parameters
    ----------
    in_channels:
        Size of the input feature dimension ``C_in``.
    out_channels:
        Size of the output feature dimension ``C_out``.
    num_groups:
        Number of channel groups ``G``. Must divide both ``in_channels`` and
        ``out_channels``.
    overlap:
        Number of extra input channels each group reads from each of its two
        cyclic neighbours. Must satisfy ``2 * overlap <= group_in``.
    bias:
        Whether to include an additive bias after the mapping.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        overlap: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        _check_divisible("in_channels", in_channels, num_groups)
        _check_divisible("out_channels", out_channels, num_groups)
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.overlap = overlap
        self.group_in = in_channels // num_groups
        self.group_out = out_channels // num_groups
        if 2 * overlap > self.group_in:
            raise ValueError(
                f"2*overlap (={2*overlap}) must not exceed group_in "
                f"(={self.group_in})."
            )
        self.window = self.group_in + 2 * overlap

        # Precompute the cyclic gather index of shape (G, window).
        gather_idx = torch.empty(num_groups, self.window, dtype=torch.long)
        for g in range(num_groups):
            start = g * self.group_in - overlap
            for w in range(self.window):
                gather_idx[g, w] = (start + w) % in_channels
        self.register_buffer("gather_idx", gather_idx, persistent=False)

        self.weight = nn.Parameter(
            torch.empty(num_groups, self.window, self.group_out)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.window
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the overlapped-cycle channel mapping.

        Parameters
        ----------
        x:
            Tensor of shape ``(..., C_in)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., C_out)``.
        """
        *lead, c_in = x.shape
        if c_in != self.in_channels:
            raise ValueError(
                f"Expected last dim {self.in_channels}, got {c_in}."
            )
        # Gather overlapping windows: x[..., gather_idx] -> (..., G, window)
        x_g = x.index_select(dim=-1, index=self.gather_idx.reshape(-1))
        x_g = x_g.reshape(*lead, self.num_groups, self.window)
        # (..., G, window) x (G, window, group_out) -> (..., G, group_out)
        y_g = torch.einsum("...gw,gwo->...go", x_g, self.weight)
        y = y_g.reshape(*lead, self.out_channels)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"num_groups={self.num_groups}, overlap={self.overlap}, "
            f"bias={self.bias is not None}"
        )
