from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_float_dtype

from ._nd_pool_base import NDPoolBase


def _channel_view(param: Tensor, ndim: int) -> Tensor:
    """Reshape ``(C,)`` → ``(1, C, 1, …, 1)`` with *ndim* total dims."""
    shape = [1] * ndim
    shape[1] = -1
    return param.view(*shape)


class AsinhAvgPoolND(NDPoolBase):
    r"""N-dimensional pooling via SinhMeanArsinh.

    Each channel has a learnable scale parameter :math:`\alpha_c`.

    Parameters
    ----------
    channels : int
        Number of input channels.
    kernel_size, stride, padding
        Passed to :class:`NDPoolBase`.
    alpha_init : float
        Initial value for the per-channel :math:`\alpha`. Default ``1.0``.
    dtype_idx : FPDTypeIdx
        Floating-point precision index. Default ``64``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        alpha_init: float = 1.0,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__(kernel_size, stride, padding)
        self.alpha: nn.Parameter = nn.Parameter(
            torch.full((channels,), alpha_init, dtype=get_float_dtype(dtype_idx))
        )

    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        # Compute inline to handle dimension change across mean reduction.
        alpha_w = _channel_view(self.alpha, windows.ndim)
        z = torch.asinh(windows / alpha_w)
        mean_z = z.mean(dim=dim)
        alpha_r = _channel_view(self.alpha, mean_z.ndim)
        return alpha_r * torch.sinh(mean_z)
