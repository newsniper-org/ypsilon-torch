from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_float_dtype

from ._complex_nd_pool_base import ComplexNDPoolBase


def _channel_view(param: Tensor, ndim: int) -> Tensor:
    shape = [1] * ndim
    shape[1] = -1
    return param.view(*shape)


def _circular_mean_phase(z: Tensor, dim: int) -> Tensor:
    """Compute the circular mean of phases along *dim*.

    Returns real-valued angles in radians.
    """
    return torch.atan2(
        z.imag.sum(dim=dim),
        z.real.sum(dim=dim),
    )


class ComplexAsinhAvgPoolND(ComplexNDPoolBase):
    r"""Complex N-dimensional pooling: SinhMeanArsinh on magnitude + circular mean phase.

    Applies :func:`sinh_mean_arsinh` to the **magnitude** of each complex
    window element and reconstructs the output using the circular-mean phase
    of the window.

    Parameters
    ----------
    channels : int
        Number of input channels.
    kernel_size, stride, padding
        Passed to :class:`ComplexNDPoolBase`.
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
        # Magnitude pooling via SinhMeanArsinh
        mag = windows.abs()                                     # real
        alpha_w = _channel_view(self.alpha, mag.ndim)
        u = torch.asinh(mag / alpha_w)
        mean_u = u.mean(dim=dim)
        alpha_r = _channel_view(self.alpha, mean_u.ndim)
        pooled_mag = alpha_r * torch.sinh(mean_u)

        # Phase: circular mean
        phase = _circular_mean_phase(windows, dim=dim)

        return torch.polar(pooled_mag, phase)
